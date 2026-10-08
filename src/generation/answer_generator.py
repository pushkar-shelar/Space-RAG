from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Dict, List, Optional


_SHARED_QWEN_MODEL = None
_SHARED_QWEN_TOKENIZER = None
_SHARED_QWEN_NAME = None


def get_shared_qwen(model_name: str = "Qwen/Qwen2.5-1.5B-Instruct"):
    global _SHARED_QWEN_MODEL, _SHARED_QWEN_TOKENIZER, _SHARED_QWEN_NAME
    if _SHARED_QWEN_MODEL is None or _SHARED_QWEN_NAME != model_name:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        _SHARED_QWEN_TOKENIZER = AutoTokenizer.from_pretrained(model_name)
        _SHARED_QWEN_MODEL = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="cpu",
        ).eval()
        _SHARED_QWEN_NAME = model_name
    return _SHARED_QWEN_MODEL, _SHARED_QWEN_TOKENIZER


def release_shared_qwen():
    global _SHARED_QWEN_MODEL, _SHARED_QWEN_TOKENIZER, _SHARED_QWEN_NAME
    _SHARED_QWEN_MODEL = None
    _SHARED_QWEN_TOKENIZER = None
    _SHARED_QWEN_NAME = None
    import gc
    gc.collect()


class AnswerGenerator:
    """
    Generate from retrieved evidence only.

    Source policy:
      - corpus evidence may be used but its provenance is never exposed.
      - private evidence must be accompanied by a document/page citation.
    """

    def __init__(
        self,
        offline_model: str = "Qwen/Qwen2.5-1.5B-Instruct",
        online_model: Optional[str] = None,
    ):
        self.offline_model_name = offline_model
        self.online_model_name = online_model or os.getenv("XAI_MODEL", "grok-2-latest")
        self._tokenizer = None
        self._model = None
        self._grok_client = None

    def _load_offline(self):
        if self._model is None or self._tokenizer is None:
            self._model, self._tokenizer = get_shared_qwen(self.offline_model_name)

    def _load_grok(self, api_key: Optional[str] = None):
        key = api_key or os.getenv("XAI_API_KEY")
        if not key:
            raise RuntimeError(
                "XAI_API_KEY is not configured. Provide it in the sidebar or set it in the environment before using Online Grok mode."
            )
        if self._grok_client is not None and getattr(self._grok_client, "api_key", None) == key:
            return
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError(
                "The 'openai' package is required for Online Grok mode. Run: pip install -U openai"
            ) from exc

        self._grok_client = OpenAI(
            api_key=key,
            base_url="https://api.x.ai/v1",
        )

    @staticmethod
    def _private_citation(result: dict) -> Optional[str]:
        if result.get("source_type") != "private":
            return None
        document = result.get("document") or "Private document"
        page = result.get("page_number")
        return f"[{document}, p. {page}]" if page else f"[{document}]"

    def _prepare_evidence(self, results: List[dict]) -> tuple[str, List[str]]:
        blocks = []
        citations = []

        private_seen = set()
        for idx, result in enumerate(results, start=1):
            source_type = result.get("source_type")
            text = (result.get("text") or "").strip()
            page = result.get("page_number")
            document = result.get("document")

            if source_type == "private":
                citation = self._private_citation(result)
                if citation and citation not in private_seen:
                    citations.append(citation)
                    private_seen.add(citation)
                source_label = (
                    f"PRIVATE_EVIDENCE_{idx}: "
                    f"document={document}; page={page}"
                )
                rule = "This evidence is from the user's temporary private document."
            else:
                source_label = f"CORPUS_EVIDENCE_{idx}"
                rule = "This evidence comes from the permanent Space RAG knowledge corpus. Do not reveal its provenance."

            visual = result.get("verification")
            verification_text = ""
            if visual:
                verification_text = f"\nVerification: {visual}"

            blocks.append(
                f"{source_label}\n{rule}\nEvidence:\n{text}{verification_text}"
            )

        return "\n\n".join(blocks), citations

    def _system_prompt(self) -> str:
        return """
You are Space RAG, an advanced conversational intelligence assistant specialized in satellite systems, remote sensing, and aerospace engineering.

Use ONLY the evidence supplied in the context.
Do not invent facts that are not supported by the supplied evidence.
When evidence is insufficient, say that the retrieved evidence is insufficient.

SOURCE RULES:
1. Reference knowledge is permanent background aerospace knowledge. Synthesize it naturally as an AI expert without ever mentioning "corpus", "corpus pdf", "corpus documents", "corpus evidence", file paths, or internal identifiers. Never say "according to the corpus" or "in the corpus".
2. PRIVATE_EVIDENCE comes from the user's uploaded document.
   When using private evidence, identify the document and page in natural language (e.g. [document.pdf, p. X]).
3. Be direct, conversational, and technically precise. State answers and numerical values directly without repetitive meta-talk like "as shown in the table" or "according to the table below".
4. When answering questions about mission maneuvers, dates, or orbital parameters (apogee, perigee, delta-V), strictly match the specific maneuver name (e.g., EBN #1..#5, TCM #1) with its exact row/entry in the data. If a user query combines an event name with an incorrect or mismatched date, clarify the true date and value from the evidence rather than confusing data between different maneuvers.
""".strip()

    def _prompt(self, query: str, results: List[dict]) -> tuple[str, List[str]]:
        evidence_text, citations = self._prepare_evidence(results)
        user_prompt = f"""
Question:
{query}

Retrieved technical context:
{evidence_text}

Answer the question conversationally using only the technical context provided.
Do not mention the words 'corpus' or 'corpus pdf'. Integrate background knowledge seamlessly.
For private evidence from attached documents, cite the document and page number clearly.
""".strip()
        return user_prompt, citations

    def _build_dialogue_messages(
        self,
        user_prompt: str,
        chat_history: Optional[List[dict]] = None,
    ) -> List[dict]:
        messages = [{"role": "system", "content": self._system_prompt()}]
        if chat_history:
            # Include up to the last 4 turns for conversational context
            for turn in chat_history[-4:]:
                r = turn.get("role")
                c = turn.get("content")
                if r in ("user", "assistant") and c:
                    messages.append({"role": r, "content": c})
        messages.append({"role": "user", "content": user_prompt})
        return messages

    def generate(
        self,
        query: str,
        retrieval_results: List[dict],
        model_mode: str = "offline",
        chat_history: Optional[List[dict]] = None,
        api_key: Optional[str] = None,
    ) -> Dict:
        if not retrieval_results:
            return {
                "answer": "I could not find sufficient evidence in the available knowledge base or attached document to answer that question.",
                "citations": [],
                "model": "none",
                "source_mix": {"corpus": 0, "private": 0},
            }

        user_prompt, citations = self._prompt(query, retrieval_results)
        source_mix = {
            "corpus": sum(r.get("source_type") == "corpus" for r in retrieval_results),
            "private": sum(r.get("source_type") == "private" for r in retrieval_results),
        }

        if model_mode == "online":
            answer = self._generate_grok(user_prompt, chat_history=chat_history, api_key=api_key)
            model_name = self.online_model_name
        else:
            answer = self._generate_offline(user_prompt, chat_history=chat_history)
            model_name = self.offline_model_name

        answer = self._sanitize_output(answer)
        return {
            "answer": answer,
            "citations": citations,
            "model": model_name,
            "source_mix": source_mix,
        }

    def generate_stream(
        self,
        query: str,
        retrieval_results: List[dict],
        model_mode: str = "offline",
        chat_history: Optional[List[dict]] = None,
        api_key: Optional[str] = None,
    ):
        if not retrieval_results:
            yield "I could not find sufficient evidence in the available knowledge base or attached document."
            return

        user_prompt, _ = self._prompt(query, retrieval_results)
        messages = self._build_dialogue_messages(user_prompt, chat_history=chat_history)

        if model_mode == "online":
            self._load_grok(api_key=api_key)
            response = self._grok_client.chat.completions.create(
                model=self.online_model_name,
                messages=messages,
                stream=True,
            )
            for chunk in response:
                content = chunk.choices[0].delta.content or ""
                if content:
                    yield content
        else:
            self._load_offline()
            import torch
            from threading import Thread
            from transformers import TextIteratorStreamer

            prompt = self._tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            inputs = self._tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=6000,
            )
            streamer = TextIteratorStreamer(self._tokenizer, skip_prompt=True, skip_special_tokens=True)
            generation_kwargs = dict(
                **inputs,
                streamer=streamer,
                max_new_tokens=700,
                do_sample=False,
                pad_token_id=self._tokenizer.eos_token_id,
            )
            thread = Thread(target=self._model.generate, kwargs=generation_kwargs)
            thread.start()
            for token in streamer:
                yield token

    def _generate_grok(
        self,
        user_prompt: str,
        chat_history: Optional[List[dict]] = None,
        api_key: Optional[str] = None,
    ) -> str:
        self._load_grok(api_key=api_key)
        messages = self._build_dialogue_messages(user_prompt, chat_history=chat_history)
        response = self._grok_client.chat.completions.create(
            model=self.online_model_name,
            messages=messages,
        )
        content = response.choices[0].message.content or ""
        return content.strip()

    def _generate_offline(
        self,
        user_prompt: str,
        chat_history: Optional[List[dict]] = None,
    ) -> str:
        self._load_offline()
        import torch

        messages = self._build_dialogue_messages(user_prompt, chat_history=chat_history)
        prompt = self._tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self._tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=6000,
        )

        with torch.no_grad():
            output = self._model.generate(
                **inputs,
                max_new_tokens=700,
                do_sample=False,
                pad_token_id=self._tokenizer.eos_token_id,
            )

        new_tokens = output[0][inputs["input_ids"].shape[1]:]
        return self._tokenizer.decode(
            new_tokens,
            skip_special_tokens=True,
        ).strip()

    @staticmethod
    def _sanitize_output(text: str) -> str:
        text = re.sub(r"\bCORPUS_EVIDENCE_\d+\b", "", text, flags=re.I)
        text = re.sub(r"\bPRIVATE_EVIDENCE_\d+\b", "", text, flags=re.I)
        text = re.sub(r"\bthe corpus pdf\b", "the reference documentation", text, flags=re.I)
        text = re.sub(r"\bcorpus pdfs?\b", "reference documentation", text, flags=re.I)
        text = re.sub(r"\bcorpus documents?\b", "reference documentation", text, flags=re.I)
        text = re.sub(r"\bin the corpus\b", "in the reference literature", text, flags=re.I)
        text = re.sub(r"\bfrom the corpus\b", "from the documentation", text, flags=re.I)
        text = re.sub(r"\bthe corpus\b", "the knowledge base", text, flags=re.I)
        text = re.sub(r"\bcorpus\b", "knowledge base", text, flags=re.I)
        text = re.sub(r"\s{2,}", " ", text).strip()
        return text
