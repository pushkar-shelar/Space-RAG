from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Dict, List, Optional


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
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(self.offline_model_name)
        self._model = AutoModelForCausalLM.from_pretrained(
            self.offline_model_name,
            torch_dtype=torch.float32,
            device_map="cpu",
        ).eval()

    def _load_grok(self):
        if self._grok_client is not None:
            return
        api_key = os.getenv("XAI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "XAI_API_KEY is not configured. Set it in the environment before using Online Grok mode."
            )
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError(
                "The 'openai' package is required for Online Grok mode. Run: pip install -U openai"
            ) from exc

        self._grok_client = OpenAI(
            api_key=api_key,
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
                # Deliberately no corpus document/page/path is placed in this block.
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
You are Space RAG, a retrieval-grounded assistant for satellite and remote-sensing documents.

Use ONLY the evidence supplied in the user prompt.
Do not invent facts that are not supported by the supplied evidence.
When evidence is insufficient, say that the retrieved evidence is insufficient.

SOURCE RULES:
1. CORPUS_EVIDENCE is permanent background knowledge. You may use it, but NEVER reveal:
   - corpus filenames
   - corpus document titles
   - corpus paths
   - corpus page numbers
   - internal source identifiers
   - the phrase CORPUS_EVIDENCE
2. PRIVATE_EVIDENCE comes from the user's temporary uploaded document.
   When the answer uses private evidence, identify the document and page in normal language,
   using the supplied document/page information.
3. Never claim that corpus material came from the private document.
4. Never claim that private-document information came from the corpus.
5. Prefer direct evidence over general knowledge.
6. Do not browse the web or use outside knowledge for the answer.

Answer clearly and directly. For technical questions, preserve important numbers, units,
and conditions from the evidence.
""".strip()

    def _prompt(self, query: str, results: List[dict]) -> tuple[str, List[str]]:
        evidence_text, citations = self._prepare_evidence(results)
        user_prompt = f"""
Question:
{query}

Retrieved evidence:
{evidence_text}

Answer the question using only the retrieved evidence.
For private evidence, the final answer should make the document/page source clear.
Do not expose corpus provenance.
""".strip()
        return user_prompt, citations

    def generate(
        self,
        query: str,
        retrieval_results: List[dict],
        model_mode: str = "offline",
    ) -> Dict:
        if not retrieval_results:
            return {
                "answer": "I could not find sufficient evidence in the available corpus or private document.",
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
            answer = self._generate_grok(user_prompt)
            model_name = self.online_model_name
        else:
            answer = self._generate_offline(user_prompt)
            model_name = self.offline_model_name

        answer = self._sanitize_output(answer)
        return {
            "answer": answer,
            "citations": citations,
            "model": model_name,
            "source_mix": source_mix,
        }

    def _generate_grok(self, user_prompt: str) -> str:
        self._load_grok()
        response = self._grok_client.chat.completions.create(
            model=self.online_model_name,
            messages=[
                {"role": "system", "content": self._system_prompt()},
                {"role": "user", "content": user_prompt},
            ],
        )
        content = response.choices[0].message.content or ""
        return content.strip()

    def _generate_offline(self, user_prompt: str) -> str:
        self._load_offline()
        import torch

        messages = [
            {"role": "system", "content": self._system_prompt()},
            {"role": "user", "content": user_prompt},
        ]
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
        text = re.sub(r"\s{2,}", " ", text).strip()
        return text
