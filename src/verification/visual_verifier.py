import torch

from pathlib import Path

from PIL import Image

from transformers import (
    AutoProcessor,
    AutoModelForImageTextToText
)


class VisualVerifier:

    def __init__(
        self,
        model_name="HuggingFaceTB/SmolVLM-500M-Instruct"
    ):
        self.model_name = model_name

        # ---------------------------------------------------------
        # Device
        # ---------------------------------------------------------

        self.device = (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        print(
            f"Visual verifier device: {self.device}"
        )

        # ---------------------------------------------------------
        # Load processor
        # ---------------------------------------------------------

        print(
            "Loading SmolVLM processor..."
        )

        self.processor = (
            AutoProcessor.from_pretrained(
                self.model_name
            )
        )

        print(
            "SmolVLM processor loaded."
        )

        # ---------------------------------------------------------
        # Load model
        # ---------------------------------------------------------

        print(
            "Loading SmolVLM model..."
        )

        self.model = (
            AutoModelForImageTextToText.from_pretrained(
                self.model_name,
                torch_dtype=torch.float32
                if self.device == "cpu"
                else torch.float16
            )
        )

        self.model.to(self.device)
        self.model.eval()

        print(
            "SmolVLM model loaded."
        )

    # ---------------------------------------------------------
    # Clean VLM output
    # ---------------------------------------------------------

    def clean_model_output(
        self,
        text
    ):
        """
        Clean SmolVLM output.
        """

        if not text:
            return ""

        text = str(
            text
        ).strip()

        # Remove conversation markers
        if "Assistant:" in text:
            text = (
                text
                .rsplit(
                    "Assistant:",
                    1
                )[1]
                .strip()
            )

        if "User:" in text:
            text = (
                text
                .split(
                    "User:",
                    1
                )[0]
                .strip()
            )

        # Remove answer prefix
        if text.lower().startswith(
            "answer:"
        ):
            text = text[
                len("answer:")
            :].strip()

        return text
    # ---------------------------------------------------------
    # Analyze image
    # ---------------------------------------------------------

    def analyze_image(
        self,
        image_path,
        question
    ):
        """
        Analyze a PDF page image with SmolVLM.
        """

        image_path = Path(
            image_path
        )

        if not image_path.exists():

            raise FileNotFoundError(
                f"Image not found: {image_path}"
            )

        # ---------------------------------------------------------
        # Load image
        # ---------------------------------------------------------

        image = Image.open(
            image_path
        ).convert("RGB")

        # ---------------------------------------------------------
        # Conversation
        # ---------------------------------------------------------

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": str(image_path)
                    },
                    {
                        "type": "text",
                        "text": (
                             "Analyze ONLY the graph or figure relevant "
                            "to the question. Ignore unrelated text, "
                            "headers, page numbers, and other figures "
                            "on the page.\n\n"

                            "Identify the overall purpose of the graph, "
                            "what quantities or functions it represents, "
                            "and which channels or variables are involved. "
                            "Do not merely describe axis labels or individual "
                            "numbers.\n\n"

                            "If a figure caption is visible, use it to identify "
                            "what the figure represents.\n\n"

                            "Give one concise factual sentence.\n\n"

                            f"Question: {question}"
                        )
                    }
                ]
            }
        ]

        # ---------------------------------------------------------
        # Prepare processor inputs
        # ---------------------------------------------------------

        inputs = (
            self.processor.apply_chat_template(
                messages,
                tokenize=True,
                return_dict=True,
                return_tensors="pt"
            )
        )

        # ---------------------------------------------------------
        # Move tensors to device
        # ---------------------------------------------------------

        inputs = {
            key: value.to(self.device)
            if hasattr(value, "to")
            else value
            for key, value in inputs.items()
        }

        # ---------------------------------------------------------
        # Remember input token length
        #
        # generate() returns:
        #
        # input tokens + generated tokens
        #
        # We only want the generated part.
        # ---------------------------------------------------------

        input_token_length = (
            inputs["input_ids"].shape[-1]
        )

        # ---------------------------------------------------------
        # Generate answer
        # ---------------------------------------------------------

        with torch.no_grad():

            generated_ids = (
                self.model.generate(
                    **inputs,
                    max_new_tokens=64,
                    do_sample=False
                )
            )

        # ---------------------------------------------------------
        # Keep ONLY newly generated tokens
        # ---------------------------------------------------------

        generated_answer_ids = (
            generated_ids[
                :,
                input_token_length:
            ]
        )

        # ---------------------------------------------------------
        # Decode only the answer
        # ---------------------------------------------------------

        generated_text = (
            self.processor.batch_decode(
                generated_answer_ids,
                skip_special_tokens=True
            )[0]
        )

        # ---------------------------------------------------------
        # Clean answer
        # ---------------------------------------------------------

        cleaned_text = (
            self.clean_model_output(
                generated_text
            )
        )

        # ---------------------------------------------------------
        # Debug fallback
        # ---------------------------------------------------------

        if not cleaned_text:

            print(
                "\nWARNING: SmolVLM returned "
                "an empty answer."
            )

            print(
                "Raw generated text:"
            )

            print(
                repr(generated_text)
            )

        return cleaned_text