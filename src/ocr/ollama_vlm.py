"""OCR via ollama vision model (gemma4:e2b)."""
import ollama
from typing import List

MODEL = "gemma4:e2b"

PROMPT = """Extract all Japanese text from this image.
Return only the extracted text, nothing else."""


class OllamaVisionOCR:
    def run(self, image_path: str = "temp/image.png") -> str:
        response = ollama.chat(
            model=MODEL,
            messages=[{
                "role": "user",
                "content": PROMPT,
                "images": [image_path],
            }],
        )
        return response["message"]["content"]

    def extract(self, raw: str) -> List[str]:
        lines = [
            line.strip()
            for line in raw.split("\n")
            if line.strip() and len(line.strip()) > 1
        ]
        return lines
