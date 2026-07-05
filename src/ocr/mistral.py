import os
import base64
from dotenv import load_dotenv
from mistralai.client import Mistral
from typing import List

load_dotenv()
api_key = os.environ["MISTRAL_API_KEY"]

client = Mistral(api_key=api_key)


class MistralOCR:
    def run(self, image_path="temp/image.png"):
        base64_image = self.encode_image(image_path)

        return client.ocr.process(
            model="mistral-ocr-latest",
            document={
                "type": "image_url",
                "image_url": f"data:image/jpeg;base64,{base64_image}",
            },
            # table_format=None,
            include_image_base64=True,
        )

    def encode_image(self, image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    def extract(ocr_response) -> List[str]:
        """Print clean lines from Mistral OCR response as a list."""
        lines = []
        seen = set()
        for page in ocr_response.pages:
            for line in page.markdown.split("\n"):
                stripped = line.strip()
                if stripped and len(stripped) > 3:
                    if stripped not in seen:
                        seen.add(stripped)
                        print(f"- {stripped}")
                        lines.append(stripped)
        return lines
