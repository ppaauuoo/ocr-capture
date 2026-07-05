"""OCR via PaddleOCR (lightweight, multi-language)."""
from paddleocr import PaddleOCR
from typing import List


class PaddleOCREngine:
    def __init__(self, lang: str = "japan"):
        # ponytail: PaddleOCR downloads model weights on first run
        self._ocr = PaddleOCR(
            lang=lang,
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
        )

    def run(self, image_path: str = "temp/image.png") -> list:
        return self._ocr.predict(image_path)

    def extract(self, raw: list) -> List[str]:
        lines = []
        if not raw or not raw[0]:
            return lines
        for region in raw[0]:
            text = region[1][0].strip()
            if text and len(text) > 1:
                lines.append(text)
        return lines
