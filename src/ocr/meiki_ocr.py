"""OCR via MeikiOCR (ONNX-based, lightweight, Japanese games)."""
import cv2
import numpy as np
from meikiocr import MeikiOCR
from typing import List


class MeikiOCREngine:
    def __init__(self):
        # ponytail: downloads model on first run
        self._ocr = MeikiOCR()

    def run(self, image_path: str = "temp/image.png") -> List[dict]:
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Could not read image: {image_path}")
        result = self._ocr.run_ocr(image)
        return result

    def extract(self, raw: List[dict]) -> List[str]:
        lines = [
            item["text"].strip()
            for item in raw
            if item.get("text", "").strip() and len(item["text"].strip()) > 1
        ]
        return lines
