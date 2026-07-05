"""OCR via YomiToku (Japanese-optimized document AI)."""
from yomitoku import DocumentAnalyzer
from PIL import Image
import numpy as np
from typing import List


class YomiTokuOCR:
    def __init__(self, device: str = "cuda"):
        self._pipe = DocumentAnalyzer(configs={}, device=device, visualize=False)

    def run(self, image_path: str = "temp/image.png") -> tuple:
        img = np.array(Image.open(image_path))
        return self._pipe(img)

    def extract(self, raw: tuple) -> List[str]:
        lines = []
        schema = raw[0]
        for p in schema.paragraphs:
            if p.contents and len(p.contents) > 1:
                lines.append(p.contents.strip())
        for w in schema.words:
            if w.content and len(w.content) > 1:
                lines.append(w.content.strip())
        return lines
