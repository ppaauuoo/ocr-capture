"""Japanese OCR via Manga OCR."""
from typing import List

from manga_ocr import MangaOcr


class MangaOCREngine:
    def __init__(self):
        # Current project Torch build does not include RTX 5060 Ti (sm_120) kernels.
        self._ocr = MangaOcr(force_cpu=True)

    def run(self, image_path: str = "temp/image.png") -> str:
        return self._ocr(image_path)

    def extract(self, raw: str) -> List[str]:
        text = raw.strip()
        return [text] if text else []
