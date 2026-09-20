from PIL import Image as PILImage
from src.capture import Capture
from src.click_region import get_region
from src.translate.ollama import TranslateGemma

# ── OCR picker ──────────────────────────────────────────
# Change OCR to: "mistral", "deepseek", "ollama", "yomitoku", "paddle", "meiki"
OCR = "meiki"

if OCR == "mistral":
    from src.ocr.mistral import MistralOCR as OCRImpl
elif OCR == "deepseek":
    from src.ocr.deepseekocr2 import DeepseekOCR2 as OCRImpl
elif OCR == "ollama":
    from src.ocr.ollama_vlm import OllamaVisionOCR as OCRImpl
elif OCR == "yomitoku":
    from src.ocr.yomitoku_ocr import YomiTokuOCR as OCRImpl
elif OCR == "paddle":
    from src.ocr.paddle_ocr import PaddleOCREngine as OCRImpl
elif OCR == "meiki":
    from src.ocr.meiki_ocr import MeikiOCREngine as OCRImpl
else:
    raise ValueError(f"Unknown OCR: {OCR}")

ocr = OCRImpl()


def main():
    region = get_region()

    Capture().run()

    img = PILImage.open("temp/image.png")
    cropped = img.crop((region.x1, region.y1, region.x2, region.y2))
    cropped.save("temp/image.png")

    src = ocr.run()
    texts = ocr.extract(src)
    if not texts or all(len(t) < 3 for t in texts):
        print(None)
        return
    for translated_text in TranslateGemma().run_many(texts):
        print(translated_text)


if __name__ == "__main__":
    main()
