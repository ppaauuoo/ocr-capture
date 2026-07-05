from PIL import Image as PILImage
from src.capture import Capture
from src.click_region import get_region
from src.ocr.mistral import MistralOCR
from src.translate.ollama import TranslateGemma

mistral = MistralOCR()


def main():
    region = get_region()

    Capture().run()          # saves full screen to temp/image.png

    # Crop full-screen capture to the user-defined region
    img = PILImage.open("temp/image.png")
    cropped = img.crop((region.x1, region.y1, region.x2, region.y2))
    cropped.save("temp/image.png")

    src = mistral.run()
    texts = mistral.extract(src)
    for txt in texts:
        translated_text = TranslateGemma().run(txt)
        print(translated_text)


if __name__ == "__main__":
    main()
