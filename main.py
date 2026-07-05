from src.capture import Capture
from src.ocr.mistral import MistralOCR
from src.translate.ollama import TranslateGemma

mistral = MistralOCR()


def main():
    Capture().run()
    src = mistral.run()
    texts = mistral.extract(src)
    for txt in texts:
        translated_text = TranslateGemma().run(txt)
        print(translated_text)


if __name__ == "__main__":
    main()
