from src.capture import Capture
from src.ocr.mistral import MistralOCR, extract


def main():
    Capture().run()
    text = MistralOCR().run()
    extract(text)


if __name__ == "__main__":
    main()
