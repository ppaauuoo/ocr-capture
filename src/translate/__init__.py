DEFAULT_TRANSLATION_ENGINE = "Hy-MT2"
TRANSLATION_ENGINES = ("Hy-MT2", "TranslateGemma")


def create_translator(name=DEFAULT_TRANSLATION_ENGINE, status_callback=None):
    if name == "Hy-MT2":
        from src.translate.hy_mt2 import HyMT2
        return HyMT2(status_callback=status_callback)
    if name == "TranslateGemma":
        from src.translate.ollama import TranslateGemma
        return TranslateGemma()
    raise ValueError(f"Unknown translation engine: {name}")
