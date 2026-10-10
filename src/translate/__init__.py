HY_MT2_ENGINE = "Hy-MT2"
TRANSLATEGEMMA_ENGINE = "TranslateGemma"
INDEX_TRANSLATE_ENGINE = "Index-Translate-2B"
DEFAULT_TRANSLATION_ENGINE = INDEX_TRANSLATE_ENGINE
TRANSLATION_ENGINES = (HY_MT2_ENGINE, TRANSLATEGEMMA_ENGINE, INDEX_TRANSLATE_ENGINE)


def create_translator(name=DEFAULT_TRANSLATION_ENGINE, status_callback=None):
    if name == HY_MT2_ENGINE:
        from src.translate.hy_mt2 import HyMT2
        return HyMT2(status_callback=status_callback)
    if name == TRANSLATEGEMMA_ENGINE:
        from src.translate.ollama import TranslateGemma
        return TranslateGemma()
    if name == INDEX_TRANSLATE_ENGINE:
        from src.translate.index_translate import IndexTranslate
        return IndexTranslate(status_callback=status_callback)
    raise ValueError(f"Unknown translation engine: {name}")
