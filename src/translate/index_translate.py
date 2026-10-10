"""Index-Translate-2B served by Ollama."""
import threading

import ollama

from src.translate import INDEX_TRANSLATE_ENGINE

OLLAMA_MODEL = "hf.co/IndexTeam/Index-Translate-2B-GGUF:Q4_K_M"


_MODEL_READY = False
_MODEL_LOCK = threading.Lock()


def _ensure_model(status_callback):
    global _MODEL_READY
    if _MODEL_READY:
        return
    with _MODEL_LOCK:
        if _MODEL_READY:
            return
        try:
            if status_callback:
                status_callback(f"Preparing {INDEX_TRANSLATE_ENGINE} in Ollama…")
            for progress in ollama.pull(OLLAMA_MODEL, stream=True):
                if status_callback:
                    status_callback(f"Ollama: {progress['status']}")
        except Exception as error:
            raise RuntimeError(
                f"Could not prepare {INDEX_TRANSLATE_ENGINE} in Ollama. "
                f"Install and start Ollama, then retry: {error}"
            ) from error
        _MODEL_READY = True


class IndexTranslate:
    def __init__(self, target="English", status_callback=None):
        self.target = target
        self._status_callback = status_callback

    def run(self, text):
        _ensure_model(self._status_callback)
        if self._status_callback:
            self._status_callback(f"Translating with {INDEX_TRANSLATE_ENGINE}…")
        target = {"english": "英语"}.get(self.target.casefold(), self.target)
        prompt = (
            f"请将以下文本翻译为{target}，直接输出翻译结果，不要进行任何解释。\n\n"
            f"{text.strip()}"
        )
        try:
            response = ollama.chat(
                model=OLLAMA_MODEL,
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": 0, "num_predict": 1024},
                think=False,
            )
            result = response["message"]["content"].strip()
        except Exception as error:
            raise RuntimeError(f"{INDEX_TRANSLATE_ENGINE} translation failed: {error}") from error
        if not result:
            raise RuntimeError(f"{INDEX_TRANSLATE_ENGINE} returned an empty translation.")
        return result

    def run_many(self, texts):
        return [self.run(text) for text in texts]
