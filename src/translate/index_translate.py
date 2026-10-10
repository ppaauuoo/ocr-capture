"""Index-Translate-2B through a locally built llama.cpp runtime."""
import atexit
import http.client
import json
import urllib.request

from src.translate import INDEX_TRANSLATE_ENGINE
from src.translate.llama_cpp import LlamaCppRuntime

LLAMA_CPP_REPOSITORY = "https://github.com/ggml-org/llama.cpp.git"
LLAMA_CPP_COMMIT = "aa94f2086147cd4bfbcff818ce998dffce5fcb39"
MODEL_REPOSITORY = "IndexTeam/Index-Translate-2B-GGUF"
MODEL_REVISION = "0a622669d15a139787818187e6263d19e4a1ffb5"
MODEL_FILENAME = "Index-Translate-2B.Q4_K_M.gguf"
MODEL_SIZE = 1_312_164_352
MODEL_SHA256 = "044b313d29342bd3b2c77cbb64023ca0d209bd9b3247763f9d162767ef2d746a"
MODEL_ALIAS = INDEX_TRANSLATE_ENGINE


class IndexTranslateRuntime(LlamaCppRuntime):
    name = INDEX_TRANSLATE_ENGINE
    cache_name = "index-translate-2b"
    repository = LLAMA_CPP_REPOSITORY
    commit = LLAMA_CPP_COMMIT
    model_repository = MODEL_REPOSITORY
    model_revision = MODEL_REVISION
    model_filename = MODEL_FILENAME
    model_size = MODEL_SIZE
    model_sha256 = MODEL_SHA256
    model_alias = MODEL_ALIAS
    context_size = 32768
    server_arguments = ("--n-gpu-layers", "0", "--jinja", "--temp", "0")


_RUNTIME = IndexTranslateRuntime()
atexit.register(_RUNTIME.shutdown)


class IndexTranslate:
    def __init__(self, target="English", runtime=None, status_callback=None):
        self.target = target
        self._runtime = runtime or _RUNTIME
        self._status_callback = status_callback

    def run(self, text):
        base_url = self._runtime.ensure_ready(self._status_callback)
        if self._status_callback:
            self._status_callback(f"Translating with {INDEX_TRANSLATE_ENGINE}…")
        target = {"english": "英语"}.get(self.target.casefold(), self.target)
        prompt = (
            f"请将以下文本翻译为{target}，直接输出翻译结果，不要进行任何解释。\n\n"
            f"{text.strip()}"
        )
        request = urllib.request.Request(
            f"{base_url}/v1/chat/completions",
            data=json.dumps({
                "model": MODEL_ALIAS,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0,
                "max_tokens": 1024,
                "chat_template_kwargs": {"enable_thinking": False},
            }).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=1800) as response:
                result = json.load(response)["choices"][0]["message"]["content"].strip()
        except (
            OSError, http.client.HTTPException, KeyError, IndexError,
            TypeError, ValueError, AttributeError,
        ) as error:
            raise RuntimeError(f"{INDEX_TRANSLATE_ENGINE} translation failed: {error}") from error
        if not result:
            raise RuntimeError(f"{INDEX_TRANSLATE_ENGINE} returned an empty translation.")
        return result

    def run_many(self, texts):
        return [self.run(text) for text in texts]
