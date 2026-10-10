"""Hy-MT2 translation through a locally built, STQ-capable llama.cpp server."""
import atexit
import http.client
import json
import urllib.request

from src.translate.llama_cpp import LlamaCppRuntime

LLAMA_CPP_REPOSITORY = "https://github.com/sjl623/llama.cpp.git"
LLAMA_CPP_COMMIT = "1e411d8f5a1e23525fa3265dfb4bd76265465397"
MODEL_REPOSITORY = "tencent/Hy-MT2-1.8B-1.25Bit-GGUF"
MODEL_REVISION = "9df5c824a00a744fb0512a29c640466f4d97dfb0"
MODEL_FILENAME = "Hy-MT2-1.8B-1.25Bit.gguf"
MODEL_SIZE = 461_860_800
MODEL_SHA256 = "cc497fe8f033b52b3b8b00a7669e9661435432f9d4cd43f7ed24400c01507a93"
MODEL_ALIAS = "hy-mt2-1.8b-stq"


class HyMT2Runtime(LlamaCppRuntime):
    name = "Hy-MT2"
    cache_name = "hy-mt2"
    repository = LLAMA_CPP_REPOSITORY
    commit = LLAMA_CPP_COMMIT
    model_repository = MODEL_REPOSITORY
    model_revision = MODEL_REVISION
    model_filename = MODEL_FILENAME
    model_size = MODEL_SIZE
    model_sha256 = MODEL_SHA256
    model_alias = MODEL_ALIAS
    context_size = 8192
    server_arguments = (
        "--n-gpu-layers", "0", "--jinja", "--temp", "0.7", "--top-p", "0.6",
        "--top-k", "20", "--repeat-penalty", "1.05",
    )


_RUNTIME = HyMT2Runtime()
atexit.register(_RUNTIME.shutdown)


class HyMT2:
    def __init__(self, target="English", runtime=None, status_callback=None):
        self.target = target
        self._runtime = runtime or _RUNTIME
        self._status_callback = status_callback

    def run(self, text):
        base_url = self._runtime.ensure_ready(self._status_callback)
        if self._status_callback:
            self._status_callback("Translating with Hy-MT2…")
        prompt = (
            f"Translate the following text into {self.target}. "
            "Note that you should only output the translated result without any additional explanation:\n\n"
            f"{text}"
        )
        request = urllib.request.Request(
            f"{base_url}/v1/chat/completions",
            data=json.dumps({
                "model": MODEL_ALIAS,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 4096,
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
            raise RuntimeError(f"Hy-MT2 translation failed: {error}") from error
        if not result:
            raise RuntimeError("Hy-MT2 returned an empty translation.")
        return result

    def run_many(self, texts):
        return [self.run(text) for text in texts]


