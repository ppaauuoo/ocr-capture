"""Hy-MT2 translation through a locally built, STQ-capable llama.cpp server."""
import atexit
import hashlib
import json
import os
import shutil
import socket
import subprocess
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path


LLAMA_CPP_REPOSITORY = "https://github.com/sjl623/llama.cpp.git"
LLAMA_CPP_COMMIT = "1e411d8f5a1e23525fa3265dfb4bd76265465397"
MODEL_REPOSITORY = "tencent/Hy-MT2-1.8B-1.25Bit-GGUF"
MODEL_REVISION = "9df5c824a00a744fb0512a29c640466f4d97dfb0"
MODEL_FILENAME = "Hy-MT2-1.8B-1.25Bit.gguf"
MODEL_SIZE = 461_860_800
MODEL_SHA256 = "cc497fe8f033b52b3b8b00a7669e9661435432f9d4cd43f7ed24400c01507a93"
MODEL_ALIAS = "hy-mt2-1.8b-stq"


class HyMT2Runtime:
    def __init__(self, cache_dir=None):
        data_root = Path(os.environ.get("LOCALAPPDATA") or Path.home() / ".cache")
        self.cache_dir = Path(cache_dir).resolve() if cache_dir else data_root / "ocr-capture" / "hy-mt2"
        self._lock = threading.Lock()
        self._process: subprocess.Popen | None = None
        self._base_url: str | None = None

    def ensure_ready(self, status_callback=None) -> str:
        with self._lock:
            if self._process and self._process.poll() is None:
                if self._base_url and self._is_model_server(self._base_url):
                    return self._base_url
                raise RuntimeError("The Hy-MT2 llama.cpp server stopped responding.")

            binary = self._ensure_server_binary(status_callback)
            model = self._ensure_model(status_callback)

            port = self._free_port()
            base_url = f"http://127.0.0.1:{port}"
            self._base_url = base_url
            command = [
                str(binary), "--model", str(model), "--alias", MODEL_ALIAS,
                "--host", "127.0.0.1", "--port", str(port),
                "--ctx-size", "8192", "--n-gpu-layers", "0", "--jinja",
                "--temp", "0.7", "--top-p", "0.6", "--top-k", "20",
                "--repeat-penalty", "1.05",
            ]
            self._notify(status_callback, "Starting the Hy-MT2 local runtime…")
            log_path = self.cache_dir / "server.log"
            creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
            with log_path.open("ab") as log:
                self._process = subprocess.Popen(
                    command, stdout=log, stderr=subprocess.STDOUT,
                    creationflags=creationflags,
                )
            process = self._process
            if process is None:
                raise RuntimeError("Could not start the Hy-MT2 llama.cpp server.")
            self._wait_until_ready(status_callback, process, base_url)
            return base_url

    def shutdown(self):
        with self._lock:
            process, self._process = self._process, None
            self._base_url = None
            if process and process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()

    def _ensure_server_binary(self, status_callback):
        source = self.cache_dir / f"llama.cpp-{LLAMA_CPP_COMMIT[:12]}"
        build = source / "build"
        binary_name = "llama-server.exe" if os.name == "nt" else "llama-server"
        candidates = (
            build / "bin" / "Release" / binary_name,
            build / "bin" / binary_name,
        )
        for candidate in candidates:
            if candidate.is_file():
                return candidate
        for candidate in build.rglob(binary_name) if build.exists() else ():
            return candidate

        self._notify(status_callback, "Building the pinned STQ-capable llama.cpp runtime (first use)…")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        if not (source / ".git").exists():
            if source.exists():
                shutil.rmtree(source)
            source.mkdir(parents=True)
            self._run_logged(["git", "init", "--quiet"], source)
            self._run_logged(["git", "remote", "add", "origin", LLAMA_CPP_REPOSITORY], source)
        self._run_logged(["git", "fetch", "--depth", "1", "origin", LLAMA_CPP_COMMIT], source)
        self._run_logged(["git", "checkout", "--detach", "FETCH_HEAD"], source)
        self._run_logged([
            "cmake", "-S", str(source), "-B", str(build),
            "-DCMAKE_BUILD_TYPE=Release", "-DGGML_CUDA=OFF", "-DGGML_VULKAN=OFF",
        ], source)
        self._run_logged([
            "cmake", "--build", str(build), "--config", "Release",
            "--target", "llama-server", "--parallel", "2",
        ], source, timeout=1800)
        for candidate in candidates:
            if candidate.is_file():
                return candidate
        for candidate in build.rglob(binary_name):
            return candidate
        raise RuntimeError(f"llama.cpp build completed without producing {binary_name}.")

    def _ensure_model(self, status_callback):
        model = self.cache_dir / MODEL_FILENAME
        if model.is_file():
            self._notify(status_callback, "Checking the cached Hy-MT2 model…")
            if self._sha256(model) != MODEL_SHA256:
                raise RuntimeError(f"Cached Hy-MT2 model failed its SHA-256 check: {model}")
            return model

        self._notify(status_callback, "Downloading the 462 MB Hy-MT2 STQ model…")
        url = (
            f"https://huggingface.co/{MODEL_REPOSITORY}/resolve/"
            f"{MODEL_REVISION}/{MODEL_FILENAME}?download=true"
        )
        part = model.with_suffix(model.suffix + ".part")
        digest = hashlib.sha256()
        received = 0
        last_percent = -1
        try:
            with urllib.request.urlopen(url, timeout=60) as response, part.open("wb") as output:
                while chunk := response.read(1024 * 1024):
                    output.write(chunk)
                    digest.update(chunk)
                    received += len(chunk)
                    percent = received * 100 // MODEL_SIZE
                    if percent != last_percent:
                        self._notify(status_callback, f"Downloading Hy-MT2 model… {percent}%")
                        last_percent = percent
            if received != MODEL_SIZE or digest.hexdigest() != MODEL_SHA256:
                raise RuntimeError("Downloaded Hy-MT2 model has an unexpected size or SHA-256 hash.")
            part.replace(model)
            return model
        except Exception:
            part.unlink(missing_ok=True)
            raise

    def _run_logged(self, command, cwd, timeout=300):
        log_path = self.cache_dir / "build.log"
        try:
            with log_path.open("ab") as log:
                log.write(("\n> " + " ".join(command) + "\n").encode())
                result = subprocess.run(
                    command, cwd=cwd, stdout=log, stderr=subprocess.STDOUT,
                    check=False, timeout=timeout,
                )
        except FileNotFoundError as error:
            raise RuntimeError(
                f"Required build tool {command[0]!r} was not found. Install Git, CMake, "
                "and a C++ compiler (Visual Studio Build Tools with the C++ workload on Windows)."
            ) from error
        except subprocess.TimeoutExpired as error:
            raise RuntimeError(f"Timed out while running {command[0]} to prepare the Hy-MT2 runtime.") from error
        if result.returncode:
            details = log_path.read_text(encoding="utf-8", errors="replace")[-4000:]
            raise RuntimeError(
                f"Could not prepare the STQ-capable llama.cpp runtime ({command[0]} exited "
                f"{result.returncode}). Install Git, CMake, and a C++ compiler; build log: "
                f"{log_path}\n{details}"
            )

    def _wait_until_ready(self, status_callback, process, base_url):
        deadline = time.monotonic() + 300
        while time.monotonic() < deadline:
            if process.poll() is not None:
                details = self._read_server_log()
                raise RuntimeError(f"Hy-MT2 runtime exited while loading the STQ model.\n{details}")
            if self._is_model_server(base_url):
                return
            self._notify(status_callback, "Loading the Hy-MT2 model into the local runtime…")
            time.sleep(1)
        details = self._read_server_log()
        raise RuntimeError(f"Timed out loading the Hy-MT2 STQ model.\n{details}")

    def _is_model_server(self, base_url: str) -> bool:
        try:
            with urllib.request.urlopen(f"{base_url}/health", timeout=1):
                pass
            with urllib.request.urlopen(f"{base_url}/v1/models", timeout=1) as response:
                models = json.load(response).get("data", [])
            return any(model.get("id") == MODEL_ALIAS for model in models)
        except (OSError, ValueError, AttributeError):
            return False

    def _read_server_log(self):
        path = self.cache_dir / "server.log"
        return path.read_text(encoding="utf-8", errors="replace")[-4000:] if path.exists() else ""

    @staticmethod
    def _sha256(path):
        digest = hashlib.sha256()
        with path.open("rb") as model:
            while chunk := model.read(1024 * 1024):
                digest.update(chunk)
        return digest.hexdigest()

    @staticmethod
    def _free_port():
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            return sock.getsockname()[1]

    @staticmethod
    def _notify(callback, message):
        if callback:
            callback(message)


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
        except (urllib.error.URLError, KeyError, IndexError, TypeError, ValueError) as error:
            raise RuntimeError(f"Hy-MT2 translation failed: {error}") from error
        if not result:
            raise RuntimeError("Hy-MT2 returned an empty translation.")
        return result

    def run_many(self, texts):
        return [self.run(text) for text in texts]


