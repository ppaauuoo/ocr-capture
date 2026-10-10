"""Build, cache, and manage a local llama.cpp server for translation engines."""
import hashlib
import json
import os
import shutil
import socket
import subprocess
import threading
import time
import urllib.request
from pathlib import Path


BUILD_TOOLS_HINT = (
    "Install Git, CMake, and a C++ compiler "
    "(Visual Studio Build Tools with the C++ workload on Windows)."
)


class LlamaCppRuntime:
    name = "local model"
    cache_name = "llama.cpp"
    repository = ""
    commit = ""
    model_repository = ""
    model_revision = ""
    model_filename = ""
    model_size = 0
    model_sha256 = ""
    model_alias = ""
    context_size = 8192
    server_arguments = ("--n-gpu-layers", "0", "--jinja")

    def __init__(self, cache_dir=None):
        data_root = Path(os.environ.get("LOCALAPPDATA") or Path.home() / ".cache")
        self.cache_dir = Path(cache_dir).resolve() if cache_dir else data_root / "ocr-capture" / self.cache_name
        self._lock = threading.Lock()
        self._process_lock = threading.Lock()
        self._process: subprocess.Popen | None = None
        self._build_process: subprocess.Popen | None = None
        self._base_url: str | None = None
        self._closing = threading.Event()

    def ensure_ready(self, status_callback=None) -> str:
        with self._lock:
            self._raise_if_closing()
            with self._process_lock:
                process, base_url = self._process, self._base_url
            if process and process.poll() is None:
                if base_url and self._is_model_server(base_url):
                    return base_url
                self._discard_server(process)

            binary = self._ensure_server_binary(status_callback)
            self._raise_if_closing()
            model = self._ensure_model(status_callback)
            self._raise_if_closing()

            port = self._free_port()
            base_url = f"http://127.0.0.1:{port}"
            command = [
                str(binary), "--model", str(model), "--alias", self.model_alias,
                "--host", "127.0.0.1", "--port", str(port),
                "--ctx-size", str(self.context_size), *self.server_arguments,
            ]
            self._notify(status_callback, f"Starting the {self.name} local runtime…")
            log_path = self.cache_dir / "server.log"
            creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
            with log_path.open("ab") as log:
                with self._process_lock:
                    self._raise_if_closing()
                    process = subprocess.Popen(
                        command, stdout=log, stderr=subprocess.STDOUT,
                        creationflags=creationflags,
                    )
                    self._process = process
                    self._base_url = base_url
            try:
                self._wait_until_ready(status_callback, process, base_url)
            except Exception:
                self._discard_server(process)
                raise
            return base_url

    def shutdown(self):
        self._closing.set()
        with self._process_lock:
            processes = self._build_process, self._process
            self._build_process = None
            self._process = None
            self._base_url = None
        for process in processes:
            self._stop_process(process)

    @staticmethod
    def _stop_process(process):
        if process is None or process.poll() is not None:
            return
        try:
            process.terminate()
        except OSError:
            return
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()

    def _discard_server(self, process):
        self._stop_process(process)
        with self._process_lock:
            if self._process is process:
                self._process = None
                self._base_url = None

    def _raise_if_closing(self):
        if self._closing.is_set():
            raise RuntimeError(f"{self.name} runtime is shutting down.")

    def _ensure_server_binary(self, status_callback):
        source = self.cache_dir / f"llama.cpp-{self.commit[:12]}"
        build = source / "build"
        binary_name = "llama-server.exe" if os.name == "nt" else "llama-server"
        candidates = (
            build / "bin" / "Release" / binary_name,
            build / "bin" / binary_name,
        )

        def find_binary():
            for candidate in candidates:
                if candidate.is_file():
                    return candidate
            return next(build.rglob(binary_name), None) if build.exists() else None

        binary = find_binary()
        if binary:
            return binary

        self._notify(status_callback, f"Preparing the {self.name} llama.cpp runtime (first use)…")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        git_config = source / ".git" / "config"
        config_text = (
            git_config.read_text(encoding="utf-8", errors="replace")
            if git_config.is_file()
            else ""
        )
        if '[remote "origin"]' not in config_text or f"url = {self.repository}" not in config_text:
            if source.exists():
                shutil.rmtree(source)
            source.mkdir(parents=True)
            self._run_logged(["git", "init", "--quiet"], source)
            self._run_logged(["git", "remote", "add", "origin", self.repository], source)
        self._run_logged(["git", "fetch", "--depth", "1", "origin", self.commit], source)
        self._run_logged(["git", "checkout", "--detach", "FETCH_HEAD"], source)
        self._run_logged([
            "cmake", "-S", str(source), "-B", str(build),
            "-DCMAKE_BUILD_TYPE=Release", "-DGGML_CUDA=OFF", "-DGGML_VULKAN=OFF",
        ], source)
        self._run_logged([
            "cmake", "--build", str(build), "--config", "Release",
            "--target", "llama-server", "--parallel", "4",
        ], source, timeout=1800)
        binary = find_binary()
        if binary:
            return binary
        raise RuntimeError(f"llama.cpp build completed without producing {binary_name}.")

    def _ensure_model(self, status_callback):
        model = self.cache_dir / self.model_filename
        if model.is_file():
            self._notify(status_callback, f"Checking the cached {self.name} model…")
            if self._sha256(model) != self.model_sha256:
                raise RuntimeError(f"Cached {self.name} model failed its SHA-256 check: {model}")
            return model

        self._notify(status_callback, f"Downloading the {self.name} model…")
        url = (
            f"https://huggingface.co/{self.model_repository}/resolve/"
            f"{self.model_revision}/{self.model_filename}?download=true"
        )
        part = model.with_suffix(model.suffix + ".part")
        digest = hashlib.sha256()
        received = 0
        last_percent = -1
        try:
            with urllib.request.urlopen(url, timeout=60) as response, part.open("wb") as output:
                while chunk := response.read(1024 * 1024):
                    self._raise_if_closing()
                    output.write(chunk)
                    digest.update(chunk)
                    received += len(chunk)
                    percent = received * 100 // self.model_size
                    if percent != last_percent:
                        self._notify(status_callback, f"Downloading {self.name} model… {percent}%")
                        last_percent = percent
            if received != self.model_size or digest.hexdigest() != self.model_sha256:
                raise RuntimeError(f"Downloaded {self.name} model has an unexpected size or SHA-256 hash.")
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
                with self._process_lock:
                    self._raise_if_closing()
                    process = subprocess.Popen(
                        command, cwd=cwd, stdout=log, stderr=subprocess.STDOUT,
                    )
                    self._build_process = process
                try:
                    return_code = process.wait(timeout=timeout)
                except subprocess.TimeoutExpired as error:
                    self._stop_process(process)
                    raise RuntimeError(
                        f"Timed out while running {command[0]} to prepare the {self.name} runtime."
                    ) from error
                finally:
                    with self._process_lock:
                        if self._build_process is process:
                            self._build_process = None
        except FileNotFoundError as error:
            raise RuntimeError(
                f"Required build tool {command[0]!r} was not found. {BUILD_TOOLS_HINT}"
            ) from error
        if return_code:
            details = log_path.read_text(encoding="utf-8", errors="replace")[-4000:]
            raise RuntimeError(
                f"Could not prepare the {self.name} llama.cpp runtime ({command[0]} exited "
                f"{return_code}). {BUILD_TOOLS_HINT} Build log: {log_path}\n{details}"
            )

    def _wait_until_ready(self, status_callback, process, base_url):
        deadline = time.monotonic() + 300
        while time.monotonic() < deadline:
            if process.poll() is not None:
                details = self._read_server_log()
                raise RuntimeError(f"{self.name} runtime exited while loading its model.\n{details}")
            if self._is_model_server(base_url):
                return
            self._notify(status_callback, f"Loading the {self.name} model into the local runtime…")
            time.sleep(1)
        details = self._read_server_log()
        raise RuntimeError(f"Timed out loading the {self.name} model.\n{details}")

    def _is_model_server(self, base_url: str) -> bool:
        try:
            with urllib.request.urlopen(f"{base_url}/health", timeout=1):
                pass
            with urllib.request.urlopen(f"{base_url}/v1/models", timeout=1) as response:
                models = json.load(response).get("data", [])
            return any(model.get("id") == self.model_alias for model in models)
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
