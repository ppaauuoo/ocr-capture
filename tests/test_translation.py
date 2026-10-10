import json
import os
import tempfile
import threading
import unittest
from http.client import IncompleteRead
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import Mock, patch

from src.translate import DEFAULT_TRANSLATION_ENGINE, create_translator
from src.translate.hy_mt2 import HyMT2, HyMT2Runtime, LLAMA_CPP_COMMIT, LLAMA_CPP_REPOSITORY


class TranslationEndpoint(BaseHTTPRequestHandler):
    request_data = None

    def do_POST(self):
        size = int(self.headers["Content-Length"])
        type(self).request_data = json.loads(self.rfile.read(size))
        body = json.dumps({"choices": [{"message": {"content": "Hello"}}]}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args):
        pass


class FakeRuntime:
    def __init__(self, url):
        self.url = url

    def ensure_ready(self, status_callback=None):
        return self.url


class TranslationTests(unittest.TestCase):
    def test_default_engine_translates_using_local_hy_mt2_runtime(self):
        server = ThreadingHTTPServer(("127.0.0.1", 0), TranslationEndpoint)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(server.server_close)
        self.addCleanup(thread.join)
        self.addCleanup(server.shutdown)
        TranslationEndpoint.request_data = None

        selected = create_translator(DEFAULT_TRANSLATION_ENGINE)
        self.assertIsInstance(selected, HyMT2)
        engine = HyMT2(runtime=FakeRuntime(f"http://127.0.0.1:{server.server_port}"))
        self.assertEqual(engine.run("こんにちは"), "Hello")
        request = TranslationEndpoint.request_data
        if request is None:
            self.fail("translation request was not sent")
        self.assertEqual(request["model"], "hy-mt2-1.8b-stq")
        self.assertIn("Translate the following text into English", request["messages"][0]["content"])
        self.assertIn("こんにちは", request["messages"][0]["content"])

    def test_unknown_engine_is_rejected_instead_of_falling_back(self):
        with self.assertRaisesRegex(ValueError, "Unknown translation engine"):
            create_translator("unavailable")

    def test_runtime_failure_is_not_hidden_or_rerouted(self):
        class FailedRuntime:
            def ensure_ready(self, status_callback=None):
                raise RuntimeError("STQ kernel is unavailable")

        engine = HyMT2(runtime=FailedRuntime())
        with self.assertRaisesRegex(RuntimeError, "STQ kernel is unavailable"):
            engine.run("こんにちは")

    def test_incomplete_response_is_wrapped_as_translation_error(self):
        engine = HyMT2(runtime=FakeRuntime("http://127.0.0.1:12345"))
        with patch(
            "src.translate.hy_mt2.urllib.request.urlopen",
            side_effect=IncompleteRead(b"partial response"),
        ):
            with self.assertRaisesRegex(RuntimeError, "Hy-MT2 translation failed"):
                engine.run("こんにちは")


class RuntimeLifecycleTests(unittest.TestCase):
    def test_unhealthy_runtime_is_restarted(self):
        with tempfile.TemporaryDirectory() as cache_dir:
            runtime = HyMT2Runtime(cache_dir)
            stale_process = Mock()
            stale_process.poll.return_value = None
            started_process = Mock()
            started_process.poll.return_value = None
            runtime._process = stale_process
            runtime._base_url = "http://127.0.0.1:12345"

            with (
                patch.object(runtime, "_is_model_server", return_value=False),
                patch.object(runtime, "_ensure_server_binary", return_value=Path("llama-server")),
                patch.object(runtime, "_ensure_model", return_value=Path("model.gguf")),
                patch.object(runtime, "_wait_until_ready"),
                patch("src.translate.hy_mt2.subprocess.Popen", return_value=started_process),
            ):
                runtime.ensure_ready()

            stale_process.terminate.assert_called_once()
            self.assertIs(runtime._process, started_process)
            runtime.shutdown()

    def test_partial_git_setup_is_recreated(self):
        with tempfile.TemporaryDirectory() as cache_dir:
            runtime = HyMT2Runtime(cache_dir)
            source = Path(cache_dir) / f"llama.cpp-{LLAMA_CPP_COMMIT[:12]}"
            (source / ".git").mkdir(parents=True)
            (source / ".git" / "config").write_text("[core]\nrepositoryformatversion = 0\n")
            commands = []
            binary_name = "llama-server.exe" if os.name == "nt" else "llama-server"

            def record_command(command, cwd, timeout=300):
                commands.append(command)
                if command[:2] == ["cmake", "--build"]:
                    binary = Path(command[2]) / "bin" / binary_name
                    binary.parent.mkdir(parents=True, exist_ok=True)
                    binary.touch()

            runtime._run_logged = record_command
            binary = runtime._ensure_server_binary(None)
            self.assertEqual(binary.name, binary_name)
            self.assertIn(
                ["git", "remote", "add", "origin", LLAMA_CPP_REPOSITORY],
                commands,
            )
            self.assertLess(
                commands.index(["git", "remote", "add", "origin", LLAMA_CPP_REPOSITORY]),
                commands.index(["git", "fetch", "--depth", "1", "origin", LLAMA_CPP_COMMIT]),
            )

    def test_shutdown_does_not_wait_for_runtime_initialization(self):
        with tempfile.TemporaryDirectory() as cache_dir:
            runtime = HyMT2Runtime(cache_dir)
            entered_build = threading.Event()
            release_build = threading.Event()
            shutdown_done = threading.Event()
            errors = []

            def blocking_build(status_callback):
                entered_build.set()
                release_build.wait()
                return Path("llama-server")

            def unexpected_model_download(status_callback):
                raise AssertionError("shutdown should cancel initialization before model download")

            runtime._ensure_server_binary = blocking_build
            runtime._ensure_model = unexpected_model_download

            def ensure_runtime():
                try:
                    runtime.ensure_ready()
                except Exception as error:
                    errors.append(error)

            worker = threading.Thread(target=ensure_runtime, daemon=True)
            worker.start()
            self.assertTrue(entered_build.wait(2))
            closer = threading.Thread(
                target=lambda: (runtime.shutdown(), shutdown_done.set()),
                daemon=True,
            )
            closer.start()
            try:
                self.assertTrue(shutdown_done.wait(1), "shutdown blocked on runtime initialization")
            finally:
                release_build.set()
                worker.join(2)
                closer.join(2)
            self.assertIsInstance(errors[0], RuntimeError)
            self.assertIn("shutting down", str(errors[0]))

if __name__ == "__main__":
    unittest.main()
