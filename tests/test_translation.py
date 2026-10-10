import json
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from src.translate import DEFAULT_TRANSLATION_ENGINE, create_translator
from src.translate.hy_mt2 import HyMT2


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


if __name__ == "__main__":
    unittest.main()
