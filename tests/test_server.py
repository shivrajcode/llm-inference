"""HTTP integration tests; set POCKET_TEST_MODEL for opt-in real inference."""

import http.client
import json
import os
from pathlib import Path
import socket
import subprocess
import tempfile
import time
import unittest

ROOT = Path(__file__).resolve().parents[1]
BINARY = Path(os.environ.get("POCKET_TEST_BINARY", str(ROOT / "build/pocket-inference")))


class ServerFixture(unittest.TestCase):
    launch_options = ["--context", "512", "--batch", "64"]

    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.folder = Path(cls.temporary.name)
        cls.models = cls.folder / "models"
        cls.models.mkdir()
        cls.configure_models()
        with socket.socket() as reservation:
            reservation.bind(("127.0.0.1", 0))
            cls.port = reservation.getsockname()[1]
        cls.log = (cls.folder / "server.log").open("w+")
        cls.process = subprocess.Popen(
            [str(BINARY), "--models", str(cls.models), "--web", str(ROOT / "web"),
             "--port", str(cls.port), *cls.launch_options],
            stdout=cls.log, stderr=subprocess.STDOUT)
        try:
            deadline = time.monotonic() + 15
            while time.monotonic() < deadline:
                if cls.process.poll() is not None:
                    break
                try:
                    connection = http.client.HTTPConnection("127.0.0.1", cls.port, timeout=1)
                    connection.request("GET", "/api/status")
                    response = connection.getresponse()
                    response.read()
                    connection.close()
                    if response.status == 200:
                        return
                except OSError:
                    pass
                time.sleep(0.05)
            cls.log.seek(0)
            raise RuntimeError("Server did not become ready:\n" + cls.log.read())
        except BaseException:
            cls.tearDownClass()
            raise

    @classmethod
    def configure_models(cls):
        pass

    @classmethod
    def tearDownClass(cls):
        if cls.process.poll() is None:
            cls.process.terminate()
            try:
                cls.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                cls.process.kill()
                cls.process.wait()
        cls.log.close()
        cls.temporary.cleanup()

    def request(self, method, path, body=None, headers=None, raw=False, timeout=120):
        request_headers = {"Content-Type": "application/json"}
        request_headers.update(headers or {})
        if body is not None and not raw:
            body = json.dumps(body)
        connection = http.client.HTTPConnection("127.0.0.1", self.port, timeout=timeout)
        try:
            connection.request(method, path, body=body, headers=request_headers)
            response = connection.getresponse()
            payload = response.read().decode("utf-8")
            return response.status, dict(response.getheaders()), payload
        finally:
            connection.close()

    def assert_error(self, status, result):
        self.assertEqual(result[0], status, result[2])
        self.assertIsInstance(json.loads(result[2])["error"], str)

    def completion(self, request_body):
        status, headers, payload = self.request("POST", "/api/chat", request_body)
        self.assertEqual(status, 200, payload)
        self.assertIn("text/event-stream", headers["Content-Type"])
        events = [json.loads(line[6:]) for line in payload.splitlines() if line.startswith("data: ")]
        self.assertTrue(events, payload)
        self.assertFalse(any("error" in event for event in events), events)
        self.assertTrue(events[-1].get("done"), events)
        return events[-1]


class ServerTests(ServerFixture):
    @classmethod
    def configure_models(cls):
        (cls.models / "broken.gguf").write_bytes(b"GGUFinvalid")
        (cls.models / "notes.txt").write_text("not a model")
        cls.outside = cls.folder / "outside.gguf"
        cls.outside.write_bytes(b"GGUFoutside")
        (cls.models / "escape.gguf").symlink_to(cls.outside)

    def test_static_ui_and_security_headers(self):
        status, headers, body = self.request("GET", "/")
        self.assertEqual(status, 200)
        self.assertIn("Pocket", body)
        self.assertIn("text/html", headers["Content-Type"])
        self.assertEqual(headers["X-Content-Type-Options"], "nosniff")
        self.assertIn("frame-ancestors 'none'", headers["Content-Security-Policy"])

    def test_status_and_model_listing(self):
        result = self.request("GET", "/api/status")
        self.assertEqual(result[0], 200)
        status = json.loads(result[2])
        self.assertFalse(status["loaded"])
        self.assertEqual(status["context"], 512)
        result = self.request("GET", "/api/models")
        models = json.loads(result[2])["models"]
        self.assertEqual([model["path"] for model in models], ["broken.gguf"])
        self.assertEqual(models[0]["size_bytes"], 11)

    def test_malformed_and_missing_load_parameters(self):
        for body in ("{", "null", "[]", "{}", '{"path":42}'):
            self.assert_error(400, self.request("POST", "/api/load", body, raw=True))

    def test_unloaded_chat_and_idempotent_unload(self):
        self.assert_error(400, self.request("POST", "/api/chat", {
            "messages": [{"role": "user", "content": "hello"}]}))
        for _ in range(2):
            result = self.request("POST", "/api/unload", {})
            self.assertEqual(result[0], 200)
            self.assertFalse(json.loads(result[2])["loaded"])

    def test_host_and_origin_protection(self):
        self.assert_error(403, self.request("GET", "/api/status", headers={"Host": "evil.example"}))
        for origin in ("https://evil.example", "null", "http://localhost:9999"):
            self.assert_error(403, self.request("POST", "/api/unload", {}, headers={"Origin": origin}))
        result = self.request("POST", "/api/unload", {}, headers={"Origin": f"http://127.0.0.1:{self.port}"})
        self.assertEqual(result[0], 200)
        self.assert_error(403, self.request("POST", "/api/unload", "{}",
                                           headers={"Content-Type": "text/plain"}, raw=True))

    def test_model_path_confinement_and_invalid_model(self):
        for path in ("../outside.gguf", str(self.outside), "escape.gguf", "notes.txt",
                     "", "broken.gguf\u0000", "missing.gguf", "broken.gguf"):
            self.assert_error(400, self.request("POST", "/api/load", {"path": path}))
        self.assertFalse(json.loads(self.request("GET", "/api/status")[2])["loaded"])

    def test_oversized_request_is_rejected(self):
        result = self.request("POST", "/api/load", "x" * (1024 * 1024 + 1), raw=True)
        self.assertEqual(result[0], 413)


@unittest.skipUnless(os.environ.get("POCKET_TEST_MODEL"), "Set POCKET_TEST_MODEL to test real inference")
class RealModelTests(ServerFixture):
    @classmethod
    def configure_models(cls):
        cls.model = Path(os.environ["POCKET_TEST_MODEL"]).resolve(strict=True)
        cls.models = cls.model.parent

    def test_real_generation_validation_and_recovery(self):
        result = self.request("POST", "/api/load", {"path": self.model.name})
        self.assertEqual(result[0], 200, result[2])
        loaded = json.loads(self.request("GET", "/api/status")[2])
        self.assertFalse(loaded["auto_context"])
        self.assertEqual(loaded["context"], 512)
        self.assertLessEqual(loaded["context"], loaded["model_context"])
        message = {"messages": [{"role": "user", "content": "Say hello briefly."}], "max_tokens": 8}
        for limit in (0, -1, 513, 1.5, 2**64, "8"):
            self.assert_error(400, self.request("POST", "/api/chat", dict(message, max_tokens=limit)))
        for temperature in (-1, 3, "warm"):
            self.assert_error(400, self.request("POST", "/api/chat", dict(message, temperature=temperature)))

        # Synchronize on the prefill heartbeat, not an arbitrary sleep. The
        # short input leaves room for 400 output tokens in our 512-token context.
        # Request a long answer so a normal completion cannot race the busy check
        # during the first prompt-evaluation batch on the target CPU backend.
        active = http.client.HTTPConnection("127.0.0.1", self.port, timeout=30)
        stream = None
        try:
            active.request("POST", "/api/chat", json.dumps({
                "messages": [{"role": "user", "content":
                    "Write a very long detailed numbered list of 100 practical tips "
                    "for learning programming. Explain every tip in several sentences."}],
                "max_tokens": 400, "temperature": 0,
            }), headers={"Content-Type": "application/json"})
            stream = active.getresponse()
            self.assertEqual(stream.status, 200)
            self.assertEqual(stream.readline(1024).strip(), b": ping")
            self.assert_error(409, self.request("GET", "/api/status", timeout=5))
        finally:
            # HTTPResponse retains a socket file; close both ends explicitly so
            # cancellation reaches the server even while unread chunks remain.
            if active.sock is not None:
                try:
                    active.sock.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass
            if stream is not None:
                stream.close()
            active.close()

        deadline = time.monotonic() + 30
        while True:
            recovered = self.request("GET", "/api/status", timeout=2)
            if recovered[0] == 200:
                self.assertTrue(json.loads(recovered[2])["loaded"])
                break
            self.assert_error(409, recovered)
            if time.monotonic() >= deadline:
                self.fail("Inference remained busy for 30 seconds after disconnect")
            time.sleep(0.05)

        # A new generation must succeed after cancellation, not merely status.
        status, headers, payload = self.request("POST", "/api/chat", message)
        self.assertEqual(status, 200)
        self.assertIn("text/event-stream", headers["Content-Type"])
        events = [json.loads(line[6:]) for line in payload.splitlines() if line.startswith("data: ")]
        self.assertFalse(any("error" in event for event in events), events)
        self.assertTrue(events[-1]["done"])
        self.assertGreater(events[-1]["prompt_tokens"], 0)
        self.assertLessEqual(events[-1]["generated_tokens"], 8)
        self.assertEqual(events[-1]["context"], 512)
        self.assertFalse(events[-1]["history_trimmed"])

        # Only messages are required: output length and temperature are automatic.
        automatic = self.completion({"messages": [
            {"role": "user", "content": "Reply with just OK. /no_think"}]})
        self.assertEqual(automatic["context"], 512)
        self.assertFalse(automatic["history_trimmed"])
        self.assertLessEqual(automatic["prompt_tokens"] + automatic["generated_tokens"], 512)

        # Old turns must be discarded as complete turns while preserving the
        # system message and latest user request; the response reports trimming.
        trimmed = self.completion({"messages": [
            {"role": "system", "content": "Respond very briefly."},
            {"role": "user", "content": "old question " * 512},
            {"role": "assistant", "content": "old answer " * 512},
            {"role": "user", "content": "Reply with just OK. /no_think"},
        ]})
        self.assertEqual(trimmed["history_trimmed"], 2)
        self.assertEqual(trimmed["context"], 512)
        self.assertGreater(trimmed["prompt_tokens"], 0)
        self.assertLessEqual(trimmed["prompt_tokens"] + trimmed["generated_tokens"], 512)

        # Auto mode must still reject an oversized current user message.
        result = self.request("POST", "/api/chat", {"messages": [
            {"role": "user", "content": "hello " * 1024}]})
        self.assertIn('"error"', result[2])
        long_message = {"messages": [{"role": "user", "content": "hello " * 1024}], "max_tokens": 8}
        result = self.request("POST", "/api/chat", long_message)
        self.assertIn('"error"', result[2])
        result = self.request("POST", "/api/chat", message)
        self.assertIn('"done":true', result[2])
        self.assertEqual(self.request("POST", "/api/unload", {})[0], 200)


@unittest.skipUnless(os.environ.get("POCKET_TEST_MODEL"), "Set POCKET_TEST_MODEL to test real inference")
class AutomaticContextTests(ServerFixture):
    launch_options = ["--batch", "64"]

    @classmethod
    def configure_models(cls):
        cls.model = Path(os.environ["POCKET_TEST_MODEL"]).resolve(strict=True)
        cls.models = cls.model.parent

    def test_default_context_selected_from_model(self):
        result = self.request("POST", "/api/load", {"path": self.model.name})
        self.assertEqual(result[0], 200, result[2])
        status = json.loads(self.request("GET", "/api/status")[2])
        self.assertTrue(status["loaded"])
        self.assertTrue(status["auto_context"])
        self.assertGreater(status["context"], 0)
        self.assertLessEqual(status["context"], status["model_context"])
        self.assertEqual(self.request("POST", "/api/unload", {})[0], 200)


if __name__ == "__main__":
    unittest.main()
