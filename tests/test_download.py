import hashlib
import importlib.util
import io
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location(
    "download_model", Path(__file__).resolve().parents[1] / "scripts/download_model.py")
downloader = importlib.util.module_from_spec(spec)
spec.loader.exec_module(downloader)


class Response(io.BytesIO):
    def __init__(self, payload, length=None):
        super().__init__(payload)
        self.headers = {"Content-Length": str(len(payload) if length is None else length)}
        self.read_sizes = []

    def geturl(self):
        return "https://cdn.example/model.gguf"

    def read(self, size=-1):
        if size <= 0 or size > downloader.CHUNK_BYTES:
            raise AssertionError("Download reads must have a bounded chunk size")
        self.read_sizes.append(size)
        return super().read(size)


class DownloadTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.folder = Path(self.directory.name)

    def run_download(self, payload=b"GGUFtest", **kwargs):
        response = Response(payload, kwargs.pop("length", None))
        result = downloader.download("owner/model", "model.gguf",
                                     output_dir=self.folder,
                                     opener=lambda request, timeout: response, **kwargs)
        return result, response

    def assert_no_partial(self):
        self.assertEqual(list(self.folder.glob("*.partial")), [])

    def test_stream_checksum_and_publish(self):
        payload = b"GGUF" + b"x" * (downloader.CHUNK_BYTES * 2 + 31)
        digest = hashlib.sha256(payload).hexdigest()
        (path, size, checksum), response = self.run_download(payload, sha256=digest)
        self.assertEqual(path.read_bytes(), payload)
        self.assertEqual(size, len(payload))
        self.assertEqual(checksum, digest)
        self.assertGreaterEqual(len(response.read_sizes), 4)
        self.assert_no_partial()

    def test_checksum_failure_discards_file(self):
        with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
            self.run_download(sha256="0" * 64)
        self.assertFalse((self.folder / "model.gguf").exists())
        self.assert_no_partial()

    def test_partial_transfer_is_not_published(self):
        with self.assertRaisesRegex(ValueError, "Incomplete download"):
            self.run_download(length=999)
        self.assertFalse((self.folder / "model.gguf").exists())
        self.assert_no_partial()

    def test_invalid_gguf_is_rejected(self):
        for payload in (b"<html>not a model</html>", b"GG"):
            with self.assertRaises(ValueError):
                self.run_download(payload)
            self.assert_no_partial()

    def test_existing_destination_preserved(self):
        destination = self.folder / "model.gguf"
        destination.write_bytes(b"original")
        with self.assertRaises(FileExistsError):
            self.run_download()
        self.assertEqual(destination.read_bytes(), b"original")
        with self.assertRaises(ValueError):
            self.run_download(b"invalid", force=True)
        self.assertEqual(destination.read_bytes(), b"original")
        self.run_download(force=True)
        self.assertEqual(destination.read_bytes(), b"GGUFtest")

    def test_publication_race_does_not_overwrite(self):
        def open_response(request, timeout):
            (self.folder / "model.gguf").write_bytes(b"other download")
            return Response(b"GGUFtest")
        with self.assertRaises(FileExistsError):
            downloader.download("owner/model", "model.gguf", output_dir=self.folder,
                                opener=open_response)
        self.assertEqual((self.folder / "model.gguf").read_bytes(), b"other download")
        self.assert_no_partial()

    def test_network_error_cleans_partial(self):
        class Broken(Response):
            def read(self, size):
                raise OSError("disconnected")
        with self.assertRaises(OSError):
            downloader.download("owner/model", "model.gguf", output_dir=self.folder,
                                opener=lambda request, timeout: Broken(b"GGUFtest"))
        self.assert_no_partial()

    def test_paths_and_hash_are_validated(self):
        for filename in ("../escape.gguf", "/escape.gguf", "x/../escape.gguf",
                         "x\\escape.gguf", "x//m.gguf", "model.bin"):
            with self.assertRaises(ValueError):
                downloader.model_url("owner/model", filename, "main")
        for repo in ("https://example.com", "../model", "a/b/c", "model"):
            with self.assertRaises(ValueError):
                downloader.model_url(repo, "model.gguf", "main")
        with self.assertRaises(ValueError):
            self.run_download(sha256="not a hash")
        self.assertIn("refs%2Fpr%2F1", downloader.model_url("owner/model", "m.gguf", "refs/pr/1"))


if __name__ == "__main__":
    unittest.main()
