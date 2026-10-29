#!/usr/bin/env python3
"""Download a single GGUF from Hugging Face without loading it into memory."""

import argparse
import hashlib
import os
from pathlib import Path, PurePosixPath
import re
import sys
import tempfile
from urllib.parse import quote, urlparse
from urllib.request import Request, urlopen

CHUNK_BYTES = 1024 * 1024


def model_url(repo, filename, revision):
    if not re.fullmatch(r"[\w.-]+/[\w.-]+", repo) or any(
        part in (".", "..") for part in repo.split("/")
    ):
        raise ValueError("--repo must be an owner/repository name")
    path = PurePosixPath(filename)
    if (not filename or path.is_absolute() or "\\" in filename
            or any(p in ("", ".", "..") for p in filename.split("/"))
            or not filename.lower().endswith(".gguf")):
        raise ValueError("--file must be a relative GGUF filename without traversal")
    if not revision or any(ord(c) < 32 for c in revision):
        raise ValueError("--revision must not be empty or contain control characters")
    return (f"https://huggingface.co/{repo}/resolve/"
            f"{quote(revision, safe='')}/{quote(filename, safe='/')}")


def download(repo, filename, revision="main", output_dir="models", sha256=None,
             force=False, opener=urlopen):
    """Stream to a temporary file and publish only after validation succeeds.

    Non-force publication uses a hard link to atomically refuse an existing
    destination, including one created by another downloader during the transfer.
    """
    url = model_url(repo, filename, revision)
    if sha256 is not None and not re.fullmatch(r"[a-fA-F0-9]{64}", sha256):
        raise ValueError("--sha256 must contain exactly 64 hexadecimal characters")
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / PurePosixPath(filename).name
    if not force and os.path.lexists(destination):
        raise FileExistsError(f"{destination} already exists; use --force to replace it")
    request = Request(url, headers={"User-Agent": "Pocket-Inference/0.1",
                                    "Accept-Encoding": "identity"})
    temporary = None
    try:
        with opener(request, timeout=60) as response:
            if urlparse(response.geturl()).scheme != "https":
                raise ValueError("Refusing a non-HTTPS download redirect")
            length = response.headers.get("Content-Length")
            expected_size = int(length) if length is not None else None
            if expected_size is not None and expected_size < 4:
                raise ValueError("Response is too small to contain a GGUF model")
            digest = hashlib.sha256()
            size = 0
            magic = bytearray()
            with tempfile.NamedTemporaryFile(dir=directory, prefix=".download-",
                                             suffix=".partial", delete=False) as output:
                temporary = Path(output.name)
                while True:
                    chunk = response.read(CHUNK_BYTES)
                    if not chunk:
                        break
                    magic.extend(chunk[:max(0, 4 - len(magic))])
                    if len(magic) == 4 and magic != b"GGUF":
                        raise ValueError("Downloaded response is not a GGUF file")
                    output.write(chunk)
                    digest.update(chunk)
                    size += len(chunk)
                if len(magic) != 4:
                    raise ValueError("Downloaded response is not a GGUF file")
                if expected_size is not None and size != expected_size:
                    raise ValueError(f"Incomplete download: expected {expected_size}, got {size} bytes")
                actual_sha = digest.hexdigest()
                if sha256 is not None and actual_sha != sha256.lower():
                    raise ValueError("SHA-256 mismatch; the downloaded file was discarded")
                output.flush()
                os.fsync(output.fileno())
            if force:
                os.replace(temporary, destination)
            else:
                os.link(temporary, destination)
            return destination, size, actual_sha
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default="unsloth/Qwen3-0.6B-GGUF")
    parser.add_argument("--file", default="Qwen3-0.6B-Q4_K_M.gguf")
    parser.add_argument("--revision", default="main", help="Prefer a commit SHA for reproducibility")
    parser.add_argument("--output-dir", default="models")
    parser.add_argument("--sha256", help="Expected SHA-256 from a trusted source")
    parser.add_argument("--force", action="store_true", help="Replace an existing destination after validation")
    args = parser.parse_args()
    try:
        path, size, checksum = download(args.repo, args.file, args.revision,
                                        args.output_dir, args.sha256, args.force)
    except (OSError, ValueError) as error:
        print(f"Download failed: {error}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("Download cancelled; temporary file removed.", file=sys.stderr)
        return 130
    print(f"Saved {path} ({size:,} bytes)\nSHA-256: {checksum}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
