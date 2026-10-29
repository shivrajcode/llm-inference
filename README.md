# Pocket Inference

A small local chat application for quantized Qwen models on modest laptops. C++20 owns inference and serves a dependency-free browser UI. llama.cpp supplies the GGUF loader, tokenizer, quantized kernels, and CPU/GPU backends. This is an application built on llama.cpp, not a new tensor engine or a vLLM-compatible server.

## Quick start

Requires CMake 3.20+, a C++20 compiler, and Python 3.9+ for the optional downloader. macOS and Linux are covered by CI. Windows is not yet validated. The initial build downloads pinned dependencies; inference is offline once a model is present.

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel 2
# Qwen3 1.7B, quantized to Q4_K_M (~1.11 GB download)
python3 scripts/download_model.py \
  --repo unsloth/Qwen3-1.7B-GGUF --file Qwen3-1.7B-Q4_K_M.gguf
./build/pocket-inference --models ./models
```

Open **http://127.0.0.1:8080**, select the model, and click **Load model**. Existing GGUF files can be copied into `models/`, or point `--models` at their directory. Files are loaded from disk, not uploaded through the browser. Refresh the list after adding a file. Only top-level `.gguf` files are listed. Use instruction/chat, decoder-only models with a supported embedded chat template. Raw Hugging Face safetensors are not accepted; obtain a compatible GGUF conversion.

For the smaller Qwen3 0.6B Q4 model, run `python3 scripts/download_model.py` with no arguments. Qwen3's small model sizes are 0.6B and 1.7B; this project does not relabel either as exactly 1B. Other compatible approximately 1B GGUF models can also be loaded. Qwen3 may emit reasoning text; add `/no_think` to a prompt to request a short response.

## Memory behavior

Defaults target a 4 GB RAM CPU laptop: 2,048 context tokens, 128-token prompt batches, 2 inference threads, zero GPU layers, and one active generation. Weights are memory mapped without locking them into RAM; old weights are released before loading a replacement. KV state is cleared between requests. History is sent explicitly by the browser and kept only in that tab. No Python process is needed during inference.

A roughly 1B Q4 model is a practical target, but total memory includes the OS, browser, weights, KV cache, and working buffers. Memory mapping is not a hard RAM cap, and swapping can hurt responsiveness. See [memory tuning](docs/MEMORY.md). This project makes no measured 4 GB performance guarantee.

```sh
# Smaller working context for devices under memory pressure
./build/pocket-inference --context 1024 --batch 64 --threads 2
# Optional Apple Metal acceleration (CPU-only is the default)
cmake -S . -B build-metal -DCMAKE_BUILD_TYPE=Release -DGGML_METAL=ON
cmake --build build-metal --parallel 2
./build-metal/pocket-inference --gpu-layers 99
```

Run `./build/pocket-inference --help` for options, including `--web` if you move the executable away from its source checkout. Oversized conversations produce an explicit error; start a new chat or reduce the reply limit. Stop cancels generation at a batch/token boundary. No silent history truncation occurs. Concurrent model operations receive HTTP 409 instead of allocating another model context.

## Development and verification

```sh
ctest --test-dir build --output-on-failure
python3 -m unittest discover -s tests -p 'test_*.py' -v
node tests/ui_test.cjs
# Optional real-model runtime checks
./build/engine-tests models/Qwen3-1.7B-Q4_K_M.gguf
# Instrument C++ and dependencies for address/undefined behavior checks
cmake -S . -B build-asan -DCMAKE_BUILD_TYPE=Debug -DPOCKET_SANITIZE=ON
cmake --build build-asan --parallel 2
ctest --test-dir build-asan --output-on-failure
```

C++ is not inherently memory-safe. Application code uses RAII smart pointers for backend resources, explicit limits, serialized access, and checked filesystem paths. Sanitizers support validation; they do not prove all third-party native code safe. Load GGUF files from trusted sources. Downloader writes bounded chunks to a temporary file and publishes only after validation. Supply `--revision COMMIT --sha256 HASH` to verify a known artifact; a printed hash alone does not authenticate its publisher.

The HTTP API is local and intentionally small: `GET /api/status`, `GET /api/models`, `POST /api/load {"path":"file.gguf"}`, `POST /api/unload {}`, and `POST /api/chat {"messages":[{"role":"user","content":"Hello"}],"max_tokens":256,"temperature":0.7}`. Chat returns SSE JSON token events followed by completion metrics or an error. It is not OpenAI API compatible. The server binds to 127.0.0.1, validates Host/Origin, and accepts JSON mutations only. Do not expose it through a public proxy.

## Dependencies

- [llama.cpp](https://github.com/ggml-org/llama.cpp), b6500 / `a7a98e0fffed794396b3fbad4dcdbbc184963645` (MIT)
- [cpp-httplib](https://github.com/yhirose/cpp-httplib), v0.20.0 (MIT)
- [nlohmann/json](https://github.com/nlohmann/json), v3.12.0 (MIT)
- Model weights have their own licenses: [Qwen3](https://huggingface.co/Qwen/Qwen3-1.7B), [Unsloth GGUF conversion](https://huggingface.co/unsloth/Qwen3-1.7B-GGUF).

Dependency downloads are version-pinned and SHA-256 checked. `vendor/`, build outputs, and downloaded weights are ignored by Git. An existing `vendor/llama.cpp` can be used for an offline build; supply `vendor/httplib.h` and `vendor/json.hpp` too.
