# Running on a 4 GB laptop

The target is a quantized model around one billion parameters on a 4 GB CPU-only
laptop. Installed RAM includes the operating system, browser, and other apps;
4 GB is not the amount available to the inference process. Responsiveness also
depends on CPU memory bandwidth and storage. A hard RAM ceiling or a particular
token rate is not guaranteed.

Qwen3's small dense models are 0.6B and 1.7B parameters, not exactly 1B.
The downloadable Q4_K_M weights are approximately 397 MB and 1.11 GB respectively
([0.6B files](https://huggingface.co/unsloth/Qwen3-0.6B-GGUF/tree/main),
[1.7B files](https://huggingface.co/unsloth/Qwen3-1.7B-GGUF/tree/main)).
These are community quantizations of Qwen models. File size is only part of the
runtime memory requirement. Q4_K_M includes mixed tensor precisions and metadata;
the naive parameter-count times four-bit estimate understates its size.

The default downloader gets 0.6B Q4_K_M. To try the larger model on a 4 GB machine:

```sh
python3 scripts/download_model.py \
  --repo unsloth/Qwen3-1.7B-GGUF \
  --file Qwen3-1.7B-Q4_K_M.gguf
```

Use `--revision` with a Hugging Face commit SHA and `--sha256` with a checksum
from a trusted source for a reproducible, verified download. The downloader
uses one-megabyte reads, rejects invalid magic and truncated transfers, and
publishes the destination only after the transfer and optional checksum pass.
It does not resume interrupted transfers. GGUF magic is a basic format check,
not full model validation; the inference engine validates loading separately.

## What the application does to limit memory

- Loads a single model and uses one generation at a time.
- Memory-maps model weights; it does not read the entire file into a second
  application-owned buffer. Mapped pages still consume physical memory when
  resident. Memory mapping is not a hard memory limit.
- Leaves memory locking off so the OS can reclaim mapped pages under pressure.
- Defaults to CPU execution, automatic context sizing, batches of 128, and two
  CPU threads. Larger contexts increase the KV cache; larger batches increase
  temporary compute memory. Threads are principally a speed/CPU-load setting.
- Keeps the web interface dependency-free, with no bundled frontend runtime or
  Python inference server. Python is needed only for the download helper/tests.
- Automatically reserves reply space and omits the oldest completed conversation
  turns when necessary. System messages and the latest user message are retained;
  the completion event reports the number of omitted messages. If those retained
  messages are too large, the request fails rather than cutting the current prompt.
  An explicit positive API `max_tokens` uses strict mode: oversized history fails
  instead of being trimmed.

The KV cache stores intermediate attention keys and values. Its size depends on
layer count, KV heads, head dimensions, cache precision, and context length; model
parameter count alone cannot predict it. The process also needs tokenizer data,
compute buffers, backend allocations, and HTTP request/response buffers.

## Automatic context selection

Context is chosen when a model loads. The engine reads the model's trained
context limit and samples available host memory, then:

1. Subtracts the model file size and reserves the greater of 512 MiB or one
   quarter of available memory.
2. Budgets half the remaining memory for the KV cache, leaving the other half
   for runtime allocations and estimation error.
3. Estimates F16 KV bytes per token as
   `2 * layers * KV_heads * (key_length + value_length)`. If the needed metadata
   is unavailable, it uses a conservative 256 KiB per token estimate.
4. Chooses the minimum of the model's trained context, the affordable token count,
   and an 8,192-token automatic ceiling; rounds down to a multiple of 256; and
   refuses loading with a clear memory error if fewer than 512 tokens fit.

If host available-memory detection fails, the estimator uses 1 GiB available.
This fallback can refuse a model that would fit on a machine with more memory;
an explicit `--context` is available for informed tuning. Automatic sizing does
not use swap as a reason to grow the context and does not promise OOM prevention.
Other applications can allocate memory after this snapshot, and backend memory
requirements vary. An explicit context should remain within the model's trained
limit and still needs measurement on the target machine.

The browser uses automatic reply length: the smaller of 512 tokens or one quarter
of the effective context. This reserves output space before old turns are
removed. If preserved instructions and the latest message still leave less room,
the reply allowance shrinks to the remaining space; a prompt with no output
space is rejected. The API's `context` and `history_trimmed` completion fields describe the
actual request. `GET /api/status` exposes selected context, trained model context,
and whether context selection was automatic. Temperature defaults to 0.7.

## Measure before increasing limits

Begin with automatic context and CPU mode. If memory pressure or swapping is
high, close other applications and reduce context and batch settings. Unload the
model when finished. GPU offload on integrated graphics can still consume the
same system RAM and is not inherently a memory saving.

On Linux, use `/usr/bin/time -v` around the server command to capture maximum
resident set size after exiting. On macOS, use `/usr/bin/time -l`; its maximum
resident set size is reported in bytes, whereas Linux reports KiB. Monitor swap
and memory pressure with the OS tools during generation, not only idle loading.

Record model repository/revision, quantization, context, batch, CPU, OS, available
RAM before launch, peak RSS, and prompt/generation token counts and durations.
Exercise a near-context-limit chat as well as a short prompt. Repeat on actual
4 GB hardware: a successful run on a larger developer machine does not establish
comfortable operation on a 4 GB laptop.

## Historical measurement with a fixed context

On 2026-10-29, a Release CPU build on an Apple M2 with 16 GiB RAM running
macOS 27.0.1 passed the real-model runtime checks at an explicit 2,048-token
context, batch 128, and two threads. `/usr/bin/time -l` reported a maximum
resident set size of **1,437,941,760 bytes (1.34 GiB)**. This is process RSS,
not system memory usage or a 4 GB device benchmark. It predates automatic context
selection and does not describe the current default memory footprint. The test exercised short
16-token replies, independent repeated generations, cancellation before prefill,
context overflow rejection, and unload/reload. It did not fill the entire context
or include browser memory. No sustained throughput claim follows from this test.

Artifact: `unsloth/Qwen3-1.7B-GGUF/Qwen3-1.7B-Q4_K_M.gguf`, 1,107,409,472 bytes,
observed SHA-256 `b139949c5bd74937ad8ed8c8cf3d9ffb1e99c866c823204dc42c0d91fa181897`.
Downloaded from `main`; the digest identifies the tested bytes but was not
independently authenticated against a publisher checksum.

To measure the current automatic defaults, time the server, load the model in the
browser, record the selected context, run representative chats, then exit:

```sh
/usr/bin/time -l ./build/pocket-inference --models models
# Linux: replace -l with -v
```
