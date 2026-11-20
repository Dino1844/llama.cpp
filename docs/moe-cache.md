# MoE cache

When the experts of MoE layers stay in host memory (`-cmoe`, `-ncmoe`, `-ot`), the MoE cache keeps the most used ones in device memory for small batches (decode, speculative decoding). Large batches still upload the experts they use.

| Argument | |
|---|---|
| `--moe-cache-mib N` | cache size in MiB, split among the GPUs like the layers (default: 0, disabled) |
| `--moe-cache-cpu-share F` | share of the cache misses computed on the CPU from the host experts instead of uploaded, once the cache is full (default: 0) |

The cache hits run on the device. With `--moe-cache-cpu-share F` the CPU computes a share `F` of the misses while the device computes the hits and uploads the other misses into the cache. This is faster when the CPU reads the host experts faster than the device can upload them, for example with fast RAM, a PCIe 4 link or experts that are not in pinned memory. The host experts are pinned with `-lm none`, `-lm mlock` and `-lm dio`, unless `--no-host`, but not when they are mmap'd (the default). The device uploads pinned experts at the full speed of the link, so the CPU share often does not help then. Until the cache is full every miss is uploaded, so the cache fills as fast as without the CPU share. The best share depends on the machine, the model and the cache size.

## Hexagon

Use the complete GGUF and keep Hexagon runtime repacking enabled. The cache supports routed expert tensors that the backend can store in its tiled quantized layout. Hexagon packs each missed expert into its tiled device slot and keeps the other slots intact. Use `--load-mode mmap --no-host` to read the host weights through the file mapping.
If the CPU puts the experts in a `CPU_REPACK` buffer, add `--no-repack` to keep the host weights in GGUF layout. Hexagon runtime repacking stays enabled.

```sh
llama-cli -m model.gguf --device HTP0 -ngl 99 --cpu-moe \
  --load-mode mmap --no-host --fit off --moe-cache-mib 2048 -c 512 -b 8 -ub 8
```

On Linux or Android, use a packed expert directory to skip repacking on cache misses. Use a complete, single-file GGUF. The packer writes one file per tileable MoE layer and aligns each expert record to 4 KiB. It derives expert counts from each layer's tensors. A host build of the Hexagon libraries can pack weights without a DSP or FastRPC driver.

```sh
LD_LIBRARY_PATH=/path/to/lib python3 scripts/pack-hexagon-experts.py \
  --model model.gguf --lib-dir /path/to/lib --out-dir experts-htp-tiled
```

Use `--verify experts-htp-tiled` instead of `--out-dir` to byte-compare an existing pack against the exact GGUF used for inference and atomically write `index.json`. This also migrates legacy `hexagon-tiled-v1` directories without rewriting their weights. Rerun verification on the execution target after copying or replacing the GGUF. The index records the local identity of the verified file, which the runtime compares to the file it actually loaded. Keep the GGUF and index unchanged during model load and inference.

Use the Snapdragon runner to pass the packed directory and enable HTP completion polling:

```sh
python3 scripts/snapdragon/run.py --hex-oppoll 1 --moe-cache-packed-dir /path/to/experts-htp-tiled -- \
  llama-cli -m /path/to/model.gguf --device HTP0 -ngl 99 --cpu-moe \
  --load-mode mmap --no-host --fit off --moe-cache-mib 4096 -c 2048 -b 512 -ub 32
```

The paths refer to the execution target. Choose the cache size to fit the model and available device memory. When invoking `llama-cli` directly, use `LLAMA_MOE_CACHE_PACKED_DIR` and `GGML_HEXAGON_OPPOLL=1`. Polling can reduce completion latency at the cost of host CPU activity.

On a miss the reader uses one aligned read per expert and checks the record's CRC32 before uploading it. `LLAMA_MOE_CACHE_IO_THREADS` controls the parallel read limit (default 8, range 1-64). Measure sustained decode and prompt processing separately when choosing a value. The reader tries direct I/O first and uses buffered reads when the filesystem rejects direct I/O at open. An unavailable layer file, bad record or failed read switches that layer to the GGUF weights. Layers outside the pack also use the GGUF weights. A requested pack must have a matching index and at least one usable layer at startup. The CRC detects accidental corruption. Use a trusted index produced by the packer.

`GGML_HEXAGON_Q3K_COMPACT=1` enables the compact layout for Q3_K expert banks. It requires matching DSP support. Add `--q3k-compact` when creating or verifying the packed directory for those banks. Other quantizations keep their existing layout.

`GGML_HEXAGON_MOE_HMX_PIPELINE=1` enables the routed HMX pipeline when the tensor shape and VTCM budget permit it. `LLAMA_MOE_CACHE_READY=1` optionally lets eligible multi-row packed batches submit same-layer expert work while missing weights are read and verified. Both options default to off.

`LLAMA_MOE_CACHE_LAYOUT=window` selects per-layer expert windows for batches up to eight rows, including speculative verification. Larger batches use a separate shared bank. `LLAMA_MOE_CACHE_PREFETCH=N` sets the per-layer prefetch budget (default 0, range 0-8). With no layout specified, positive depth selects windows and zero depth selects the shared cache. Set `LLAMA_MOE_CACHE_LAYOUT=window` in both runs when comparing prefetch depths at the same slot capacity. Window mode requires packed experts, enough cache for the windows, and a CPU miss share of zero. The Snapdragon runner exposes these settings as `--moe-cache-layout` and `--moe-cache-prefetch`.

The router predicts experts two layers ahead and refines pending requests from the next prediction. Both predictions share the target layer's budget. Demand joins only in-progress reads for selected experts. The scheduler keeps the selected slots pinned until the graph completes. `LLAMA_MOE_CACHE_IO_THREADS` sets the demand and prefetch pool sizes. Depth zero starts no prefetch workers.

`LLAMA_MOE_CACHE_OVERLAP=1` enables foreground single-miss overlap on supported Hexagon MoE graphs in window mode. The slot lookup reserves the missing slot. The backend submits the resident experts, then reads, checks and fills the reserved slot on the inference thread before submitting the remaining expert. Unsupported batches use the full-graph path. The Snapdragon runner passes these options with `--moe-cache-overlap 1` and `--hex-moe-pacing-file PATH`.

Set `GGML_HEXAGON_MOE_PACING_FILE` to supply an optional protection interval for the first Hexagon batch. An external controller can atomically replace the file with one line of six integers in nanoseconds:

```text
anchor_ns period_ns protected_ns batch_ns max_hold_ns expires_ns
```

`anchor_ns` is a period-aligned start of a protected interval on the host monotonic clock. `batch_ns` is the estimated time from submission to completion of the hit batch. A hit batch that would overlap a protected interval can wait until the end of that interval if it fits the hold budget. The missing expert read proceeds during the hold. Early read completion releases the held dispatch. An expired, missing or malformed file disables holds. Keep this option off to measure the overlap path without pacing.

`LLAMA_MOE_CACHE_PROFILE=1` reports `pread`, CRC and upload timings. The worker sums can overlap. Demand wait includes scheduling and all three stages. The prefetch wait field sums worker elapsed time without implying a stall in inference. CRC CPU time excludes time when the worker is descheduled. `LLAMA_MOE_CACHE_TRACE` also records routed expert IDs and physical slot sizes. The tuning script simulates the shared cache policy.

## Picking the arguments

Put the script in front of your server command:

```bash
python3 scripts/moe-cache-tune.py -- build/bin/llama-server -m model.gguf -cmoe -t 8 -c 32768
```

It starts the server 6 times, at two cache sizes × shares 0, 0.5 and 0.95, and times the generation of a few thousand tokens. That takes a few minutes, mostly model loading.

From these runs it fits the upload and CPU speeds of the machine, then replays the routing of all the runs through a copy of the cache logic for other sizes and shares. It prints the measured speeds, the predicted t/s and the recommended arguments: the lowest share within 5% of the fastest.

- **Server command:** keep your own arguments (`-t`, `-c`, expert placement, load mode); the speeds and the free device memory depend on them.
- **`--max-mib N`:** tunes a smaller cache than what fits on the device. The runs in `--out` are reused while the command stays the same, so trying another value is instant. Use a fresh `--out` directory when the executable, model, pack, or environment changes.
- **`--markdown`:** prints Markdown tables to paste in a PR.

### Using measured decode times

Use `--samples` to choose a cache size from sustained decode measurements. Pass the memory budget with `--max-mib`.

```sh
python3 scripts/moe-cache-tune.py --samples measured.json --max-mib 5120
```

Provide a JSON array with at least three successful measured runs per candidate within the budget. A run has the form `{"mib": 4096, "tokens": 512, "decode_ms": 45000, "wait_ms": 18000}`. Use the same model, prompt, decode length and runtime settings across candidates. Warm the device first and alternate candidate order to expose drift. Use decode wall time for `decode_ms` and the number of steps timed in that interval for `tokens`. In llama.cpp the first generated token comes from the prompt evaluation and does not count as a timed decode step. The optional `wait_ms` records expert read and CRC wait within that same decode interval.

The tool finds the highest median throughput, then looks for the smallest cache that reaches that group's slowest run. The recommended cache also matches or beats the median throughput of every smaller cache. Read and CRC wait is reported as a diagnostic in milliseconds per token.
