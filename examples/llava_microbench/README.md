# LLaVA / Qwen Host Microbenchmark

This directory contains a host-side profiling skeleton for the FP32 `LLaVA/Qwen` path before MCU porting.

Current scope:
- individual FP32 operator benchmark entries
- `mm_projector_fp` benchmark
- `lm_head_last_token_fp` benchmark
- `kv_cache` size/append/read benchmark
- `image_text_fusion_fp` benchmark
- `qwen_block_fp` benchmark with multiple config variants
- CSV output for easy post-processing
- host-side memory estimator script

Not included:
- STM32 / MCU porting
- quantization
- real LLaVA weight loading
- accuracy validation against framework outputs

## Host Build

Windows MinGW example:

```powershell
gcc examples\llava_microbench\main.c `
  TinyEngine\src\kernels\fp_backward_op\add_fp.c `
  TinyEngine\src\kernels\fp_backward_op\image_text_fusion_fp.c `
  TinyEngine\src\kernels\fp_backward_op\kv_cache_fp.c `
  TinyEngine\src\kernels\fp_backward_op\llm_ops_fp.c `
  TinyEngine\src\kernels\fp_backward_op\lm_head_last_token_fp.c `
  TinyEngine\src\kernels\fp_backward_op\mm_projector_fp.c `
  TinyEngine\src\kernels\fp_backward_op\mul_fp.c `
  TinyEngine\src\kernels\fp_backward_op\qwen_block_fp.c `
  -I TinyEngine\include `
  -std=c11 -Wall -Wextra -pedantic -lm `
  -o qwen_block_microbench_host.exe
```

POSIX-style example:

```bash
gcc examples/llava_microbench/main.c \
  TinyEngine/src/kernels/fp_backward_op/add_fp.c \
  TinyEngine/src/kernels/fp_backward_op/image_text_fusion_fp.c \
  TinyEngine/src/kernels/fp_backward_op/kv_cache_fp.c \
  TinyEngine/src/kernels/fp_backward_op/llm_ops_fp.c \
  TinyEngine/src/kernels/fp_backward_op/lm_head_last_token_fp.c \
  TinyEngine/src/kernels/fp_backward_op/mm_projector_fp.c \
  TinyEngine/src/kernels/fp_backward_op/mul_fp.c \
  TinyEngine/src/kernels/fp_backward_op/qwen_block_fp.c \
  -I TinyEngine/include \
  -std=c11 -Wall -Wextra -pedantic -lm \
  -o qwen_block_microbench_host
```

## Host Run

Windows PowerShell:

```powershell
& .\qwen_block_microbench_host.exe
```

POSIX shell:

```bash
./qwen_block_microbench_host
```

## CSV Output

The program prints a single CSV header followed by one CSV row per benchmark.

Columns:

```text
benchmark,variant,seq_len,hidden_size,intermediate_size,vision_hidden,num_image_tokens,text_tokens,num_heads,num_kv_heads,rotary_dim,head_dim,num_layers,vocab_size,dtype_bytes,runs,workspace_bytes,cache_bytes,elapsed_ms,avg_ms,checksum,status
```

Typical benchmark rows include:
- `operator,rmsnorm_fp,...`
- `operator,rotary_embedding_fp,...`
- `operator,repeat_kv_fp,...`
- `operator,silu_fp,...`
- `operator,gelu_fp,...`
- `operator,qkv_attention_fp,...`
- `operator,embedding_lookup_fp,...`
- `operator,argmax_fp,...`
- `wrapper,mm_projector_fp,...`
- `wrapper,lm_head_last_token_fp,...`
- `wrapper,kv_cache_size,...`
- `wrapper,kv_cache_append_fp,...`
- `wrapper,kv_cache_read_fp,...`
- `wrapper,image_text_fusion_fp,...`
- `wrapper,qwen_toy_s4_h32_i64,...`
- `wrapper,qwen_toy_s8_h64_i128,...`
- `wrapper,qwen_toy_s16_h128_i256,...`

`checksum` is only a lightweight sanity check so that silent all-zero or invalid outputs are easier to notice.

## What The Driver Covers

`main.c` benchmarks:
- `rmsnorm_fp`
- `rotary_embedding_fp`
- `repeat_kv_fp`
- `silu_fp`
- `gelu_fp`
- `qkv_attention_fp`
- `embedding_lookup_fp`
- `argmax_fp`
- `mm_projector_fp`
- `lm_head_last_token_fp`
- `kv_cache_size`
- `kv_cache_append_fp`
- `kv_cache_read_fp`
- `image_text_fusion_fp`
- `qwen_block_fp`

`lm_head_last_token_fp` is measured with:
- `vocab_size = 1024`
- `vocab_size = 4096`
- `vocab_size = 8192`
- `vocab_size = 16384`
- `vocab_size = 151647`

`image_text_fusion_fp` is measured with:
- `image_tokens = 16`
- `image_tokens = 32`
- `image_tokens = 64`
- `image_tokens = 196`
- `image_tokens = 729`

`kv_cache` is measured with several `(num_layers, seq_len, num_kv_heads, head_dim)` presets so cache growth is visible directly in CSV.

`qwen_block_fp` runs the following order:
- `RMSNorm`
- `Q Linear`
- `K Linear`
- `V Linear`
- `RoPE`
- `repeat_kv`
- `GQA attention`
- `O Linear`
- residual add
- `RMSNorm`
- `gate Linear`
- `up Linear`
- `SiLU`
- gated multiply
- `down Linear`
- residual add

## Notes

- Dummy inputs and weights are initialized conservatively to avoid `NaN` / `Inf`.
- The main purpose is reproducible latency and workspace profiling structure on host.
- `TinyEngine/src/kernels/fp_backward_op/qwen_block_fp.c` and `mm_projector_fp.c` are wrappers, not optimized kernels.
- If you later move to MCU, keep the CSV schema unchanged so host and board results are easy to compare.

## Memory Estimator

`tools/estimate_llava_mcu_memory.py` gives a rough host-side memory estimate for LLaVA shrink studies before MCU porting.

Example:

```powershell
python tools\estimate_llava_mcu_memory.py `
  --vision-params 433000000 `
  --vision-tokens 729 `
  --vision-hidden 1152 `
  --decoder-layers 24 `
  --hidden-size 896 `
  --num-heads 14 `
  --num-kv-heads 2 `
  --mlp-ratio 5.428571 `
  --text-tokens 32 `
  --vocab-size 151647 `
  --dtype-bytes 4
```

The script reports:
- `vision_weight_MB`
- `projector_weight_MB`
- `decoder_weight_MB`
- `embedding_weight_MB`
- `lm_head_weight_MB`
- `kv_cache_MB`
- `activation_MB`
- `attention_score_MB`
- `total_weight_MB`
- `total_runtime_MB`

These estimates are intentionally aligned to the current host microbenchmark skeleton:
- `attention_score_MB` follows the current `qkv_attention_fp` scratch shape of `seq_len x seq_len`
- `activation_MB` is a rough peak across projector, fusion, and one decoder-block stage
- the output is for shrink-direction planning, not framework-exact runtime accounting

## MCU Config Sweep

`tools/sweep_llava_mcu_configs.py` walks a reduced LLaVA/Qwen search space and writes:
- `results/mcu_config_sweep.csv`
- `results/mcu_config_feasible.csv`

Default assumptions:
- `vision_params = 0`
- `vision_hidden = 256`
- `dtype_bytes = 4`

These are defaults so decoder/projector-side shrink trends are visible even before a final vision tower choice is fixed. You can override them from the command line.

Example:

```powershell
python tools\sweep_llava_mcu_configs.py
```

Feasibility budgets:
- `total_weight_MB <= 80`
- `kv_cache_MB <= 4`
- `activation_MB + attention_score_MB <= 4`
- `total_runtime_MB <= 32`
