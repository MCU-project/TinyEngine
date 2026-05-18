#!/usr/bin/env python3
"""Estimate LLaVA/Qwen FP32 memory for host-side shrink studies.

This script is intentionally simple and follows the current TinyEngine
microbenchmark skeleton rather than framework-exact tensor lifetimes.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict


BYTES_PER_MB = 1024.0 * 1024.0


@dataclass
class LlavaConfig:
    vision_params: int
    vision_tokens: int
    vision_hidden: int
    decoder_layers: int
    hidden_size: int
    num_heads: int
    num_kv_heads: int
    mlp_ratio: float
    text_tokens: int
    vocab_size: int
    dtype_bytes: int


def _load_config_from_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Estimate LLaVA/Qwen memory for TinyEngine host-side planning."
    )
    parser.add_argument("--config", type=Path, help="Optional JSON config path.")
    parser.add_argument("--vision-params", type=int)
    parser.add_argument("--vision-tokens", type=int)
    parser.add_argument("--vision-hidden", type=int)
    parser.add_argument("--decoder-layers", type=int)
    parser.add_argument("--hidden-size", type=int)
    parser.add_argument("--num-heads", type=int)
    parser.add_argument("--num-kv-heads", type=int)
    parser.add_argument("--mlp-ratio", type=float)
    parser.add_argument("--text-tokens", type=int)
    parser.add_argument("--vocab-size", type=int)
    parser.add_argument("--dtype-bytes", type=int, default=4)
    parser.add_argument(
        "--format",
        choices=("json", "csv"),
        default="json",
        help="Output format.",
    )
    return parser.parse_args()


def _build_config(args: argparse.Namespace) -> LlavaConfig:
    raw: Dict[str, Any] = {}
    if args.config is not None:
        raw.update(_load_config_from_json(args.config))

    cli_values = {
        "vision_params": args.vision_params,
        "vision_tokens": args.vision_tokens,
        "vision_hidden": args.vision_hidden,
        "decoder_layers": args.decoder_layers,
        "hidden_size": args.hidden_size,
        "num_heads": args.num_heads,
        "num_kv_heads": args.num_kv_heads,
        "mlp_ratio": args.mlp_ratio,
        "text_tokens": args.text_tokens,
        "vocab_size": args.vocab_size,
        "dtype_bytes": args.dtype_bytes,
    }
    for key, value in cli_values.items():
        if value is not None:
            raw[key] = value

    missing = [key for key in cli_values if key not in raw or raw[key] is None]
    if missing:
        raise SystemExit(f"missing required config fields: {', '.join(missing)}")

    return LlavaConfig(**raw)


def _bytes_to_mb(value: float) -> float:
    return value / BYTES_PER_MB


def estimate_memory(cfg: LlavaConfig) -> Dict[str, float]:
    if cfg.num_heads <= 0 or cfg.num_kv_heads <= 0:
        raise ValueError("num_heads and num_kv_heads must be positive")
    if cfg.hidden_size % cfg.num_heads != 0:
        raise ValueError("hidden_size must be divisible by num_heads")
    if cfg.num_heads % cfg.num_kv_heads != 0:
        raise ValueError("num_heads must be divisible by num_kv_heads")

    seq_len = cfg.vision_tokens + cfg.text_tokens
    head_dim = cfg.hidden_size // cfg.num_heads
    intermediate_size = int(round(cfg.hidden_size * cfg.mlp_ratio))
    kv_hidden = cfg.num_kv_heads * head_dim

    vision_weight_bytes = cfg.vision_params * cfg.dtype_bytes

    projector_weight_params = (
        cfg.vision_hidden * cfg.hidden_size
        + cfg.hidden_size
        + cfg.hidden_size * cfg.hidden_size
        + cfg.hidden_size
    )
    projector_weight_bytes = projector_weight_params * cfg.dtype_bytes

    decoder_layer_params = (
        2 * cfg.hidden_size
        + cfg.hidden_size * cfg.hidden_size
        + cfg.hidden_size
        + cfg.hidden_size * kv_hidden
        + kv_hidden
        + cfg.hidden_size * kv_hidden
        + kv_hidden
        + cfg.hidden_size * cfg.hidden_size
        + cfg.hidden_size
        + cfg.hidden_size * intermediate_size
        + intermediate_size
        + cfg.hidden_size * intermediate_size
        + intermediate_size
        + intermediate_size * cfg.hidden_size
        + cfg.hidden_size
    )
    decoder_weight_bytes = decoder_layer_params * cfg.decoder_layers * cfg.dtype_bytes

    embedding_weight_bytes = cfg.vocab_size * cfg.hidden_size * cfg.dtype_bytes
    lm_head_weight_bytes = (cfg.vocab_size * cfg.hidden_size + cfg.vocab_size) * cfg.dtype_bytes

    kv_cache_bytes = (
        2
        * cfg.decoder_layers
        * seq_len
        * cfg.num_kv_heads
        * head_dim
        * cfg.dtype_bytes
    )

    projector_stage_bytes = (
        cfg.vision_tokens * cfg.vision_hidden
        + 2 * cfg.vision_tokens * cfg.hidden_size
    ) * cfg.dtype_bytes
    fusion_stage_bytes = (
        cfg.vision_tokens * cfg.hidden_size
        + cfg.text_tokens * cfg.hidden_size
        + seq_len * cfg.hidden_size
    ) * cfg.dtype_bytes

    qwen_non_score_floats = (
        6 * seq_len * cfg.hidden_size
        + 2 * seq_len * kv_hidden
        + 2 * seq_len * intermediate_size
    )
    decoder_stage_bytes = (
        2 * seq_len * cfg.hidden_size + qwen_non_score_floats
    ) * cfg.dtype_bytes
    activation_bytes = max(projector_stage_bytes, fusion_stage_bytes, decoder_stage_bytes)

    attention_score_bytes = (seq_len * seq_len) * cfg.dtype_bytes

    total_weight_bytes = (
        vision_weight_bytes
        + projector_weight_bytes
        + decoder_weight_bytes
        + embedding_weight_bytes
        + lm_head_weight_bytes
    )
    total_runtime_bytes = (
        total_weight_bytes
        + kv_cache_bytes
        + activation_bytes
        + attention_score_bytes
    )

    return {
        "vision_weight_MB": _bytes_to_mb(vision_weight_bytes),
        "projector_weight_MB": _bytes_to_mb(projector_weight_bytes),
        "decoder_weight_MB": _bytes_to_mb(decoder_weight_bytes),
        "embedding_weight_MB": _bytes_to_mb(embedding_weight_bytes),
        "lm_head_weight_MB": _bytes_to_mb(lm_head_weight_bytes),
        "kv_cache_MB": _bytes_to_mb(kv_cache_bytes),
        "activation_MB": _bytes_to_mb(activation_bytes),
        "attention_score_MB": _bytes_to_mb(attention_score_bytes),
        "total_weight_MB": _bytes_to_mb(total_weight_bytes),
        "total_runtime_MB": _bytes_to_mb(total_runtime_bytes),
    }


def _print_csv(cfg: LlavaConfig, estimate: Dict[str, float]) -> None:
    payload = {
        **asdict(cfg),
        **estimate,
    }
    keys = list(payload.keys())
    print(",".join(keys))
    print(",".join(str(payload[key]) for key in keys))


def main() -> None:
    args = _parse_args()
    cfg = _build_config(args)
    estimate = estimate_memory(cfg)

    if args.format == "csv":
        _print_csv(cfg, estimate)
    else:
        print(json.dumps(estimate, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
