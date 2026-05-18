#!/usr/bin/env python3
"""Sweep reduced LLaVA/Qwen configs for MCU memory planning."""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Iterable, List

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from estimate_llava_mcu_memory import LlavaConfig, estimate_memory


HIDDEN_SIZES = (128, 256, 384, 512)
DECODER_LAYERS = (2, 4, 6, 8)
MLP_RATIOS = (2.0, 3.0, 4.0)
HEAD_CANDIDATES = (4, 6, 8)
NUM_KV_HEADS = (1, 2)
VISION_TOKENS = (16, 32, 64, 128)
TEXT_TOKENS = (32, 64, 128)
VOCAB_SIZES = (1024, 4096, 8192)

WEIGHT_BUDGET_MB = 80.0
KV_CACHE_BUDGET_MB = 4.0
ACTIVATION_BUDGET_MB = 4.0
RUNTIME_BUDGET_MB = 32.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sweep reduced LLaVA/Qwen configs for MCU memory feasibility."
    )
    parser.add_argument(
        "--vision-params",
        type=int,
        default=0,
        help="Fixed vision tower parameter count for all candidates. Default: 0",
    )
    parser.add_argument(
        "--vision-hidden",
        type=int,
        default=256,
        help="Fixed projector input width for all candidates. Default: 256",
    )
    parser.add_argument(
        "--dtype-bytes",
        type=int,
        default=4,
        help="Bytes per element. Default: 4",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results"),
        help="Directory where sweep CSV files are written.",
    )
    return parser.parse_args()


def iter_candidate_configs(args: argparse.Namespace) -> Iterable[LlavaConfig]:
    hidden_size: int
    decoder_layers: int
    mlp_ratio: float
    num_heads: int
    num_kv_heads: int
    vision_tokens: int
    text_tokens: int
    vocab_size: int

    for hidden_size in HIDDEN_SIZES:
        valid_heads = [heads for heads in HEAD_CANDIDATES if hidden_size % heads == 0]
        for decoder_layers in DECODER_LAYERS:
            for mlp_ratio in MLP_RATIOS:
                for num_heads in valid_heads:
                    for num_kv_heads in NUM_KV_HEADS:
                        if num_heads % num_kv_heads != 0:
                            continue
                        for vision_tokens in VISION_TOKENS:
                            for text_tokens in TEXT_TOKENS:
                                for vocab_size in VOCAB_SIZES:
                                    yield LlavaConfig(
                                        vision_params=args.vision_params,
                                        vision_tokens=vision_tokens,
                                        vision_hidden=args.vision_hidden,
                                        decoder_layers=decoder_layers,
                                        hidden_size=hidden_size,
                                        num_heads=num_heads,
                                        num_kv_heads=num_kv_heads,
                                        mlp_ratio=mlp_ratio,
                                        text_tokens=text_tokens,
                                        vocab_size=vocab_size,
                                        dtype_bytes=args.dtype_bytes,
                                    )


def evaluate_candidate(cfg: LlavaConfig) -> Dict[str, object]:
    estimate = estimate_memory(cfg)
    activation_total = estimate["activation_MB"] + estimate["attention_score_MB"]
    reasons: List[str] = []

    if estimate["total_weight_MB"] > WEIGHT_BUDGET_MB:
        reasons.append("weight_over_budget")
    if estimate["kv_cache_MB"] > KV_CACHE_BUDGET_MB:
        reasons.append("kv_cache_over_budget")
    if activation_total > ACTIVATION_BUDGET_MB:
        reasons.append("activation_over_budget")
    if estimate["total_runtime_MB"] > RUNTIME_BUDGET_MB:
        reasons.append("runtime_over_budget")

    row: Dict[str, object] = {
        **asdict(cfg),
        **estimate,
        "activation_plus_attention_MB": activation_total,
        "feasible": "yes" if not reasons else "no",
        "reason": ";".join(reasons),
    }
    return row


def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    all_rows = [evaluate_candidate(cfg) for cfg in iter_candidate_configs(args)]
    feasible_rows = [row for row in all_rows if row["feasible"] == "yes"]
    feasible_rows.sort(key=lambda row: (row["total_weight_MB"], row["total_runtime_MB"]))

    args.results_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.results_dir / "mcu_config_sweep.csv", all_rows)
    write_csv(args.results_dir / "mcu_config_feasible.csv", feasible_rows)


if __name__ == "__main__":
    main()
