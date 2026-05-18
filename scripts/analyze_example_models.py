#!/usr/bin/env python3

import re
from pathlib import Path


SRAM_N6570_DK = 4.2 * 1024 * 1024

MODELS = [
    Path("experimental/vww/lib/TinyEngine/codegen/Include/genModel.h"),
    Path("examples/openmv_face_mask_detection/codegen/Include/genModel.h"),
]


def parse_model(header_path: Path) -> dict:
    text = header_path.read_text(errors="ignore")

    sram_match = re.search(r"(?:#define PEAK_MEM|/\* sram:)(?:\s*|)(\d+)", text)
    flash_match = re.search(r"(?:#define MODEL_SIZE|flash:)(?:\s*|)(\d+)", text)
    buffer_match = re.search(r"static signed char buffer\[(\d+)\];", text)
    sbuf_match = re.search(r"const int SBuffer_size = (\d+);", text)
    kbuf_match = re.search(r"const int KBuffer_size = (\d+);", text)

    if not sram_match and not buffer_match:
        raise ValueError(f"Could not find SRAM usage in {header_path}")

    peak_mem = int(sram_match.group(1)) if sram_match else int(buffer_match.group(1))
    model_size = int(flash_match.group(1)) if flash_match else None
    buffer_size = int(buffer_match.group(1)) if buffer_match else peak_mem
    sbuf_size = int(sbuf_match.group(1)) if sbuf_match else None
    kbuf_size = int(kbuf_match.group(1)) if kbuf_match else None

    return {
        "name": str(header_path.parent.parent.parent),
        "header": str(header_path),
        "peak_mem": peak_mem,
        "model_size": model_size,
        "buffer_size": buffer_size,
        "sbuf_size": sbuf_size,
        "kbuf_size": kbuf_size,
        "pct_of_n657_sram": peak_mem / SRAM_N6570_DK * 100.0,
    }


def fmt_kib(value: int | None) -> str:
    if value is None:
        return "-"
    return f"{value / 1024:.1f} KiB"


def fmt_mib(value: int | None) -> str:
    if value is None:
        return "-"
    return f"{value / 1024 / 1024:.3f} MiB"


def main() -> None:
    rows = [parse_model(path) for path in MODELS]

    print("STM32N6570-DK SRAM baseline: 4.2 MiB internal SRAM")
    print()
    for row in rows:
        print(f"Model: {row['name']}")
        print(f"  Header: {row['header']}")
        print(f"  Peak memory: {row['peak_mem']} B ({fmt_kib(row['peak_mem'])}, {fmt_mib(row['peak_mem'])})")
        print(f"  Model size: {row['model_size']} B ({fmt_kib(row['model_size'])}, {fmt_mib(row['model_size'])})")
        print(f"  Buffer: {row['buffer_size']} B ({fmt_kib(row['buffer_size'])})")
        print(f"  SBuffer: {fmt_kib(row['sbuf_size'])}")
        print(f"  KBuffer: {fmt_kib(row['kbuf_size'])}")
        print(f"  N657 SRAM share: {row['pct_of_n657_sram']:.2f}%")
        print()


if __name__ == "__main__":
    main()
