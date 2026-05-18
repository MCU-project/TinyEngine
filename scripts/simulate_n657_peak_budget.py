#!/usr/bin/env python3

SRAM_TOTAL = int(4.2 * 1024 * 1024)

SCENARIOS = [
    ("Current VWW example", 177_704),
    ("Current face mask example", 262_568),
    ("Target 3.0 MiB peak", 3 * 1024 * 1024),
    ("Target 3.5 MiB peak", int(3.5 * 1024 * 1024)),
    ("Target 4.0 MiB peak", 4 * 1024 * 1024),
]

OVERHEADS = [
    ("Minimal runtime reserve", 64 * 1024),
    ("Practical runtime reserve", 128 * 1024),
    ("Conservative runtime reserve", 256 * 1024),
]


def fmt_bytes(n: int) -> str:
    return f"{n} B"


def fmt_kib(n: int) -> str:
    return f"{n / 1024:.1f} KiB"


def fmt_mib(n: int) -> str:
    return f"{n / 1024 / 1024:.3f} MiB"


def main() -> None:
    print(f"STM32N6570-DK internal SRAM total: {fmt_bytes(SRAM_TOTAL)} ({fmt_mib(SRAM_TOTAL)})")
    print()
    for name, peak in SCENARIOS:
        free_after_model = SRAM_TOTAL - peak
        print(name)
        print(f"  Model peak: {fmt_bytes(peak)} ({fmt_mib(peak)})")
        print(f"  Free after model: {fmt_bytes(free_after_model)} ({fmt_kib(free_after_model)})")
        for overhead_name, reserve in OVERHEADS:
            remaining = free_after_model - reserve
            status = "OK" if remaining >= 0 else "FAIL"
            print(
                f"  {overhead_name}: reserve {fmt_kib(reserve)} -> remaining {fmt_kib(remaining)} [{status}]"
            )
        print()


if __name__ == "__main__":
    main()
