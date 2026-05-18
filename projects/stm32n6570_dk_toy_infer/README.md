# STM32N6570-DK Toy Inference

Minimal standalone project for `STM32N6570-DK` memory-budget testing.

What it does:
- runs from `DTCM` (`0x20000000`)
- allocates a fake RGB input buffer (`96x96x3`)
- touches a `3 MiB` model arena at `0x34100000`
- writes a small output buffer
- exposes result flags/checksum in globals for debugger reads

Artifacts:
- `build/stm32n6570_dk_toy_infer.elf`
- `build/stm32n6570_dk_toy_infer.bin`

Build:

```sh
make -C projects/stm32n6570_dk_toy_infer
```

Notes:
- This project is intended for debugger-driven RAM execution.
- The current board session still has unstable memory access, so build success does not imply runtime verification success.
