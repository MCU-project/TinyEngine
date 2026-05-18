#!/bin/sh
set -eu

TOOLROOT="/Applications/STM32CubeIDE.app/Contents/Eclipse/plugins/com.st.stm32cube.ide.mcu.externaltools.gnu-tools-for-stm32.13.3.rel1.macos64_1.0.100.202509120712/tools/bin"
CC="$TOOLROOT/arm-none-eabi-gcc"
SIZE="$TOOLROOT/arm-none-eabi-size"

OUTDIR="tests/n657_toy_infer/build"
mkdir -p "$OUTDIR"

"$CC" \
  -mcpu=cortex-m55 \
  -mthumb \
  -std=c11 \
  -O2 \
  -ffreestanding \
  -fdata-sections \
  -ffunction-sections \
  -fno-builtin \
  -nostdlib \
  -Wl,--gc-sections \
  -T tests/n657_toy_infer/linker.ld \
  tests/n657_toy_infer/main.c \
  -o "$OUTDIR/n657_toy_infer.elf"

"$TOOLROOT/arm-none-eabi-objcopy" -O binary \
  "$OUTDIR/n657_toy_infer.elf" "$OUTDIR/n657_toy_infer.bin"

"$SIZE" "$OUTDIR/n657_toy_infer.elf"
