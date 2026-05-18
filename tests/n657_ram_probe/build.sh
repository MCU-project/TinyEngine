#!/bin/sh
set -eu

TOOLROOT="/Applications/STM32CubeIDE.app/Contents/Eclipse/plugins/com.st.stm32cube.ide.mcu.externaltools.gnu-tools-for-stm32.13.3.rel1.macos64_1.0.100.202509120712/tools/bin"
CC="$TOOLROOT/arm-none-eabi-gcc"
OBJCOPY="$TOOLROOT/arm-none-eabi-objcopy"
SIZE="$TOOLROOT/arm-none-eabi-size"

OUTDIR="$(dirname "$0")/build"
mkdir -p "$OUTDIR"

"$CC" \
  -mcpu=cortex-m55 \
  -mthumb \
  -nostdlib \
  -nostartfiles \
  -ffreestanding \
  -Os \
  -Wl,-T,"$(dirname "$0")/linker.ld" \
  -Wl,-Map,"$OUTDIR/n657_ram_probe.map" \
  "$(dirname "$0")/main.c" \
  -o "$OUTDIR/n657_ram_probe.elf"

"$OBJCOPY" -O binary "$OUTDIR/n657_ram_probe.elf" "$OUTDIR/n657_ram_probe.bin"
"$SIZE" "$OUTDIR/n657_ram_probe.elf"
