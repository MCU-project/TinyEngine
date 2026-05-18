TOOLROOT := /Applications/STM32CubeIDE.app/Contents/Eclipse/plugins/com.st.stm32cube.ide.mcu.externaltools.gnu-tools-for-stm32.13.3.rel1.macos64_1.0.100.202509120712/tools/bin
CC := $(TOOLROOT)/arm-none-eabi-gcc
OBJCOPY := $(TOOLROOT)/arm-none-eabi-objcopy
SIZE := $(TOOLROOT)/arm-none-eabi-size

OUTDIR := build
TARGET := $(OUTDIR)/stm32n6570_dk_toy_infer

CFLAGS := \
	-mcpu=cortex-m55 \
	-mthumb \
	-std=c11 \
	-O2 \
	-ffreestanding \
	-fdata-sections \
	-ffunction-sections \
	-fno-builtin \
	-nostdlib

LDFLAGS := \
	-Wl,--gc-sections \
	-T linker.ld

all: $(TARGET).elf $(TARGET).bin

$(OUTDIR):
	mkdir -p $(OUTDIR)

$(TARGET).elf: src/main.c linker.ld | $(OUTDIR)
	$(CC) $(CFLAGS) $(LDFLAGS) src/main.c -o $@
	$(SIZE) $@

$(TARGET).bin: $(TARGET).elf
	$(OBJCOPY) -O binary $< $@

clean:
	rm -rf $(OUTDIR)

.PHONY: all clean
