set pagination off
set confirm off
file tests/n657_ram_probe/build/n657_ram_probe.elf
target remote :61235
monitor halt
load
set $sp = 0x20040000
set $pc = 0x200000dd
set $xpsr = 0x01000000
tbreak HardFault_Handler
tbreak Default_Handler
tbreak *0x20000134
continue
info registers pc xpsr sp
x/8wx 0x20000174
quit
