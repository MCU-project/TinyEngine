ENTRY(Reset_Handler)

_estack = ORIGIN(DTCM) + LENGTH(DTCM);

MEMORY
{
  DTCM (xrw)  : ORIGIN = 0x20000000, LENGTH = 256K
}

SECTIONS
{
  .isr_vector :
  {
    . = ALIGN(256);
    KEEP(*(.isr_vector))
    . = ALIGN(4);
  } > DTCM

  .text :
  {
    . = ALIGN(4);
    *(.text*)
    *(.rodata*)
    . = ALIGN(4);
    _etext = .;
  } > DTCM

  _sidata = LOADADDR(.data);

  .data :
  {
    . = ALIGN(4);
    _sdata = .;
    *(.data*)
    . = ALIGN(4);
    _edata = .;
  } > DTCM

  .bss (NOLOAD) :
  {
    . = ALIGN(8);
    _sbss = .;
    *(.bss*)
    *(COMMON)
    . = ALIGN(8);
    _ebss = .;
  } > DTCM
}
