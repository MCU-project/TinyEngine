ENTRY(Reset_Handler)

MEMORY
{
  DTCM (xrw) : ORIGIN = 0x20000000, LENGTH = 256K
}

_estack = ORIGIN(DTCM) + LENGTH(DTCM);

SECTIONS
{
  .isr_vector :
  {
    KEEP(*(.isr_vector))
  } > DTCM

  .text :
  {
    *(.text*)
    *(.rodata*)
  } > DTCM

  .data :
  {
    _sdata = .;
    *(.data*)
    _edata = .;
  } > DTCM

  .bss :
  {
    _sbss = .;
    *(.bss*)
    *(COMMON)
    _ebss = .;
  } > DTCM
}
