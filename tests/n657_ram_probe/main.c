#include <stdint.h>

#define SCB_CFSR (*(volatile uint32_t *)0xE000ED28u)
#define SCB_HFSR (*(volatile uint32_t *)0xE000ED2Cu)
#define SCB_BFAR (*(volatile uint32_t *)0xE000ED38u)

#define OK_MAGIC 0x600D0001u
#define FAIL_MAGIC 0xBAD00001u

extern uint32_t _estack;
extern uint32_t _sidata;
extern uint32_t _sdata;
extern uint32_t _edata;
extern uint32_t _sbss;
extern uint32_t _ebss;

void Reset_Handler(void);
void reset_main(void);

volatile uint32_t g_stage = 0;
volatile uint32_t g_result_3_5 = 0;
volatile uint32_t g_result_4_0 = 0;
volatile uint32_t g_fault_cfsr = 0;
volatile uint32_t g_fault_hfsr = 0;
volatile uint32_t g_fault_bfar = 0;

typedef struct {
  uintptr_t addr;
  uint32_t size;
} region_t;

/* STM32N657 internal SRAM is exposed as a contiguous 0x41A000-byte region
 * starting at 0x34000000 in the CubeIDE device database. Probe the first
 * 3.5 MiB, then extend to 4.0 MiB. */
static const region_t regions_3_5[] = {
    {0x34000000u, 0x00380000u}, /* 3.5 MiB */
};

/* Extra 512 KiB to reach 4.0 MiB total */
static const region_t regions_extra_4_0[] = {
    {0x34380000u, 0x00080000u}, /* +0.5 MiB = 4.0 MiB total */
};

static void bkpt_forever(void) {
  __asm volatile("bkpt #0");
  while (1) {
  }
}

void Default_Handler(void) {
  g_stage = 0xDEAD0000u;
  bkpt_forever();
}

void HardFault_Handler(void) {
  g_fault_cfsr = SCB_CFSR;
  g_fault_hfsr = SCB_HFSR;
  g_fault_bfar = SCB_BFAR;
  bkpt_forever();
}

__attribute__((section(".isr_vector")))
void (*const g_pfnVectors[])(void) = {
    (void (*)(void))(&_estack),
    (void (*)(void))Reset_Handler,
    Default_Handler,
    HardFault_Handler,
    Default_Handler,
    Default_Handler,
    Default_Handler,
    0,
    0,
    0,
    0,
    Default_Handler,
    Default_Handler,
    0,
    Default_Handler,
    Default_Handler,
};

static void runtime_init(void) {
  uint32_t *dst = &_sbss;
  while (dst < &_ebss) {
    *dst++ = 0;
  }
}

static void touch_region(uintptr_t addr, uint32_t size, uint32_t seed) {
  volatile uint32_t *p = (volatile uint32_t *)addr;
  uint32_t words = size / sizeof(uint32_t);
  uint32_t i;

  for (i = 0; i < words; ++i) {
    p[i] = seed ^ i;
  }
  for (i = 0; i < words; ++i) {
    if (p[i] != (seed ^ i)) {
      g_fault_bfar = (uint32_t)(addr + i * sizeof(uint32_t));
      g_fault_cfsr = 0xFFFFFFFFu;
      g_fault_hfsr = 0xFFFFFFFFu;
      bkpt_forever();
    }
  }
}

static void run_regions(const region_t *regions, uint32_t count, uint32_t stage_base) {
  uint32_t i;
  for (i = 0; i < count; ++i) {
    g_stage = stage_base + i;
    touch_region(regions[i].addr, regions[i].size, 0xA5A50000u ^ g_stage);
  }
}

void Reset_Handler(void) __attribute__((naked));
void Reset_Handler(void) {
  __asm volatile(
      "ldr sp, =_estack\n"
      "b reset_main\n");
}

void reset_main(void) {
  (*(volatile uint32_t *)0xE000ED08u) = (uint32_t)g_pfnVectors;
  runtime_init();

  g_stage = 1;
  run_regions(regions_3_5, sizeof(regions_3_5) / sizeof(regions_3_5[0]), 0x3500u);
  g_result_3_5 = OK_MAGIC;

  g_stage = 2;
  run_regions(regions_extra_4_0, sizeof(regions_extra_4_0) / sizeof(regions_extra_4_0[0]), 0x4000u);
  g_result_4_0 = OK_MAGIC;

  g_stage = 3;
  bkpt_forever();
}
