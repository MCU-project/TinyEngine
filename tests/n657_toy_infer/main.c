#include <stdint.h>

#define SCB_CFSR (*(volatile uint32_t *)0xE000ED28u)
#define SCB_HFSR (*(volatile uint32_t *)0xE000ED2Cu)
#define SCB_BFAR (*(volatile uint32_t *)0xE000ED38u)

#define OK_MAGIC 0x600D0001u
#define FAIL_MAGIC 0xBAD00001u

#define INPUT_W 96u
#define INPUT_H 96u
#define INPUT_C 3u
#define INPUT_BYTES (INPUT_W * INPUT_H * INPUT_C)
#define OUTPUT_BYTES 1024u

#define MODEL_ARENA_BASE 0x34100000u
#define MODEL_ARENA_SIZE 0x00300000u /* 3 MiB */

extern uint32_t _estack;
extern uint32_t _sbss;
extern uint32_t _ebss;

void Reset_Handler(void);
void reset_main(void);

volatile uint32_t g_stage = 0;
volatile uint32_t g_status = 0;
volatile uint32_t g_fault_cfsr = 0;
volatile uint32_t g_fault_hfsr = 0;
volatile uint32_t g_fault_bfar = 0;
volatile uint32_t g_checksum = 0;
volatile uint32_t g_bytes_touched = 0;

static uint8_t g_input[INPUT_BYTES];
static volatile uint8_t g_output[OUTPUT_BYTES];

static void bkpt_forever(void) {
  __asm volatile("bkpt #0");
  while (1) {
  }
}

void Default_Handler(void) {
  g_status = FAIL_MAGIC;
  bkpt_forever();
}

void HardFault_Handler(void) {
  g_fault_cfsr = SCB_CFSR;
  g_fault_hfsr = SCB_HFSR;
  g_fault_bfar = SCB_BFAR;
  g_status = FAIL_MAGIC;
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

static void fill_input(void) {
  uint32_t i;
  for (i = 0; i < INPUT_BYTES; ++i) {
    g_input[i] = (uint8_t)((i * 37u + 11u) & 0xFFu);
  }
}

static uint32_t seed_from_input(void) {
  uint32_t i;
  uint32_t acc = 0x12345678u;
  for (i = 0; i < INPUT_BYTES; ++i) {
    acc ^= ((uint32_t)g_input[i] << (i & 7u));
    acc = (acc << 3) | (acc >> 29);
    acc += 0x9E3779B9u;
  }
  return acc;
}

static void run_toy_inference(uint32_t seed) {
  volatile uint32_t *arena = (volatile uint32_t *)MODEL_ARENA_BASE;
  uint32_t words = MODEL_ARENA_SIZE / sizeof(uint32_t);
  uint32_t i;
  uint32_t acc = seed;

  for (i = 0; i < words; ++i) {
    acc ^= (0xA5A50000u + i);
    acc = (acc << 5) | (acc >> 27);
    arena[i] = acc;
  }

  g_stage = 3;
  for (i = 0; i < words; i += 97u) {
    acc ^= arena[i];
  }

  g_stage = 4;
  for (i = 0; i < OUTPUT_BYTES; ++i) {
    g_output[i] = (uint8_t)(acc >> ((i & 3u) * 8u));
    acc ^= g_output[i] + i;
  }

  g_checksum = acc;
  g_bytes_touched = MODEL_ARENA_SIZE + INPUT_BYTES + OUTPUT_BYTES;
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
  fill_input();

  g_stage = 2;
  run_toy_inference(seed_from_input());

  g_stage = 5;
  g_status = OK_MAGIC;
  bkpt_forever();
}
