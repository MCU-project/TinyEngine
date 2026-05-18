#include "benchmark_timer.h"

#include <stdint.h>
#include <time.h>

#if defined(__has_include)
#if __has_include("stm32n6xx_hal.h")
#include "stm32n6xx_hal.h"
#define BENCH_HAS_STM32_HAL 1
#endif
#endif

#if defined(BENCH_HAS_STM32_HAL) && defined(DWT) && defined(CoreDebug) && defined(CoreDebug_DEMCR_TRCENA_Msk) && \
    defined(DWT_CTRL_CYCCNTENA_Msk)
#define BENCH_HAS_DWT_TIMER 1
#endif

static uint8_t g_use_dwt_timer = 0;
static uint8_t g_use_hal_tick = 0;

void benchmark_timer_init(void) {
#if defined(BENCH_HAS_DWT_TIMER)
  CoreDebug->DEMCR |= CoreDebug_DEMCR_TRCENA_Msk;
  DWT->CYCCNT = 0;
  DWT->CTRL |= DWT_CTRL_CYCCNTENA_Msk;
  g_use_dwt_timer = 1;
  g_use_hal_tick = 0;
#elif defined(BENCH_HAS_STM32_HAL)
  g_use_dwt_timer = 0;
  g_use_hal_tick = 1;
#else
  g_use_dwt_timer = 0;
  g_use_hal_tick = 0;
#endif
}

uint32_t benchmark_timer_now(void) {
  if (g_use_dwt_timer != 0u) {
#if defined(BENCH_HAS_DWT_TIMER)
    return DWT->CYCCNT;
#endif
  }

  if (g_use_hal_tick != 0u) {
#if defined(BENCH_HAS_STM32_HAL)
    return HAL_GetTick();
#endif
  }

  return (uint32_t)clock();
}

uint32_t benchmark_timer_elapsed_cycles(uint32_t start, uint32_t end) {
  if (g_use_dwt_timer != 0u) {
    return end - start;
  }

  return 0u;
}

float benchmark_timer_elapsed_ms(uint32_t start, uint32_t end) {
  if (g_use_dwt_timer != 0u) {
#if defined(BENCH_HAS_DWT_TIMER) && defined(SystemCoreClock)
    if (SystemCoreClock != 0u) {
      return ((float)(end - start) * 1000.0f) / (float)SystemCoreClock;
    }
#endif
    return 0.0f;
  }

  if (g_use_hal_tick != 0u) {
    return (float)(end - start);
  }

  return ((float)(end - start) * 1000.0f) / (float)CLOCKS_PER_SEC;
}
