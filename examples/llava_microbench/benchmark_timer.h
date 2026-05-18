#ifndef LLAVA_MICROBENCH_BENCHMARK_TIMER_H_
#define LLAVA_MICROBENCH_BENCHMARK_TIMER_H_

#include <stdint.h>

void benchmark_timer_init(void);
uint32_t benchmark_timer_now(void);
uint32_t benchmark_timer_elapsed_cycles(uint32_t start, uint32_t end);
float benchmark_timer_elapsed_ms(uint32_t start, uint32_t end);

#endif
