#ifndef LLAVA_MICROBENCH_BENCHMARK_COMMON_H_
#define LLAVA_MICROBENCH_BENCHMARK_COMMON_H_

#include <stddef.h>
#include <stdint.h>

void bench_fill_sequence(float* dst, int n, float scale, float bias);
void bench_fill_constant(float* dst, int n, float value);
void bench_zero(float* dst, int n);
float bench_checksum_fp32(const float* data, int n);
int bench_has_nan_or_inf_fp32(const float* data, int n);
void bench_fill_u32_sequence(uint32_t* dst, int n, uint32_t mod);
void bench_fill_rope_tables(float* cos_table, float* sin_table, int max_pos, int rotary_dim);
void bench_print_csv_header(void);
void bench_print_csv_row(const char* benchmark,
                         const char* variant,
                         uint16_t seq_len,
                         uint16_t hidden_size,
                         uint16_t intermediate_size,
                         uint16_t vision_hidden,
                         uint16_t num_image_tokens,
                         uint16_t text_tokens,
                         uint16_t num_heads,
                         uint16_t num_kv_heads,
                         uint16_t rotary_dim,
                         uint16_t head_dim,
                         uint32_t vocab_size,
                         int runs,
                         size_t workspace_bytes,
                         uint32_t cycles,
                         float latency_ms,
                         float checksum,
                         int has_nan_or_inf,
                         const char* status);

#endif
