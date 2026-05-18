#include "benchmark_common.h"

#include <math.h>
#include <stdio.h>

void bench_fill_sequence(float* dst, int n, float scale, float bias) {
  int i;

  if (dst == NULL || n <= 0) {
    return;
  }

  for (i = 0; i < n; ++i) {
    const int centered = (i % 17) - 8;
    dst[i] = (float)centered * scale + bias;
  }
}

void bench_fill_constant(float* dst, int n, float value) {
  int i;

  if (dst == NULL || n <= 0) {
    return;
  }

  for (i = 0; i < n; ++i) {
    dst[i] = value;
  }
}

void bench_zero(float* dst, int n) {
  bench_fill_constant(dst, n, 0.0f);
}

float bench_checksum_fp32(const float* data, int n) {
  int i;
  float acc = 0.0f;

  if (data == NULL || n <= 0) {
    return 0.0f;
  }

  for (i = 0; i < n; ++i) {
    acc += data[i];
  }

  return acc;
}

int bench_has_nan_or_inf_fp32(const float* data, int n) {
  int i;

  if (data == NULL || n <= 0) {
    return 0;
  }

  for (i = 0; i < n; ++i) {
    if (!isfinite(data[i])) {
      return 1;
    }
  }

  return 0;
}

void bench_fill_u32_sequence(uint32_t* dst, int n, uint32_t mod) {
  int i;

  if (dst == NULL || n <= 0 || mod == 0) {
    return;
  }

  for (i = 0; i < n; ++i) {
    dst[i] = (uint32_t)i % mod;
  }
}

void bench_fill_rope_tables(float* cos_table, float* sin_table, int max_pos, int rotary_dim) {
  int pos;
  int dim;

  if (cos_table == NULL || sin_table == NULL || max_pos <= 0 || rotary_dim <= 0) {
    return;
  }

  for (pos = 0; pos < max_pos; ++pos) {
    for (dim = 0; dim < rotary_dim; ++dim) {
      const float angle = 0.01f * (float)(pos + 1) * (float)(dim + 1);
      cos_table[pos * rotary_dim + dim] = cosf(angle);
      sin_table[pos * rotary_dim + dim] = sinf(angle);
    }
  }
}

void bench_print_csv_header(void) {
  printf("benchmark,variant,seq_len,hidden_size,intermediate_size,vision_hidden,num_image_tokens,text_tokens,");
  printf("num_heads,num_kv_heads,rotary_dim,head_dim,vocab_size,runs,workspace_bytes,cycles,latency_ms,");
  printf("checksum,has_nan_or_inf,status\n");
}

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
                         const char* status) {
  printf("%s,%s,%u,%u,%u,%u,%u,%u,%u,%u,%u,%u,%u,%d,%u,%u,%.6f,%.6f,%d,%s\n",
         benchmark,
         variant,
         seq_len,
         hidden_size,
         intermediate_size,
         vision_hidden,
         num_image_tokens,
         text_tokens,
         num_heads,
         num_kv_heads,
         rotary_dim,
         head_dim,
         vocab_size,
         runs,
         (unsigned int)workspace_bytes,
         cycles,
         latency_ms,
         checksum,
         has_nan_or_inf,
         status);
}
