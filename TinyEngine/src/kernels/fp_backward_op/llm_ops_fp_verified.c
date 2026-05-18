/* ----------------------------------------------------------------------
 * Project: TinyEngine
 * Title:   llm_ops_fp_verified.c
 *
 * Verified FP32 operators for Qwen/LLaVA style decoder blocks.
 * These are added alongside the original baseline implementations.
 *
 * Target ISA:  ARMv7E-M
 * -------------------------------------------------------------------- */

#include "tinyengine_function_fp.h"

static void apply_rotary_to_head_qwen(float* data, const float* cos_row, const float* sin_row,
                                      const uint16_t head_dim, const uint16_t rotary_dim) {
  uint16_t i;
  const uint16_t limit = rotary_dim < head_dim ? rotary_dim : head_dim;
  const uint16_t half = limit / 2;

  for (i = 0; i < half; ++i) {
    const float x0 = data[i];
    const float x1 = data[i + half];
    const float cos0 = cos_row[i];
    const float sin0 = sin_row[i];
    data[i] = x0 * cos0 - x1 * sin0;
    data[i + half] = x1 * cos0 + x0 * sin0;
  }
}

tinyengine_status_fp gelu_exact_fp_verified(const size_t size, const float* input_data, float* output_data) {
  size_t i;
  const float inv_sqrt_two = 0.7071067811865475f;

  if (input_data == NULL || output_data == NULL) {
    return PARAM_NO_SUPPORT_fp;
  }

  for (i = 0; i < size; ++i) {
    const float x = input_data[i];
    output_data[i] = 0.5f * x * (1.0f + erff(x * inv_sqrt_two));
  }

  return STATE_SUCCESS_fp;
}

tinyengine_status_fp rotary_embedding_qwen_fp(float* query, float* key, const uint16_t seq_len,
                                              const uint16_t num_query_heads, const uint16_t num_key_heads,
                                              const uint16_t head_dim, const uint32_t* position_ids,
                                              const float* cos_table, const float* sin_table,
                                              const uint16_t rotary_dim) {
  uint16_t t;
  uint16_t h;

  if (query == NULL || key == NULL || position_ids == NULL || cos_table == NULL || sin_table == NULL) {
    return PARAM_NO_SUPPORT_fp;
  }
  if ((rotary_dim % 2) != 0 || rotary_dim > head_dim) {
    return PARAM_NO_SUPPORT_fp;
  }

  for (t = 0; t < seq_len; ++t) {
    const uint32_t pos = position_ids[t];
    const float* cos_row = cos_table + ((size_t)pos * rotary_dim);
    const float* sin_row = sin_table + ((size_t)pos * rotary_dim);

    for (h = 0; h < num_query_heads; ++h) {
      float* q_head = query + (((size_t)t * num_query_heads + h) * head_dim);
      apply_rotary_to_head_qwen(q_head, cos_row, sin_row, head_dim, rotary_dim);
    }

    for (h = 0; h < num_key_heads; ++h) {
      float* k_head = key + (((size_t)t * num_key_heads + h) * head_dim);
      apply_rotary_to_head_qwen(k_head, cos_row, sin_row, head_dim, rotary_dim);
    }
  }

  return STATE_SUCCESS_fp;
}
