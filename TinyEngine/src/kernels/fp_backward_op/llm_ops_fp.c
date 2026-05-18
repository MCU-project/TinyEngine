/* ----------------------------------------------------------------------
 * Project: TinyEngine
 * Title:   llm_ops_fp.c
 *
 * Utility floating-point operators for LLM/VLM style decoder blocks.
 *
 * Target ISA:  ARMv7E-M
 * -------------------------------------------------------------------- */

#include "tinyengine_function_fp.h"

static void softmax_inplace_fp(float* data, const uint16_t length) {
  uint16_t i;
  float max_value = -FLT_MAX;
  float sum = 0.0f;

  for (i = 0; i < length; ++i) {
    if (data[i] > max_value) {
      max_value = data[i];
    }
  }

  for (i = 0; i < length; ++i) {
    data[i] = expf(data[i] - max_value);
    sum += data[i];
  }

  if (sum == 0.0f) {
    return;
  }

  for (i = 0; i < length; ++i) {
    data[i] /= sum;
  }
}

static void apply_rotary_to_head(float* data, const float* cos_row, const float* sin_row,
                                 const uint16_t head_dim, const uint16_t rotary_dim) {
  uint16_t i;
  const uint16_t limit = rotary_dim < head_dim ? rotary_dim : head_dim;

  for (i = 0; i + 1 < limit; i += 2) {
    const float x0 = data[i];
    const float x1 = data[i + 1];
    const float cos0 = cos_row[i];
    const float sin0 = sin_row[i];
    data[i] = x0 * cos0 - x1 * sin0;
    data[i + 1] = x1 * cos0 + x0 * sin0;
  }
}

tinyengine_status_fp argmax_fp(const uint16_t size, const float* input_data,
                               uint16_t* output_index, float* output_value) {
  uint16_t i;
  uint16_t best_index = 0;
  float best_value;

  if (size == 0 || input_data == NULL || output_index == NULL) {
    return PARAM_NO_SUPPORT_fp;
  }

  best_value = input_data[0];
  for (i = 1; i < size; ++i) {
    if (input_data[i] > best_value) {
      best_value = input_data[i];
      best_index = i;
    }
  }

  *output_index = best_index;
  if (output_value != NULL) {
    *output_value = best_value;
  }
  return STATE_SUCCESS_fp;
}

tinyengine_status_fp embedding_lookup_fp(const uint16_t seq_len, const uint16_t hidden_size,
                                         const uint32_t* token_ids, const float* embedding_table,
                                         float* output_data) {
  uint16_t t;
  uint16_t c;

  if (token_ids == NULL || embedding_table == NULL || output_data == NULL) {
    return PARAM_NO_SUPPORT_fp;
  }

  for (t = 0; t < seq_len; ++t) {
    const float* src = embedding_table + ((size_t)token_ids[t] * hidden_size);
    float* dst = output_data + ((size_t)t * hidden_size);
    for (c = 0; c < hidden_size; ++c) {
      dst[c] = src[c];
    }
  }

  return STATE_SUCCESS_fp;
}

tinyengine_status_fp gelu_fp(const uint16_t size, const float* input_data, float* output_data) {
  uint16_t i;
  const float sqrt_two_over_pi = 0.7978845608f;

  if (input_data == NULL || output_data == NULL) {
    return PARAM_NO_SUPPORT_fp;
  }

  for (i = 0; i < size; ++i) {
    const float x = input_data[i];
    const float x3 = x * x * x;
    const float inner = sqrt_two_over_pi * (x + 0.044715f * x3);
    output_data[i] = 0.5f * x * (1.0f + tanhf(inner));
  }

  return STATE_SUCCESS_fp;
}

tinyengine_status_fp repeat_kv_fp(const float* input_data, const uint16_t seq_len,
                                  const uint16_t num_kv_heads, const uint16_t head_dim,
                                  const uint16_t n_rep, float* output_data) {
  uint16_t t;
  uint16_t h;
  uint16_t r;
  uint16_t d;

  if (input_data == NULL || output_data == NULL || n_rep == 0) {
    return PARAM_NO_SUPPORT_fp;
  }

  for (t = 0; t < seq_len; ++t) {
    for (h = 0; h < num_kv_heads; ++h) {
      const float* src = input_data + (((size_t)t * num_kv_heads + h) * head_dim);
      for (r = 0; r < n_rep; ++r) {
        float* dst = output_data + (((size_t)t * (num_kv_heads * n_rep) + (h * n_rep + r)) * head_dim);
        for (d = 0; d < head_dim; ++d) {
          dst[d] = src[d];
        }
      }
    }
  }

  return STATE_SUCCESS_fp;
}

tinyengine_status_fp rmsnorm_fp(const float* input_data, const float* weight_data,
                                const uint16_t rows, const uint16_t cols, const float eps,
                                float* output_data) {
  uint16_t r;
  uint16_t c;

  if (input_data == NULL || weight_data == NULL || output_data == NULL) {
    return PARAM_NO_SUPPORT_fp;
  }

  for (r = 0; r < rows; ++r) {
    const float* src = input_data + ((size_t)r * cols);
    float* dst = output_data + ((size_t)r * cols);
    float mean_square = 0.0f;
    float inv_rms;

    for (c = 0; c < cols; ++c) {
      mean_square += src[c] * src[c];
    }
    mean_square /= (float)cols;
    inv_rms = 1.0f / sqrtf(mean_square + eps);

    for (c = 0; c < cols; ++c) {
      dst[c] = src[c] * inv_rms * weight_data[c];
    }
  }

  return STATE_SUCCESS_fp;
}

tinyengine_status_fp rotary_embedding_fp(float* query, float* key, const uint16_t seq_len,
                                         const uint16_t num_query_heads, const uint16_t num_key_heads,
                                         const uint16_t head_dim, const uint32_t* position_ids,
                                         const float* cos_table, const float* sin_table,
                                         const uint16_t rotary_dim) {
  uint16_t t;
  uint16_t h;

  if (query == NULL || key == NULL || position_ids == NULL || cos_table == NULL || sin_table == NULL) {
    return PARAM_NO_SUPPORT_fp;
  }

  for (t = 0; t < seq_len; ++t) {
    const uint32_t pos = position_ids[t];
    const float* cos_row = cos_table + ((size_t)pos * rotary_dim);
    const float* sin_row = sin_table + ((size_t)pos * rotary_dim);

    for (h = 0; h < num_query_heads; ++h) {
      float* q_head = query + (((size_t)t * num_query_heads + h) * head_dim);
      apply_rotary_to_head(q_head, cos_row, sin_row, head_dim, rotary_dim);
    }

    for (h = 0; h < num_key_heads; ++h) {
      float* k_head = key + (((size_t)t * num_key_heads + h) * head_dim);
      apply_rotary_to_head(k_head, cos_row, sin_row, head_dim, rotary_dim);
    }
  }

  return STATE_SUCCESS_fp;
}

tinyengine_status_fp silu_fp(const uint16_t size, const float* input_data, float* output_data) {
  uint16_t i;

  if (input_data == NULL || output_data == NULL) {
    return PARAM_NO_SUPPORT_fp;
  }

  for (i = 0; i < size; ++i) {
    output_data[i] = input_data[i] / (1.0f + expf(-input_data[i]));
  }

  return STATE_SUCCESS_fp;
}

tinyengine_status_fp qkv_attention_fp(const float* query, const float* key, const float* value,
                                      const uint16_t query_len, const uint16_t key_len,
                                      const uint16_t num_heads, const uint16_t head_dim,
                                      const bool causal, float* output_data, float* attn_buffer) {
  uint16_t h;
  uint16_t q_idx;
  uint16_t k_idx;
  uint16_t d;
  const float scale = 1.0f / sqrtf((float)head_dim);

  if (query == NULL || key == NULL || value == NULL || output_data == NULL || attn_buffer == NULL) {
    return PARAM_NO_SUPPORT_fp;
  }

  for (h = 0; h < num_heads; ++h) {
    for (q_idx = 0; q_idx < query_len; ++q_idx) {
      float* out = output_data + (((size_t)q_idx * num_heads + h) * head_dim);
      float* scores = attn_buffer + q_idx * key_len;
      const float* q_ptr = query + (((size_t)q_idx * num_heads + h) * head_dim);

      for (d = 0; d < head_dim; ++d) {
        out[d] = 0.0f;
      }

      for (k_idx = 0; k_idx < key_len; ++k_idx) {
        const float* k_ptr = key + (((size_t)k_idx * num_heads + h) * head_dim);
        float score = 0.0f;

        if (causal && k_idx > q_idx) {
          scores[k_idx] = -FLT_MAX;
          continue;
        }

        for (d = 0; d < head_dim; ++d) {
          score += q_ptr[d] * k_ptr[d];
        }
        scores[k_idx] = score * scale;
      }

      softmax_inplace_fp(scores, key_len);

      for (k_idx = 0; k_idx < key_len; ++k_idx) {
        const float* v_ptr = value + (((size_t)k_idx * num_heads + h) * head_dim);
        const float weight = scores[k_idx];
        for (d = 0; d < head_dim; ++d) {
          out[d] += weight * v_ptr[d];
        }
      }
    }
  }

  return STATE_SUCCESS_fp;
}
