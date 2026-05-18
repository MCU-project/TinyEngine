/* ----------------------------------------------------------------------
 * Project: TinyEngine
 * Title:   kv_cache_fp.c
 *
 * FP32 KV cache standalone helpers.
 *
 * Target ISA:  ARMv7E-M
 * -------------------------------------------------------------------- */

#include "llava_microbench.h"

static size_t kv_cache_layer_stride(const kv_cache_fp_config* config) {
  return (size_t)config->max_seq_len * config->num_kv_heads * config->head_dim;
}

size_t kv_cache_fp_get_cache_floats(const kv_cache_fp_config* config) {
  if (config == NULL) {
    return 0;
  }
  return (size_t)2 * config->num_layers * kv_cache_layer_stride(config);
}

size_t kv_cache_fp_get_cache_bytes(const kv_cache_fp_config* config) {
  return kv_cache_fp_get_cache_floats(config) * sizeof(float);
}

tinyengine_status_fp kv_cache_append_fp(const kv_cache_fp_config* config,
                                        float* key_cache,
                                        float* value_cache,
                                        uint16_t layer_idx,
                                        uint16_t seq_pos,
                                        const float* key_token,
                                        const float* value_token) {
  const size_t token_size = (size_t)config->num_kv_heads * config->head_dim;
  const size_t layer_stride = kv_cache_layer_stride(config);
  float* key_dst;
  float* value_dst;
  size_t i;

  if (config == NULL || key_cache == NULL || value_cache == NULL || key_token == NULL || value_token == NULL) {
    return PARAM_NO_SUPPORT_fp;
  }
  if (layer_idx >= config->num_layers || seq_pos >= config->max_seq_len) {
    return PARAM_NO_SUPPORT_fp;
  }

  key_dst = key_cache + ((size_t)layer_idx * layer_stride) + ((size_t)seq_pos * token_size);
  value_dst = value_cache + ((size_t)layer_idx * layer_stride) + ((size_t)seq_pos * token_size);

  for (i = 0; i < token_size; ++i) {
    key_dst[i] = key_token[i];
    value_dst[i] = value_token[i];
  }

  return STATE_SUCCESS_fp;
}

tinyengine_status_fp kv_cache_read_fp(const kv_cache_fp_config* config,
                                      const float* key_cache,
                                      const float* value_cache,
                                      uint16_t layer_idx,
                                      uint16_t seq_len,
                                      float* key_output,
                                      float* value_output) {
  const size_t token_size = (size_t)config->num_kv_heads * config->head_dim;
  const size_t layer_stride = kv_cache_layer_stride(config);
  const size_t read_size = (size_t)seq_len * token_size;
  const float* key_src;
  const float* value_src;
  size_t i;

  if (config == NULL || key_cache == NULL || value_cache == NULL || key_output == NULL || value_output == NULL) {
    return PARAM_NO_SUPPORT_fp;
  }
  if (layer_idx >= config->num_layers || seq_len > config->max_seq_len) {
    return PARAM_NO_SUPPORT_fp;
  }

  key_src = key_cache + ((size_t)layer_idx * layer_stride);
  value_src = value_cache + ((size_t)layer_idx * layer_stride);

  for (i = 0; i < read_size; ++i) {
    key_output[i] = key_src[i];
    value_output[i] = value_src[i];
  }

  return STATE_SUCCESS_fp;
}
