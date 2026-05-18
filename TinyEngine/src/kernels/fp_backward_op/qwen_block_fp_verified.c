/* ----------------------------------------------------------------------
 * Project: TinyEngine
 * Title:   qwen_block_fp_verified.c
 *
 * Verified FP32 Qwen decoder block wrapper.
 * Uses Qwen-style RoPE and size_t-safe elementwise loops.
 *
 * Target ISA:  ARMv7E-M
 * -------------------------------------------------------------------- */

#include "llava_microbench.h"

typedef struct {
  size_t norm1_offset;
  size_t q_offset;
  size_t k_offset;
  size_t v_offset;
  size_t k_rep_offset;
  size_t v_rep_offset;
  size_t attn_offset;
  size_t norm2_offset;
  size_t gate_offset;
  size_t up_offset;
  size_t scores_offset;
  size_t total_floats;
} qwen_block_verified_workspace_layout;

static void softmax_inplace_verified(float* data, const uint16_t length) {
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

static tinyengine_status_fp linear_fp32_verified(const float* input_data,
                                                 const uint16_t rows,
                                                 const uint16_t input_dim,
                                                 const uint16_t output_dim,
                                                 const float* weight_data,
                                                 const float* bias_data,
                                                 float* output_data) {
  uint16_t row;
  uint16_t out_ch;
  uint16_t in_ch;

  if (input_data == NULL || weight_data == NULL || output_data == NULL) {
    return PARAM_NO_SUPPORT_fp;
  }

  for (row = 0; row < rows; ++row) {
    const float* input_row = input_data + ((size_t)row * input_dim);
    float* output_row = output_data + ((size_t)row * output_dim);
    for (out_ch = 0; out_ch < output_dim; ++out_ch) {
      const float* weight_row = weight_data + ((size_t)out_ch * input_dim);
      float acc = (bias_data != NULL) ? bias_data[out_ch] : 0.0f;
      for (in_ch = 0; in_ch < input_dim; ++in_ch) {
        acc += weight_row[in_ch] * input_row[in_ch];
      }
      output_row[out_ch] = acc;
    }
  }

  return STATE_SUCCESS_fp;
}

static tinyengine_status_fp add_fp_verified_n(const size_t size,
                                              const float* input1_data,
                                              const float* input2_data,
                                              float* output_data) {
  size_t i;

  if (input1_data == NULL || input2_data == NULL || output_data == NULL) {
    return PARAM_NO_SUPPORT_fp;
  }

  for (i = 0; i < size; ++i) {
    output_data[i] = input1_data[i] + input2_data[i];
  }

  return STATE_SUCCESS_fp;
}

static tinyengine_status_fp mul_fp_verified_n(const size_t size,
                                              const float* input1_data,
                                              const float* input2_data,
                                              float* output_data) {
  size_t i;

  if (input1_data == NULL || input2_data == NULL || output_data == NULL) {
    return PARAM_NO_SUPPORT_fp;
  }

  for (i = 0; i < size; ++i) {
    output_data[i] = input1_data[i] * input2_data[i];
  }

  return STATE_SUCCESS_fp;
}

static tinyengine_status_fp silu_fp_verified_n(const size_t size,
                                               const float* input_data,
                                               float* output_data) {
  size_t i;

  if (input_data == NULL || output_data == NULL) {
    return PARAM_NO_SUPPORT_fp;
  }

  for (i = 0; i < size; ++i) {
    output_data[i] = input_data[i] / (1.0f + expf(-input_data[i]));
  }

  return STATE_SUCCESS_fp;
}

static tinyengine_status_fp qkv_attention_fp_streaming(const float* query,
                                                       const float* key,
                                                       const float* value,
                                                       const uint16_t query_len,
                                                       const uint16_t key_len,
                                                       const uint16_t num_heads,
                                                       const uint16_t head_dim,
                                                       const bool causal,
                                                       float* output_data,
                                                       float* score_row) {
  uint16_t h;
  uint16_t q_idx;
  uint16_t k_idx;
  uint16_t d;
  const float scale = 1.0f / sqrtf((float)head_dim);

  if (query == NULL || key == NULL || value == NULL || output_data == NULL || score_row == NULL) {
    return PARAM_NO_SUPPORT_fp;
  }

  for (h = 0; h < num_heads; ++h) {
    for (q_idx = 0; q_idx < query_len; ++q_idx) {
      float* out = output_data + (((size_t)q_idx * num_heads + h) * head_dim);
      const float* q_ptr = query + (((size_t)q_idx * num_heads + h) * head_dim);

      for (d = 0; d < head_dim; ++d) {
        out[d] = 0.0f;
      }

      for (k_idx = 0; k_idx < key_len; ++k_idx) {
        const float* k_ptr = key + (((size_t)k_idx * num_heads + h) * head_dim);
        float score = 0.0f;

        if (causal && k_idx > q_idx) {
          score_row[k_idx] = -FLT_MAX;
          continue;
        }

        for (d = 0; d < head_dim; ++d) {
          score += q_ptr[d] * k_ptr[d];
        }
        score_row[k_idx] = score * scale;
      }

      softmax_inplace_verified(score_row, key_len);

      for (k_idx = 0; k_idx < key_len; ++k_idx) {
        const float* v_ptr = value + (((size_t)k_idx * num_heads + h) * head_dim);
        const float weight = score_row[k_idx];
        for (d = 0; d < head_dim; ++d) {
          out[d] += weight * v_ptr[d];
        }
      }
    }
  }

  return STATE_SUCCESS_fp;
}

static qwen_block_verified_workspace_layout get_workspace_layout_verified(const qwen_block_fp_config* config) {
  qwen_block_verified_workspace_layout layout;
  const size_t token_hidden = (size_t)config->seq_len * config->hidden_size;
  const size_t head_dim = config->hidden_size / config->num_attention_heads;
  const size_t kv_hidden = (size_t)config->num_key_value_heads * head_dim;
  const size_t token_kv = (size_t)config->seq_len * kv_hidden;
  const size_t token_intermediate = (size_t)config->seq_len * config->intermediate_size;
  const size_t score_floats = config->seq_len;

  layout.norm1_offset = 0;
  layout.q_offset = layout.norm1_offset + token_hidden;
  layout.k_offset = layout.q_offset + token_hidden;
  layout.v_offset = layout.k_offset + token_kv;
  layout.k_rep_offset = layout.v_offset + token_kv;
  layout.v_rep_offset = layout.k_rep_offset + token_hidden;
  layout.attn_offset = layout.v_rep_offset + token_hidden;
  layout.norm2_offset = layout.attn_offset + token_hidden;
  layout.gate_offset = layout.norm2_offset + token_hidden;
  layout.up_offset = layout.gate_offset + token_intermediate;
  layout.scores_offset = layout.up_offset + token_intermediate;
  layout.total_floats = layout.scores_offset + score_floats;
  return layout;
}

size_t qwen_block_fp_verified_get_workspace_floats(const qwen_block_fp_config* config) {
  if (config == NULL) {
    return 0;
  }
  return get_workspace_layout_verified(config).total_floats;
}

size_t qwen_block_fp_verified_get_workspace_bytes(const qwen_block_fp_config* config) {
  return qwen_block_fp_verified_get_workspace_floats(config) * sizeof(float);
}

tinyengine_status_fp qwen_block_fp_verified_run(const qwen_block_fp_params* params,
                                                const float* input_data,
                                                float* output_data,
                                                float* workspace) {
  return qwen_block_fp_verified_run_checked(
      params,
      input_data,
      output_data,
      workspace,
      params != NULL ? qwen_block_fp_verified_get_workspace_floats(&params->config) : 0,
      params != NULL ? params->config.seq_len : 0);
}

tinyengine_status_fp qwen_block_fp_verified_run_checked(const qwen_block_fp_params* params,
                                                        const float* input_data,
                                                        float* output_data,
                                                        float* workspace,
                                                        size_t workspace_floats,
                                                        uint32_t rope_table_rows) {
  const qwen_block_fp_config* config;
  qwen_block_verified_workspace_layout layout;
  uint16_t head_dim;
  uint16_t kv_hidden;
  uint16_t kv_repeat;
  size_t hidden_floats;
  size_t intermediate_floats;
  float* norm1;
  float* query;
  float* key;
  float* value;
  float* key_rep;
  float* value_rep;
  float* attn_out;
  float* norm2;
  float* gate;
  float* up;
  float* scores;
  tinyengine_status_fp status;

  if (params == NULL || input_data == NULL || output_data == NULL || workspace == NULL) {
    return PARAM_NO_SUPPORT_fp;
  }

  config = &params->config;
  if (config->seq_len == 0 || config->hidden_size == 0 || config->intermediate_size == 0) {
    return PARAM_NO_SUPPORT_fp;
  }
  if (config->num_attention_heads == 0 || config->num_key_value_heads == 0) {
    return PARAM_NO_SUPPORT_fp;
  }
  if ((config->hidden_size % config->num_attention_heads) != 0) {
    return PARAM_NO_SUPPORT_fp;
  }
  if ((config->num_attention_heads % config->num_key_value_heads) != 0) {
    return PARAM_NO_SUPPORT_fp;
  }
  if (config->rotary_dim > 0 && rope_table_rows < config->seq_len) {
    return PARAM_NO_SUPPORT_fp;
  }

  head_dim = config->hidden_size / config->num_attention_heads;
  kv_hidden = config->num_key_value_heads * head_dim;
  kv_repeat = config->num_attention_heads / config->num_key_value_heads;
  hidden_floats = (size_t)config->seq_len * config->hidden_size;
  intermediate_floats = (size_t)config->seq_len * config->intermediate_size;
  if (config->rotary_dim > 0) {
    uint16_t t;
    for (t = 0; t < config->seq_len; ++t) {
      if (params->position_ids[t] >= rope_table_rows) {
        return PARAM_NO_SUPPORT_fp;
      }
    }
  }

  layout = get_workspace_layout_verified(config);
  if (workspace_floats < layout.total_floats) {
    return PARAM_NO_SUPPORT_fp;
  }
  norm1 = workspace + layout.norm1_offset;
  query = workspace + layout.q_offset;
  key = workspace + layout.k_offset;
  value = workspace + layout.v_offset;
  key_rep = workspace + layout.k_rep_offset;
  value_rep = workspace + layout.v_rep_offset;
  attn_out = workspace + layout.attn_offset;
  norm2 = workspace + layout.norm2_offset;
  gate = workspace + layout.gate_offset;
  up = workspace + layout.up_offset;
  scores = workspace + layout.scores_offset;

  status = rmsnorm_fp(input_data, params->input_norm_weight, config->seq_len, config->hidden_size,
                      config->rms_norm_eps, norm1);
  if (status != STATE_SUCCESS_fp) {
    return status;
  }

  status = linear_fp32_verified(norm1, config->seq_len, config->hidden_size, config->hidden_size,
                                params->q_proj_weight, params->q_proj_bias, query);
  if (status != STATE_SUCCESS_fp) {
    return status;
  }

  status = linear_fp32_verified(norm1, config->seq_len, config->hidden_size, kv_hidden,
                                params->k_proj_weight, params->k_proj_bias, key);
  if (status != STATE_SUCCESS_fp) {
    return status;
  }

  status = linear_fp32_verified(norm1, config->seq_len, config->hidden_size, kv_hidden,
                                params->v_proj_weight, params->v_proj_bias, value);
  if (status != STATE_SUCCESS_fp) {
    return status;
  }

  status = rotary_embedding_qwen_fp(query, key, config->seq_len, config->num_attention_heads,
                                    config->num_key_value_heads, head_dim, params->position_ids,
                                    params->rope_cos, params->rope_sin, config->rotary_dim);
  if (status != STATE_SUCCESS_fp) {
    return status;
  }

  status = repeat_kv_fp(key, config->seq_len, config->num_key_value_heads, head_dim, kv_repeat, key_rep);
  if (status != STATE_SUCCESS_fp) {
    return status;
  }

  status = repeat_kv_fp(value, config->seq_len, config->num_key_value_heads, head_dim, kv_repeat, value_rep);
  if (status != STATE_SUCCESS_fp) {
    return status;
  }

  status = qkv_attention_fp_streaming(query, key_rep, value_rep, config->seq_len, config->seq_len,
                                      config->num_attention_heads, head_dim, config->causal, attn_out, scores);
  if (status != STATE_SUCCESS_fp) {
    return status;
  }

  status = linear_fp32_verified(attn_out, config->seq_len, config->hidden_size, config->hidden_size,
                                params->o_proj_weight, params->o_proj_bias, output_data);
  if (status != STATE_SUCCESS_fp) {
    return status;
  }

  status = add_fp_verified_n(hidden_floats, input_data, output_data, output_data);
  if (status != STATE_SUCCESS_fp) {
    return status;
  }

  status = rmsnorm_fp(output_data, params->post_attn_norm_weight, config->seq_len, config->hidden_size,
                      config->rms_norm_eps, norm2);
  if (status != STATE_SUCCESS_fp) {
    return status;
  }

  status = linear_fp32_verified(norm2, config->seq_len, config->hidden_size, config->intermediate_size,
                                params->gate_proj_weight, params->gate_proj_bias, gate);
  if (status != STATE_SUCCESS_fp) {
    return status;
  }

  status = linear_fp32_verified(norm2, config->seq_len, config->hidden_size, config->intermediate_size,
                                params->up_proj_weight, params->up_proj_bias, up);
  if (status != STATE_SUCCESS_fp) {
    return status;
  }

  status = silu_fp_verified_n(intermediate_floats, gate, gate);
  if (status != STATE_SUCCESS_fp) {
    return status;
  }

  status = mul_fp_verified_n(intermediate_floats, gate, up, gate);
  if (status != STATE_SUCCESS_fp) {
    return status;
  }

  status = linear_fp32_verified(gate, config->seq_len, config->intermediate_size, config->hidden_size,
                                params->down_proj_weight, params->down_proj_bias, norm1);
  if (status != STATE_SUCCESS_fp) {
    return status;
  }

  status = add_fp_verified_n(hidden_floats, output_data, norm1, output_data);
  if (status != STATE_SUCCESS_fp) {
    return status;
  }

  return STATE_SUCCESS_fp;
}
