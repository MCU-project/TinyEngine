/* ----------------------------------------------------------------------
 * Project: TinyEngine
 * Title:   qwen_block_fp.c
 *
 * FP32 Qwen decoder block microbenchmark wrapper.
 * This implementation is intended for latency and workspace measurements.
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
} qwen_block_workspace_layout;

static tinyengine_status_fp linear_fp32(const float* input_data,
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

static qwen_block_workspace_layout get_workspace_layout(const qwen_block_fp_config* config) {
  qwen_block_workspace_layout layout;
  const size_t token_hidden = (size_t)config->seq_len * config->hidden_size;
  const size_t head_dim = config->hidden_size / config->num_attention_heads;
  const size_t kv_hidden = (size_t)config->num_key_value_heads * head_dim;
  const size_t token_kv = (size_t)config->seq_len * kv_hidden;
  const size_t token_intermediate = (size_t)config->seq_len * config->intermediate_size;
  const size_t score_floats = (size_t)config->seq_len * config->seq_len;

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

size_t qwen_block_fp_get_workspace_floats(const qwen_block_fp_config* config) {
  if (config == NULL) {
    return 0;
  }
  return get_workspace_layout(config).total_floats;
}

size_t qwen_block_fp_get_workspace_bytes(const qwen_block_fp_config* config) {
  return qwen_block_fp_get_workspace_floats(config) * sizeof(float);
}

tinyengine_status_fp qwen_block_fp_run(const qwen_block_fp_params* params,
                                       const float* input_data,
                                       float* output_data,
                                       float* workspace) {
  const qwen_block_fp_config* config;
  qwen_block_workspace_layout layout;
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
  if (config->num_attention_heads == 0 || config->num_key_value_heads == 0) {
    return PARAM_NO_SUPPORT_fp;
  }
  if ((config->hidden_size % config->num_attention_heads) != 0) {
    return PARAM_NO_SUPPORT_fp;
  }
  if ((config->num_attention_heads % config->num_key_value_heads) != 0) {
    return PARAM_NO_SUPPORT_fp;
  }

  head_dim = config->hidden_size / config->num_attention_heads;
  kv_hidden = config->num_key_value_heads * head_dim;
  kv_repeat = config->num_attention_heads / config->num_key_value_heads;
  hidden_floats = (size_t)config->seq_len * config->hidden_size;
  intermediate_floats = (size_t)config->seq_len * config->intermediate_size;

  layout = get_workspace_layout(config);
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

  status = linear_fp32(norm1, config->seq_len, config->hidden_size, config->hidden_size,
                       params->q_proj_weight, params->q_proj_bias, query);
  if (status != STATE_SUCCESS_fp) {
    return status;
  }

  status = linear_fp32(norm1, config->seq_len, config->hidden_size, kv_hidden,
                       params->k_proj_weight, params->k_proj_bias, key);
  if (status != STATE_SUCCESS_fp) {
    return status;
  }

  status = linear_fp32(norm1, config->seq_len, config->hidden_size, kv_hidden,
                       params->v_proj_weight, params->v_proj_bias, value);
  if (status != STATE_SUCCESS_fp) {
    return status;
  }

  status = rotary_embedding_fp(query, key, config->seq_len, config->num_attention_heads,
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

  status = qkv_attention_fp(query, key_rep, value_rep, config->seq_len, config->seq_len,
                            config->num_attention_heads, head_dim, config->causal, attn_out, scores);
  if (status != STATE_SUCCESS_fp) {
    return status;
  }

  status = linear_fp32(attn_out, config->seq_len, config->hidden_size, config->hidden_size,
                       params->o_proj_weight, params->o_proj_bias, output_data);
  if (status != STATE_SUCCESS_fp) {
    return status;
  }

  status = add_fp((uint16_t)hidden_floats, input_data, output_data, output_data);
  if (status != STATE_SUCCESS_fp) {
    return status;
  }

  status = rmsnorm_fp(output_data, params->post_attn_norm_weight, config->seq_len, config->hidden_size,
                      config->rms_norm_eps, norm2);
  if (status != STATE_SUCCESS_fp) {
    return status;
  }

  status = linear_fp32(norm2, config->seq_len, config->hidden_size, config->intermediate_size,
                       params->gate_proj_weight, params->gate_proj_bias, gate);
  if (status != STATE_SUCCESS_fp) {
    return status;
  }

  status = linear_fp32(norm2, config->seq_len, config->hidden_size, config->intermediate_size,
                       params->up_proj_weight, params->up_proj_bias, up);
  if (status != STATE_SUCCESS_fp) {
    return status;
  }

  status = silu_fp((uint16_t)intermediate_floats, gate, gate);
  if (status != STATE_SUCCESS_fp) {
    return status;
  }

  status = mul((uint16_t)intermediate_floats, gate, up, gate);
  if (status != STATE_SUCCESS_fp) {
    return status;
  }

  status = linear_fp32(gate, config->seq_len, config->intermediate_size, config->hidden_size,
                       params->down_proj_weight, params->down_proj_bias, norm1);
  if (status != STATE_SUCCESS_fp) {
    return status;
  }

  status = add_fp((uint16_t)hidden_floats, output_data, norm1, output_data);
  if (status != STATE_SUCCESS_fp) {
    return status;
  }

  return STATE_SUCCESS_fp;
}
