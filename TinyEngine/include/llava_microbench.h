#ifndef TINYENGINE_LLAVA_MICROBENCH_H_
#define TINYENGINE_LLAVA_MICROBENCH_H_

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "tinyengine_function_fp.h"

typedef struct {
  uint16_t seq_len;
  uint16_t hidden_size;
  uint16_t intermediate_size;
  uint16_t num_attention_heads;
  uint16_t num_key_value_heads;
  uint16_t rotary_dim;
  float rms_norm_eps;
  bool causal;
} qwen_block_fp_config;

typedef struct {
  qwen_block_fp_config config;
  const uint32_t* position_ids;
  const float* rope_cos;
  const float* rope_sin;
  const float* input_norm_weight;
  const float* post_attn_norm_weight;
  const float* q_proj_weight;
  const float* q_proj_bias;
  const float* k_proj_weight;
  const float* k_proj_bias;
  const float* v_proj_weight;
  const float* v_proj_bias;
  const float* o_proj_weight;
  const float* o_proj_bias;
  const float* gate_proj_weight;
  const float* gate_proj_bias;
  const float* up_proj_weight;
  const float* up_proj_bias;
  const float* down_proj_weight;
  const float* down_proj_bias;
} qwen_block_fp_params;

size_t qwen_block_fp_get_workspace_floats(const qwen_block_fp_config* config);
size_t qwen_block_fp_get_workspace_bytes(const qwen_block_fp_config* config);

tinyengine_status_fp qwen_block_fp_run(const qwen_block_fp_params* params,
                                       const float* input_data,
  float* output_data,
  float* workspace);

typedef struct {
  uint16_t num_image_tokens;
  uint16_t vision_hidden;
  uint16_t hidden_size;
} mm_projector_fp_config;

typedef struct {
  mm_projector_fp_config config;
  const float* fc1_weight;
  const float* fc1_bias;
  const float* fc2_weight;
  const float* fc2_bias;
} mm_projector_fp_params;

size_t mm_projector_fp_get_workspace_floats(const mm_projector_fp_config* config);
size_t mm_projector_fp_get_workspace_bytes(const mm_projector_fp_config* config);

tinyengine_status_fp mm_projector_fp_run(const mm_projector_fp_params* params,
                                         const float* input_data,
                                         float* output_data,
                                         float* workspace);

typedef struct {
  uint16_t hidden_size;
  uint32_t vocab_size;
} lm_head_last_token_fp_config;

typedef struct {
  lm_head_last_token_fp_config config;
  const float* weight;
  const float* bias;
} lm_head_last_token_fp_params;

size_t lm_head_last_token_fp_get_workspace_floats(const lm_head_last_token_fp_config* config);
size_t lm_head_last_token_fp_get_workspace_bytes(const lm_head_last_token_fp_config* config);

tinyengine_status_fp lm_head_last_token_fp_run(const lm_head_last_token_fp_params* params,
                                               const float* last_hidden_state,
                                               float* logits_output,
                                               float* workspace);

typedef struct {
  uint16_t num_layers;
  uint16_t max_seq_len;
  uint16_t num_kv_heads;
  uint16_t head_dim;
} kv_cache_fp_config;

size_t kv_cache_fp_get_cache_floats(const kv_cache_fp_config* config);
size_t kv_cache_fp_get_cache_bytes(const kv_cache_fp_config* config);

tinyengine_status_fp kv_cache_append_fp(const kv_cache_fp_config* config,
                                        float* key_cache,
                                        float* value_cache,
                                        uint16_t layer_idx,
                                        uint16_t seq_pos,
                                        const float* key_token,
                                        const float* value_token);

tinyengine_status_fp kv_cache_read_fp(const kv_cache_fp_config* config,
                                      const float* key_cache,
                                      const float* value_cache,
                                      uint16_t layer_idx,
                                      uint16_t seq_len,
                                      float* key_output,
                                      float* value_output);

typedef struct {
  uint16_t image_tokens;
  uint16_t text_tokens;
  uint16_t hidden_size;
} image_text_fusion_fp_config;

size_t image_text_fusion_fp_get_workspace_floats(const image_text_fusion_fp_config* config);
size_t image_text_fusion_fp_get_workspace_bytes(const image_text_fusion_fp_config* config);

tinyengine_status_fp image_text_fusion_fp_run(const image_text_fusion_fp_config* config,
                                              const float* image_embeds,
                                              const float* text_embeds,
                                              float* fused_output,
                                              float* workspace);

#endif
