/* ----------------------------------------------------------------------
 * Project: TinyEngine
 * Title:   lm_head_last_token_fp.c
 *
 * FP32 last-token lm_head benchmark wrapper.
 *
 * Target ISA:  ARMv7E-M
 * -------------------------------------------------------------------- */

#include "llava_microbench.h"

size_t lm_head_last_token_fp_get_workspace_floats(const lm_head_last_token_fp_config* config) {
  (void)config;
  return 0;
}

size_t lm_head_last_token_fp_get_workspace_bytes(const lm_head_last_token_fp_config* config) {
  return lm_head_last_token_fp_get_workspace_floats(config) * sizeof(float);
}

tinyengine_status_fp lm_head_last_token_fp_run(const lm_head_last_token_fp_params* params,
                                               const float* last_hidden_state,
                                               float* logits_output,
                                               float* workspace) {
  uint32_t vocab_idx;
  uint16_t hidden_idx;
  const float* weight;

  (void)workspace;

  if (params == NULL || last_hidden_state == NULL || logits_output == NULL || params->weight == NULL) {
    return PARAM_NO_SUPPORT_fp;
  }

  weight = params->weight;
  for (vocab_idx = 0; vocab_idx < params->config.vocab_size; ++vocab_idx) {
    float acc = params->bias != NULL ? params->bias[vocab_idx] : 0.0f;
    const float* weight_row = weight + ((size_t)vocab_idx * params->config.hidden_size);
    for (hidden_idx = 0; hidden_idx < params->config.hidden_size; ++hidden_idx) {
      acc += last_hidden_state[hidden_idx] * weight_row[hidden_idx];
    }
    logits_output[vocab_idx] = acc;
  }

  return STATE_SUCCESS_fp;
}
