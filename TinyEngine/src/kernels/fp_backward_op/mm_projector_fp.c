/* ----------------------------------------------------------------------
 * Project: TinyEngine
 * Title:   mm_projector_fp.c
 *
 * FP32 mm_projector microbenchmark wrapper.
 * Linear -> GELU -> Linear
 *
 * Target ISA:  ARMv7E-M
 * -------------------------------------------------------------------- */

#include "llava_microbench.h"

static tinyengine_status_fp linear_fp32_mm(const float* input_data,
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

size_t mm_projector_fp_get_workspace_floats(const mm_projector_fp_config* config) {
  if (config == NULL) {
    return 0;
  }
  return (size_t)config->num_image_tokens * config->hidden_size;
}

size_t mm_projector_fp_get_workspace_bytes(const mm_projector_fp_config* config) {
  return mm_projector_fp_get_workspace_floats(config) * sizeof(float);
}

tinyengine_status_fp mm_projector_fp_run(const mm_projector_fp_params* params,
                                         const float* input_data,
                                         float* output_data,
                                         float* workspace) {
  const mm_projector_fp_config* config;
  tinyengine_status_fp status;

  if (params == NULL || input_data == NULL || output_data == NULL || workspace == NULL) {
    return PARAM_NO_SUPPORT_fp;
  }

  config = &params->config;

  status = linear_fp32_mm(input_data, config->num_image_tokens, config->vision_hidden,
                          config->hidden_size, params->fc1_weight, params->fc1_bias, workspace);
  if (status != STATE_SUCCESS_fp) {
    return status;
  }

  status = gelu_fp((uint16_t)((size_t)config->num_image_tokens * config->hidden_size), workspace, workspace);
  if (status != STATE_SUCCESS_fp) {
    return status;
  }

  status = linear_fp32_mm(workspace, config->num_image_tokens, config->hidden_size,
                          config->hidden_size, params->fc2_weight, params->fc2_bias, output_data);
  if (status != STATE_SUCCESS_fp) {
    return status;
  }

  return STATE_SUCCESS_fp;
}
