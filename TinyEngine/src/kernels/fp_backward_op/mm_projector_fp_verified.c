/* ----------------------------------------------------------------------
 * Project: TinyEngine
 * Title:   mm_projector_fp_verified.c
 *
 * Verified FP32 mm_projector wrapper.
 * Linear -> exact GELU -> Linear
 *
 * Target ISA:  ARMv7E-M
 * -------------------------------------------------------------------- */

#include "llava_microbench.h"

static tinyengine_status_fp linear_fp32_mm_verified(const float* input_data,
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

static tinyengine_status_fp gelu_tanh_fp_verified_n(const size_t size,
                                                    const float* input_data,
                                                    float* output_data) {
  size_t i;
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

size_t mm_projector_fp_verified_get_workspace_floats(const mm_projector_fp_config* config) {
  if (config == NULL) {
    return 0;
  }
  return (size_t)config->num_image_tokens * config->hidden_size;
}

size_t mm_projector_fp_verified_get_workspace_bytes(const mm_projector_fp_config* config) {
  return mm_projector_fp_verified_get_workspace_floats(config) * sizeof(float);
}

tinyengine_status_fp mm_projector_fp_verified_run(const mm_projector_fp_params* params,
                                                  const float* input_data,
                                                  float* output_data,
                                                  float* workspace) {
  return mm_projector_fp_verified_run_checked(
      params,
      input_data,
      output_data,
      workspace,
      params != NULL ? mm_projector_fp_verified_get_workspace_floats(&params->config) : 0,
      true);
}

tinyengine_status_fp mm_projector_fp_verified_run_checked(const mm_projector_fp_params* params,
                                                          const float* input_data,
                                                          float* output_data,
                                                          float* workspace,
                                                          size_t workspace_floats,
                                                          bool use_exact_gelu) {
  const mm_projector_fp_config* config;
  const size_t activation_floats =
      (size_t)params->config.num_image_tokens * params->config.hidden_size;
  tinyengine_status_fp status;

  if (params == NULL || input_data == NULL || output_data == NULL || workspace == NULL) {
    return PARAM_NO_SUPPORT_fp;
  }

  config = &params->config;
  if (workspace_floats < activation_floats) {
    return PARAM_NO_SUPPORT_fp;
  }

  status = linear_fp32_mm_verified(input_data, config->num_image_tokens, config->vision_hidden,
                                   config->hidden_size, params->fc1_weight, params->fc1_bias, workspace);
  if (status != STATE_SUCCESS_fp) {
    return status;
  }

  if (use_exact_gelu) {
    status = gelu_exact_fp_verified(activation_floats, workspace, workspace);
  } else {
    status = gelu_tanh_fp_verified_n(activation_floats, workspace, workspace);
  }
  if (status != STATE_SUCCESS_fp) {
    return status;
  }

  status = linear_fp32_mm_verified(workspace, config->num_image_tokens, config->hidden_size,
                                   config->hidden_size, params->fc2_weight, params->fc2_bias, output_data);
  if (status != STATE_SUCCESS_fp) {
    return status;
  }

  return STATE_SUCCESS_fp;
}
