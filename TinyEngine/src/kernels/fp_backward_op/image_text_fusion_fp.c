/* ----------------------------------------------------------------------
 * Project: TinyEngine
 * Title:   image_text_fusion_fp.c
 *
 * FP32 image/text embed fusion helper.
 *
 * Target ISA:  ARMv7E-M
 * -------------------------------------------------------------------- */

#include "llava_microbench.h"

size_t image_text_fusion_fp_get_workspace_floats(const image_text_fusion_fp_config* config) {
  (void)config;
  return 0;
}

size_t image_text_fusion_fp_get_workspace_bytes(const image_text_fusion_fp_config* config) {
  return image_text_fusion_fp_get_workspace_floats(config) * sizeof(float);
}

tinyengine_status_fp image_text_fusion_fp_run(const image_text_fusion_fp_config* config,
                                              const float* image_embeds,
                                              const float* text_embeds,
                                              float* fused_output,
                                              float* workspace) {
  const size_t image_floats = (size_t)config->image_tokens * config->hidden_size;
  const size_t text_floats = (size_t)config->text_tokens * config->hidden_size;
  size_t i;

  (void)workspace;

  if (config == NULL || image_embeds == NULL || text_embeds == NULL || fused_output == NULL) {
    return PARAM_NO_SUPPORT_fp;
  }

  for (i = 0; i < image_floats; ++i) {
    fused_output[i] = image_embeds[i];
  }
  for (i = 0; i < text_floats; ++i) {
    fused_output[image_floats + i] = text_embeds[i];
  }

  return STATE_SUCCESS_fp;
}
