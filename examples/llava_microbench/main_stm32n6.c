#include <stddef.h>
#include <stdint.h>

#include "benchmark_common.h"
#include "benchmark_timer.h"
#include "llava_microbench.h"

#define STM32_TOY_RUNS 10
#define STM32_TOY_SEQ_LEN 4
#define STM32_TOY_HIDDEN_SIZE 32
#define STM32_TOY_INTERMEDIATE_SIZE 64
#define STM32_TOY_NUM_HEADS 4
#define STM32_TOY_NUM_KV_HEADS 2
#define STM32_TOY_ROTARY_DIM 8
#define STM32_TOY_HEAD_DIM 8
#define STM32_TOY_VISION_HIDDEN 32
#define STM32_TOY_NUM_IMAGE_TOKENS 4
#define STM32_TOY_TEXT_TOKENS 4
#define STM32_TOY_VOCAB_SIZE 128
#define STM32_TOY_KV_HIDDEN (STM32_TOY_NUM_KV_HEADS * STM32_TOY_HEAD_DIM)
#define STM32_TOY_HIDDEN_FLOATS (STM32_TOY_SEQ_LEN * STM32_TOY_HIDDEN_SIZE)
#define STM32_TOY_INTERMEDIATE_FLOATS (STM32_TOY_SEQ_LEN * STM32_TOY_INTERMEDIATE_SIZE)
#define STM32_TOY_QWEN_WORKSPACE_FLOATS 1536
#define STM32_TOY_MM_WORKSPACE_FLOATS (STM32_TOY_NUM_IMAGE_TOKENS * STM32_TOY_HIDDEN_SIZE)

typedef struct {
  const char* name;
  int seq_len;
  int hidden_size;
  int intermediate_size;
  int num_heads;
  int num_kv_heads;
  int rotary_dim;
  int head_dim;
  int vision_hidden;
  int num_image_tokens;
  int text_tokens;
  int vocab_size;
} llava_stm32_toy_config;

static const llava_stm32_toy_config kToyConfigs[] = {
    {"stm32_toy_s4_h32_i64",
     STM32_TOY_SEQ_LEN,
     STM32_TOY_HIDDEN_SIZE,
     STM32_TOY_INTERMEDIATE_SIZE,
     STM32_TOY_NUM_HEADS,
     STM32_TOY_NUM_KV_HEADS,
     STM32_TOY_ROTARY_DIM,
     STM32_TOY_HEAD_DIM,
     STM32_TOY_VISION_HIDDEN,
     STM32_TOY_NUM_IMAGE_TOKENS,
     STM32_TOY_TEXT_TOKENS,
     STM32_TOY_VOCAB_SIZE},
};

static float g_operator_input[STM32_TOY_HIDDEN_FLOATS];
static float g_operator_weight[STM32_TOY_HIDDEN_SIZE];
static float g_operator_output[STM32_TOY_HIDDEN_FLOATS];

static float g_mm_input[STM32_TOY_NUM_IMAGE_TOKENS * STM32_TOY_VISION_HIDDEN];
static float g_mm_output[STM32_TOY_NUM_IMAGE_TOKENS * STM32_TOY_HIDDEN_SIZE];
static float g_mm_workspace[STM32_TOY_MM_WORKSPACE_FLOATS];
static float g_mm_fc1_weight[STM32_TOY_HIDDEN_SIZE * STM32_TOY_VISION_HIDDEN];
static float g_mm_fc1_bias[STM32_TOY_HIDDEN_SIZE];
static float g_mm_fc2_weight[STM32_TOY_HIDDEN_SIZE * STM32_TOY_HIDDEN_SIZE];
static float g_mm_fc2_bias[STM32_TOY_HIDDEN_SIZE];

static float g_image_embeds[STM32_TOY_NUM_IMAGE_TOKENS * STM32_TOY_HIDDEN_SIZE];
static float g_text_embeds[STM32_TOY_TEXT_TOKENS * STM32_TOY_HIDDEN_SIZE];
static float g_fused_embeds[(STM32_TOY_NUM_IMAGE_TOKENS + STM32_TOY_TEXT_TOKENS) * STM32_TOY_HIDDEN_SIZE];

static float g_qwen_input[STM32_TOY_HIDDEN_FLOATS];
static float g_qwen_output[STM32_TOY_HIDDEN_FLOATS];
static float g_qwen_workspace[STM32_TOY_QWEN_WORKSPACE_FLOATS];
static uint32_t g_qwen_position_ids[STM32_TOY_SEQ_LEN];
static float g_qwen_rope_cos[STM32_TOY_SEQ_LEN * STM32_TOY_ROTARY_DIM];
static float g_qwen_rope_sin[STM32_TOY_SEQ_LEN * STM32_TOY_ROTARY_DIM];
static float g_qwen_input_norm_weight[STM32_TOY_HIDDEN_SIZE];
static float g_qwen_post_attn_norm_weight[STM32_TOY_HIDDEN_SIZE];
static float g_qwen_q_proj_weight[STM32_TOY_HIDDEN_SIZE * STM32_TOY_HIDDEN_SIZE];
static float g_qwen_q_proj_bias[STM32_TOY_HIDDEN_SIZE];
static float g_qwen_k_proj_weight[STM32_TOY_KV_HIDDEN * STM32_TOY_HIDDEN_SIZE];
static float g_qwen_k_proj_bias[STM32_TOY_KV_HIDDEN];
static float g_qwen_v_proj_weight[STM32_TOY_KV_HIDDEN * STM32_TOY_HIDDEN_SIZE];
static float g_qwen_v_proj_bias[STM32_TOY_KV_HIDDEN];
static float g_qwen_o_proj_weight[STM32_TOY_HIDDEN_SIZE * STM32_TOY_HIDDEN_SIZE];
static float g_qwen_o_proj_bias[STM32_TOY_HIDDEN_SIZE];
static float g_qwen_gate_proj_weight[STM32_TOY_INTERMEDIATE_SIZE * STM32_TOY_HIDDEN_SIZE];
static float g_qwen_gate_proj_bias[STM32_TOY_INTERMEDIATE_SIZE];
static float g_qwen_up_proj_weight[STM32_TOY_INTERMEDIATE_SIZE * STM32_TOY_HIDDEN_SIZE];
static float g_qwen_up_proj_bias[STM32_TOY_INTERMEDIATE_SIZE];
static float g_qwen_down_proj_weight[STM32_TOY_HIDDEN_SIZE * STM32_TOY_INTERMEDIATE_SIZE];
static float g_qwen_down_proj_bias[STM32_TOY_HIDDEN_SIZE];

static float g_lm_last_hidden[STM32_TOY_HIDDEN_SIZE];
static float g_lm_logits[STM32_TOY_VOCAB_SIZE];
static float g_lm_weight[STM32_TOY_VOCAB_SIZE * STM32_TOY_HIDDEN_SIZE];
static float g_lm_bias[STM32_TOY_VOCAB_SIZE];

static void print_row_with_validation(const char* benchmark,
                                      const char* variant,
                                      const llava_stm32_toy_config* config,
                                      uint16_t seq_len,
                                      uint16_t hidden_size,
                                      uint16_t intermediate_size,
                                      uint16_t vision_hidden,
                                      uint16_t num_image_tokens,
                                      uint16_t text_tokens,
                                      uint16_t num_heads,
                                      uint16_t num_kv_heads,
                                      uint16_t rotary_dim,
                                      uint16_t head_dim,
                                      uint32_t vocab_size,
                                      size_t workspace_bytes,
                                      uint32_t cycles,
                                      float latency_ms,
                                      const float* output_data,
                                      int output_count,
                                      const char* status) {
  const float checksum = bench_checksum_fp32(output_data, output_count);
  const int has_nan_or_inf = bench_has_nan_or_inf_fp32(output_data, output_count);
  const char* final_status = has_nan_or_inf != 0 ? "nan_or_inf" : status;

  (void)config;

  bench_print_csv_row(benchmark,
                      variant,
                      seq_len,
                      hidden_size,
                      intermediate_size,
                      vision_hidden,
                      num_image_tokens,
                      text_tokens,
                      num_heads,
                      num_kv_heads,
                      rotary_dim,
                      head_dim,
                      vocab_size,
                      STM32_TOY_RUNS,
                      workspace_bytes,
                      cycles,
                      latency_ms,
                      checksum,
                      has_nan_or_inf,
                      final_status);
}

static void run_rmsnorm_toy(const llava_stm32_toy_config* config) {
  tinyengine_status_fp status = STATE_SUCCESS_fp;
  uint32_t start;
  uint32_t end;
  int run;

  bench_fill_sequence(g_operator_input, STM32_TOY_HIDDEN_FLOATS, 0.01f, 0.0f);
  bench_fill_constant(g_operator_weight, STM32_TOY_HIDDEN_SIZE, 1.0f);
  bench_zero(g_operator_output, STM32_TOY_HIDDEN_FLOATS);

  start = benchmark_timer_now();
  for (run = 0; run < STM32_TOY_RUNS; ++run) {
    status = rmsnorm_fp(g_operator_input,
                        g_operator_weight,
                        (uint16_t)config->seq_len,
                        (uint16_t)config->hidden_size,
                        1.0e-6f,
                        g_operator_output);
  }
  end = benchmark_timer_now();

  print_row_with_validation("operator",
                            "rmsnorm_fp",
                            config,
                            (uint16_t)config->seq_len,
                            (uint16_t)config->hidden_size,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            benchmark_timer_elapsed_cycles(start, end),
                            benchmark_timer_elapsed_ms(start, end),
                            g_operator_output,
                            STM32_TOY_HIDDEN_FLOATS,
                            status == STATE_SUCCESS_fp ? "ok" : "fail");
}

static void run_gelu_toy(const llava_stm32_toy_config* config) {
  tinyengine_status_fp status = STATE_SUCCESS_fp;
  uint32_t start;
  uint32_t end;
  int run;

  bench_fill_sequence(g_operator_input, STM32_TOY_HIDDEN_FLOATS, 0.01f, 0.0f);
  bench_zero(g_operator_output, STM32_TOY_HIDDEN_FLOATS);

  start = benchmark_timer_now();
  for (run = 0; run < STM32_TOY_RUNS; ++run) {
    status = gelu_fp((uint16_t)STM32_TOY_HIDDEN_FLOATS, g_operator_input, g_operator_output);
  }
  end = benchmark_timer_now();

  print_row_with_validation("operator",
                            "gelu_fp",
                            config,
                            (uint16_t)config->seq_len,
                            (uint16_t)config->hidden_size,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            benchmark_timer_elapsed_cycles(start, end),
                            benchmark_timer_elapsed_ms(start, end),
                            g_operator_output,
                            STM32_TOY_HIDDEN_FLOATS,
                            status == STATE_SUCCESS_fp ? "ok" : "fail");
}

static void run_silu_toy(const llava_stm32_toy_config* config) {
  tinyengine_status_fp status = STATE_SUCCESS_fp;
  uint32_t start;
  uint32_t end;
  int run;

  bench_fill_sequence(g_operator_input, STM32_TOY_HIDDEN_FLOATS, 0.01f, 0.0f);
  bench_zero(g_operator_output, STM32_TOY_HIDDEN_FLOATS);

  start = benchmark_timer_now();
  for (run = 0; run < STM32_TOY_RUNS; ++run) {
    status = silu_fp((uint16_t)STM32_TOY_HIDDEN_FLOATS, g_operator_input, g_operator_output);
  }
  end = benchmark_timer_now();

  print_row_with_validation("operator",
                            "silu_fp",
                            config,
                            (uint16_t)config->seq_len,
                            (uint16_t)config->hidden_size,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            benchmark_timer_elapsed_cycles(start, end),
                            benchmark_timer_elapsed_ms(start, end),
                            g_operator_output,
                            STM32_TOY_HIDDEN_FLOATS,
                            status == STATE_SUCCESS_fp ? "ok" : "fail");
}

static void run_mm_projector_toy(const llava_stm32_toy_config* config) {
  mm_projector_fp_params params;
  mm_projector_fp_config mm_config;
  const size_t required_workspace = mm_projector_fp_get_workspace_bytes(
      &(mm_projector_fp_config){(uint16_t)config->num_image_tokens, (uint16_t)config->vision_hidden,
                                (uint16_t)config->hidden_size});
  tinyengine_status_fp status = STATE_SUCCESS_fp;
  uint32_t start;
  uint32_t end;
  int run;

  if (required_workspace > sizeof(g_mm_workspace)) {
    bench_print_csv_row("wrapper",
                        "mm_projector_fp",
                        0,
                        (uint16_t)config->hidden_size,
                        0,
                        (uint16_t)config->vision_hidden,
                        (uint16_t)config->num_image_tokens,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        STM32_TOY_RUNS,
                        required_workspace,
                        0,
                        0.0f,
                        0.0f,
                        0,
                        "alloc_fail");
    return;
  }

  bench_fill_sequence(g_mm_input, STM32_TOY_NUM_IMAGE_TOKENS * STM32_TOY_VISION_HIDDEN, 0.01f, 0.0f);
  bench_fill_sequence(g_mm_fc1_weight, STM32_TOY_HIDDEN_SIZE * STM32_TOY_VISION_HIDDEN, 0.001f, 0.0f);
  bench_zero(g_mm_fc1_bias, STM32_TOY_HIDDEN_SIZE);
  bench_fill_sequence(g_mm_fc2_weight, STM32_TOY_HIDDEN_SIZE * STM32_TOY_HIDDEN_SIZE, 0.001f, 0.0f);
  bench_zero(g_mm_fc2_bias, STM32_TOY_HIDDEN_SIZE);
  bench_zero(g_mm_output, STM32_TOY_NUM_IMAGE_TOKENS * STM32_TOY_HIDDEN_SIZE);
  bench_zero(g_mm_workspace, STM32_TOY_MM_WORKSPACE_FLOATS);

  mm_config.num_image_tokens = (uint16_t)config->num_image_tokens;
  mm_config.vision_hidden = (uint16_t)config->vision_hidden;
  mm_config.hidden_size = (uint16_t)config->hidden_size;
  params.config = mm_config;
  params.fc1_weight = g_mm_fc1_weight;
  params.fc1_bias = g_mm_fc1_bias;
  params.fc2_weight = g_mm_fc2_weight;
  params.fc2_bias = g_mm_fc2_bias;

  start = benchmark_timer_now();
  for (run = 0; run < STM32_TOY_RUNS; ++run) {
    status = mm_projector_fp_run(&params, g_mm_input, g_mm_output, g_mm_workspace);
  }
  end = benchmark_timer_now();

  print_row_with_validation("wrapper",
                            "mm_projector_fp",
                            config,
                            0,
                            (uint16_t)config->hidden_size,
                            0,
                            (uint16_t)config->vision_hidden,
                            (uint16_t)config->num_image_tokens,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            required_workspace,
                            benchmark_timer_elapsed_cycles(start, end),
                            benchmark_timer_elapsed_ms(start, end),
                            g_mm_output,
                            STM32_TOY_NUM_IMAGE_TOKENS * STM32_TOY_HIDDEN_SIZE,
                            status == STATE_SUCCESS_fp ? "ok" : "fail");
}

static void run_image_text_fusion_toy(const llava_stm32_toy_config* config) {
  image_text_fusion_fp_config fusion_config;
  tinyengine_status_fp status = STATE_SUCCESS_fp;
  uint32_t start;
  uint32_t end;
  int run;

  bench_fill_sequence(g_image_embeds, STM32_TOY_NUM_IMAGE_TOKENS * STM32_TOY_HIDDEN_SIZE, 0.01f, 0.0f);
  bench_fill_sequence(g_text_embeds, STM32_TOY_TEXT_TOKENS * STM32_TOY_HIDDEN_SIZE, 0.01f, 0.0f);
  bench_zero(g_fused_embeds, (STM32_TOY_NUM_IMAGE_TOKENS + STM32_TOY_TEXT_TOKENS) * STM32_TOY_HIDDEN_SIZE);

  fusion_config.image_tokens = (uint16_t)config->num_image_tokens;
  fusion_config.text_tokens = (uint16_t)config->text_tokens;
  fusion_config.hidden_size = (uint16_t)config->hidden_size;

  start = benchmark_timer_now();
  for (run = 0; run < STM32_TOY_RUNS; ++run) {
    status = image_text_fusion_fp_run(&fusion_config, g_image_embeds, g_text_embeds, g_fused_embeds, NULL);
  }
  end = benchmark_timer_now();

  print_row_with_validation("wrapper",
                            "image_text_fusion_fp",
                            config,
                            0,
                            (uint16_t)config->hidden_size,
                            0,
                            0,
                            (uint16_t)config->num_image_tokens,
                            (uint16_t)config->text_tokens,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            benchmark_timer_elapsed_cycles(start, end),
                            benchmark_timer_elapsed_ms(start, end),
                            g_fused_embeds,
                            (STM32_TOY_NUM_IMAGE_TOKENS + STM32_TOY_TEXT_TOKENS) * STM32_TOY_HIDDEN_SIZE,
                            status == STATE_SUCCESS_fp ? "ok" : "fail");
}

static void run_qwen_block_toy(const llava_stm32_toy_config* config) {
  qwen_block_fp_params params;
  qwen_block_fp_config qwen_config;
  const size_t required_workspace = qwen_block_fp_get_workspace_bytes(
      &(qwen_block_fp_config){(uint16_t)config->seq_len,
                              (uint16_t)config->hidden_size,
                              (uint16_t)config->intermediate_size,
                              (uint16_t)config->num_heads,
                              (uint16_t)config->num_kv_heads,
                              (uint16_t)config->rotary_dim,
                              1.0e-6f,
                              true});
  tinyengine_status_fp status = STATE_SUCCESS_fp;
  uint32_t start;
  uint32_t end;
  int run;

  if (required_workspace > sizeof(g_qwen_workspace)) {
    bench_print_csv_row("wrapper",
                        "qwen_block_fp",
                        (uint16_t)config->seq_len,
                        (uint16_t)config->hidden_size,
                        (uint16_t)config->intermediate_size,
                        0,
                        0,
                        0,
                        (uint16_t)config->num_heads,
                        (uint16_t)config->num_kv_heads,
                        (uint16_t)config->rotary_dim,
                        (uint16_t)config->head_dim,
                        0,
                        STM32_TOY_RUNS,
                        required_workspace,
                        0,
                        0.0f,
                        0.0f,
                        0,
                        "alloc_fail");
    return;
  }

  bench_fill_sequence(g_qwen_input, STM32_TOY_HIDDEN_FLOATS, 0.01f, 0.0f);
  bench_zero(g_qwen_output, STM32_TOY_HIDDEN_FLOATS);
  bench_zero(g_qwen_workspace, STM32_TOY_QWEN_WORKSPACE_FLOATS);
  bench_fill_u32_sequence(g_qwen_position_ids, STM32_TOY_SEQ_LEN, STM32_TOY_SEQ_LEN);
  bench_fill_rope_tables(g_qwen_rope_cos, g_qwen_rope_sin, STM32_TOY_SEQ_LEN, STM32_TOY_ROTARY_DIM);
  bench_fill_constant(g_qwen_input_norm_weight, STM32_TOY_HIDDEN_SIZE, 1.0f);
  bench_fill_constant(g_qwen_post_attn_norm_weight, STM32_TOY_HIDDEN_SIZE, 1.0f);
  bench_fill_sequence(g_qwen_q_proj_weight, STM32_TOY_HIDDEN_SIZE * STM32_TOY_HIDDEN_SIZE, 0.001f, 0.0f);
  bench_zero(g_qwen_q_proj_bias, STM32_TOY_HIDDEN_SIZE);
  bench_fill_sequence(g_qwen_k_proj_weight, STM32_TOY_KV_HIDDEN * STM32_TOY_HIDDEN_SIZE, 0.001f, 0.0f);
  bench_zero(g_qwen_k_proj_bias, STM32_TOY_KV_HIDDEN);
  bench_fill_sequence(g_qwen_v_proj_weight, STM32_TOY_KV_HIDDEN * STM32_TOY_HIDDEN_SIZE, 0.001f, 0.0f);
  bench_zero(g_qwen_v_proj_bias, STM32_TOY_KV_HIDDEN);
  bench_fill_sequence(g_qwen_o_proj_weight, STM32_TOY_HIDDEN_SIZE * STM32_TOY_HIDDEN_SIZE, 0.001f, 0.0f);
  bench_zero(g_qwen_o_proj_bias, STM32_TOY_HIDDEN_SIZE);
  bench_fill_sequence(g_qwen_gate_proj_weight, STM32_TOY_INTERMEDIATE_SIZE * STM32_TOY_HIDDEN_SIZE, 0.001f, 0.0f);
  bench_zero(g_qwen_gate_proj_bias, STM32_TOY_INTERMEDIATE_SIZE);
  bench_fill_sequence(g_qwen_up_proj_weight, STM32_TOY_INTERMEDIATE_SIZE * STM32_TOY_HIDDEN_SIZE, 0.001f, 0.0f);
  bench_zero(g_qwen_up_proj_bias, STM32_TOY_INTERMEDIATE_SIZE);
  bench_fill_sequence(g_qwen_down_proj_weight, STM32_TOY_HIDDEN_SIZE * STM32_TOY_INTERMEDIATE_SIZE, 0.001f, 0.0f);
  bench_zero(g_qwen_down_proj_bias, STM32_TOY_HIDDEN_SIZE);

  qwen_config.seq_len = (uint16_t)config->seq_len;
  qwen_config.hidden_size = (uint16_t)config->hidden_size;
  qwen_config.intermediate_size = (uint16_t)config->intermediate_size;
  qwen_config.num_attention_heads = (uint16_t)config->num_heads;
  qwen_config.num_key_value_heads = (uint16_t)config->num_kv_heads;
  qwen_config.rotary_dim = (uint16_t)config->rotary_dim;
  qwen_config.rms_norm_eps = 1.0e-6f;
  qwen_config.causal = true;

  params.config = qwen_config;
  params.position_ids = g_qwen_position_ids;
  params.rope_cos = g_qwen_rope_cos;
  params.rope_sin = g_qwen_rope_sin;
  params.input_norm_weight = g_qwen_input_norm_weight;
  params.post_attn_norm_weight = g_qwen_post_attn_norm_weight;
  params.q_proj_weight = g_qwen_q_proj_weight;
  params.q_proj_bias = g_qwen_q_proj_bias;
  params.k_proj_weight = g_qwen_k_proj_weight;
  params.k_proj_bias = g_qwen_k_proj_bias;
  params.v_proj_weight = g_qwen_v_proj_weight;
  params.v_proj_bias = g_qwen_v_proj_bias;
  params.o_proj_weight = g_qwen_o_proj_weight;
  params.o_proj_bias = g_qwen_o_proj_bias;
  params.gate_proj_weight = g_qwen_gate_proj_weight;
  params.gate_proj_bias = g_qwen_gate_proj_bias;
  params.up_proj_weight = g_qwen_up_proj_weight;
  params.up_proj_bias = g_qwen_up_proj_bias;
  params.down_proj_weight = g_qwen_down_proj_weight;
  params.down_proj_bias = g_qwen_down_proj_bias;

  start = benchmark_timer_now();
  for (run = 0; run < STM32_TOY_RUNS; ++run) {
    status = qwen_block_fp_run(&params, g_qwen_input, g_qwen_output, g_qwen_workspace);
  }
  end = benchmark_timer_now();

  print_row_with_validation("wrapper",
                            config->name,
                            config,
                            (uint16_t)config->seq_len,
                            (uint16_t)config->hidden_size,
                            (uint16_t)config->intermediate_size,
                            0,
                            0,
                            0,
                            (uint16_t)config->num_heads,
                            (uint16_t)config->num_kv_heads,
                            (uint16_t)config->rotary_dim,
                            (uint16_t)config->head_dim,
                            0,
                            required_workspace,
                            benchmark_timer_elapsed_cycles(start, end),
                            benchmark_timer_elapsed_ms(start, end),
                            g_qwen_output,
                            STM32_TOY_HIDDEN_FLOATS,
                            status == STATE_SUCCESS_fp ? "ok" : "fail");
}

static void run_lm_head_last_token_toy(const llava_stm32_toy_config* config) {
  lm_head_last_token_fp_params params;
  lm_head_last_token_fp_config lm_config;
  tinyengine_status_fp status = STATE_SUCCESS_fp;
  uint32_t start;
  uint32_t end;
  int run;

  bench_fill_sequence(g_lm_last_hidden, STM32_TOY_HIDDEN_SIZE, 0.01f, 0.0f);
  bench_fill_sequence(g_lm_weight, STM32_TOY_VOCAB_SIZE * STM32_TOY_HIDDEN_SIZE, 0.001f, 0.0f);
  bench_zero(g_lm_bias, STM32_TOY_VOCAB_SIZE);
  bench_zero(g_lm_logits, STM32_TOY_VOCAB_SIZE);

  lm_config.hidden_size = (uint16_t)config->hidden_size;
  lm_config.vocab_size = (uint32_t)config->vocab_size;
  params.config = lm_config;
  params.weight = g_lm_weight;
  params.bias = g_lm_bias;

  start = benchmark_timer_now();
  for (run = 0; run < STM32_TOY_RUNS; ++run) {
    status = lm_head_last_token_fp_run(&params, g_lm_last_hidden, g_lm_logits, NULL);
  }
  end = benchmark_timer_now();

  print_row_with_validation("wrapper",
                            "lm_head_last_token_fp",
                            config,
                            1,
                            (uint16_t)config->hidden_size,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            (uint32_t)config->vocab_size,
                            0,
                            benchmark_timer_elapsed_cycles(start, end),
                            benchmark_timer_elapsed_ms(start, end),
                            g_lm_logits,
                            STM32_TOY_VOCAB_SIZE,
                            status == STATE_SUCCESS_fp ? "ok" : "fail");
}

void llava_microbench_stm32n6_run(void) {
  const llava_stm32_toy_config* config = &kToyConfigs[0];

  benchmark_timer_init();
  bench_print_csv_header();

  run_rmsnorm_toy(config);
  run_gelu_toy(config);
  run_silu_toy(config);
  run_mm_projector_toy(config);
  run_image_text_fusion_toy(config);
  run_qwen_block_toy(config);
  run_lm_head_last_token_toy(config);
}

#if defined(LLAVA_MICROBENCH_STANDALONE_MAIN)
int main(void) {
  llava_microbench_stm32n6_run();

#if defined(__arm__) || defined(__thumb__)
  for (;;) {
  }
#else
  return 0;
#endif
}
#endif
