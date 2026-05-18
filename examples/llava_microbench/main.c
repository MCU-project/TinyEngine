#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "llava_microbench.h"

#define DEFAULT_RUNS 50
#define DEFAULT_TEXT_TOKENS 32
#define DEFAULT_DTYPE_BYTES 4

typedef struct {
  const char* name;
  qwen_block_fp_config config;
} qwen_bench_entry;

typedef struct {
  const char* name;
  kv_cache_fp_config config;
  uint16_t seq_len;
} kv_cache_bench_entry;

static const qwen_bench_entry k_qwen_configs[] = {
    {"qwen_toy_s4_h32_i64", {4, 32, 64, 4, 1, 8, 1.0e-6f, true}},
    {"qwen_toy_s8_h64_i128", {8, 64, 128, 4, 1, 16, 1.0e-6f, true}},
    {"qwen_toy_s16_h128_i256", {16, 128, 256, 8, 2, 32, 1.0e-6f, true}},
};

static const uint32_t k_lm_head_vocab_sizes[] = {1024, 4096, 8192, 16384, 151647};
static const uint16_t k_image_token_counts[] = {16, 32, 64, 196, 729};

static const kv_cache_bench_entry k_kv_cache_configs[] = {
    {"kv_cache_l4_s32_h1_d16", {4, 32, 1, 16}, 32},
    {"kv_cache_l8_s64_h2_d16", {8, 64, 2, 16}, 64},
    {"kv_cache_l24_s128_h2_d64", {24, 128, 2, 64}, 128},
};

static float checksum_fp(const float* data, size_t size) {
  size_t i;
  float acc = 0.0f;
  for (i = 0; i < size; ++i) {
    acc += data[i];
  }
  return acc;
}

static void fill_tensor(float* data, size_t size, float scale, float offset) {
  size_t i;
  for (i = 0; i < size; ++i) {
    const int centered = (int)(i % 17) - 8;
    data[i] = (float)centered * scale + offset;
  }
}

static void fill_u32(uint32_t* data, size_t size, uint32_t mod) {
  size_t i;
  for (i = 0; i < size; ++i) {
    data[i] = (uint32_t)(i % mod);
  }
}

static void fill_rope_tables(float* cos_table, float* sin_table, uint16_t max_pos, uint16_t rotary_dim) {
  uint16_t pos;
  uint16_t d;
  for (pos = 0; pos < max_pos; ++pos) {
    for (d = 0; d < rotary_dim; ++d) {
      const float angle = 0.01f * (float)(pos + 1) * (float)(d + 1);
      cos_table[(size_t)pos * rotary_dim + d] = cosf(angle);
      sin_table[(size_t)pos * rotary_dim + d] = sinf(angle);
    }
  }
}

static double elapsed_ms(clock_t start_tick, clock_t end_tick) {
  return ((double)(end_tick - start_tick) * 1000.0) / (double)CLOCKS_PER_SEC;
}

static void print_csv_header(void) {
  printf("benchmark,variant,seq_len,hidden_size,intermediate_size,vision_hidden,num_image_tokens,text_tokens,");
  printf("num_heads,num_kv_heads,rotary_dim,head_dim,num_layers,vocab_size,dtype_bytes,runs,");
  printf("workspace_bytes,cache_bytes,elapsed_ms,avg_ms,checksum,status\n");
}

static void print_csv_row(const char* benchmark,
                          const char* variant,
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
                          uint16_t num_layers,
                          uint32_t vocab_size,
                          uint16_t dtype_bytes,
                          int runs,
                          size_t workspace_bytes,
                          size_t cache_bytes,
                          double total_ms,
                          float checksum,
                          const char* status) {
  printf("%s,%s,%u,%u,%u,%u,%u,%u,%u,%u,%u,%u,%u,%u,%u,%d,%u,%u,%.6f,%.6f,%.6f,%s\n",
         benchmark,
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
         num_layers,
         vocab_size,
         dtype_bytes,
         runs,
         (unsigned int)workspace_bytes,
         (unsigned int)cache_bytes,
         total_ms,
         runs > 0 ? total_ms / (double)runs : 0.0,
         checksum,
         status);
}

static void benchmark_rmsnorm(int runs) {
  const uint16_t rows = 8;
  const uint16_t cols = 64;
  float input_data[rows * cols];
  float weight_data[cols];
  float output_data[rows * cols];
  tinyengine_status_fp status = STATE_SUCCESS_fp;
  clock_t start_tick;
  clock_t end_tick;
  int run;

  fill_tensor(input_data, rows * cols, 0.01f, 0.0f);
  fill_tensor(weight_data, cols, 0.0f, 1.0f);

  start_tick = clock();
  for (run = 0; run < runs; ++run) {
    status = rmsnorm_fp(input_data, weight_data, rows, cols, 1.0e-6f, output_data);
  }
  end_tick = clock();

  print_csv_row("operator", "rmsnorm_fp", rows, cols, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, DEFAULT_DTYPE_BYTES,
                runs, 0, 0, elapsed_ms(start_tick, end_tick), checksum_fp(output_data, rows * cols),
                status == STATE_SUCCESS_fp ? "ok" : "fail");
}

static void benchmark_rotary_embedding(int runs) {
  const uint16_t seq_len = 8;
  const uint16_t num_heads = 4;
  const uint16_t num_kv_heads = 2;
  const uint16_t head_dim = 16;
  const uint16_t rotary_dim = 16;
  float query[seq_len * num_heads * head_dim];
  float key[seq_len * num_kv_heads * head_dim];
  float cos_table[seq_len * rotary_dim];
  float sin_table[seq_len * rotary_dim];
  uint32_t position_ids[seq_len];
  tinyengine_status_fp status = STATE_SUCCESS_fp;
  clock_t start_tick;
  clock_t end_tick;
  int run;

  fill_tensor(query, seq_len * num_heads * head_dim, 0.001f, 0.0f);
  fill_tensor(key, seq_len * num_kv_heads * head_dim, 0.001f, 0.0f);
  fill_rope_tables(cos_table, sin_table, seq_len, rotary_dim);
  fill_u32(position_ids, seq_len, seq_len);

  start_tick = clock();
  for (run = 0; run < runs; ++run) {
    status = rotary_embedding_fp(query, key, seq_len, num_heads, num_kv_heads, head_dim,
                                 position_ids, cos_table, sin_table, rotary_dim);
  }
  end_tick = clock();

  print_csv_row("operator", "rotary_embedding_fp", seq_len, head_dim * num_heads, 0, 0, 0, 0, num_heads,
                num_kv_heads, rotary_dim, head_dim, 0, 0, DEFAULT_DTYPE_BYTES, runs, 0, 0,
                elapsed_ms(start_tick, end_tick),
                checksum_fp(query, seq_len * num_heads * head_dim) +
                    checksum_fp(key, seq_len * num_kv_heads * head_dim),
                status == STATE_SUCCESS_fp ? "ok" : "fail");
}

static void benchmark_repeat_kv(int runs) {
  const uint16_t seq_len = 8;
  const uint16_t num_kv_heads = 2;
  const uint16_t head_dim = 16;
  const uint16_t n_rep = 4;
  float input_data[seq_len * num_kv_heads * head_dim];
  float output_data[seq_len * num_kv_heads * n_rep * head_dim];
  tinyengine_status_fp status = STATE_SUCCESS_fp;
  clock_t start_tick;
  clock_t end_tick;
  int run;

  fill_tensor(input_data, seq_len * num_kv_heads * head_dim, 0.002f, 0.0f);

  start_tick = clock();
  for (run = 0; run < runs; ++run) {
    status = repeat_kv_fp(input_data, seq_len, num_kv_heads, head_dim, n_rep, output_data);
  }
  end_tick = clock();

  print_csv_row("operator", "repeat_kv_fp", seq_len, head_dim * num_kv_heads * n_rep, 0, 0, 0, 0, 0,
                num_kv_heads, 0, head_dim, 0, 0, DEFAULT_DTYPE_BYTES, runs, 0, 0,
                elapsed_ms(start_tick, end_tick),
                checksum_fp(output_data, seq_len * num_kv_heads * n_rep * head_dim),
                status == STATE_SUCCESS_fp ? "ok" : "fail");
}

static void benchmark_silu(int runs) {
  const uint16_t size = 512;
  float input_data[size];
  float output_data[size];
  tinyengine_status_fp status = STATE_SUCCESS_fp;
  clock_t start_tick;
  clock_t end_tick;
  int run;

  fill_tensor(input_data, size, 0.01f, 0.0f);

  start_tick = clock();
  for (run = 0; run < runs; ++run) {
    status = silu_fp(size, input_data, output_data);
  }
  end_tick = clock();

  print_csv_row("operator", "silu_fp", size, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, DEFAULT_DTYPE_BYTES, runs,
                0, 0, elapsed_ms(start_tick, end_tick), checksum_fp(output_data, size),
                status == STATE_SUCCESS_fp ? "ok" : "fail");
}

static void benchmark_gelu(int runs) {
  const uint16_t size = 512;
  float input_data[size];
  float output_data[size];
  tinyengine_status_fp status = STATE_SUCCESS_fp;
  clock_t start_tick;
  clock_t end_tick;
  int run;

  fill_tensor(input_data, size, 0.01f, 0.0f);

  start_tick = clock();
  for (run = 0; run < runs; ++run) {
    status = gelu_fp(size, input_data, output_data);
  }
  end_tick = clock();

  print_csv_row("operator", "gelu_fp", size, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, DEFAULT_DTYPE_BYTES, runs,
                0, 0, elapsed_ms(start_tick, end_tick), checksum_fp(output_data, size),
                status == STATE_SUCCESS_fp ? "ok" : "fail");
}

static void benchmark_qkv_attention(int runs) {
  const uint16_t query_len = 8;
  const uint16_t key_len = 8;
  const uint16_t num_heads = 4;
  const uint16_t head_dim = 16;
  float query[query_len * num_heads * head_dim];
  float key[key_len * num_heads * head_dim];
  float value[key_len * num_heads * head_dim];
  float output_data[query_len * num_heads * head_dim];
  float score_buffer[query_len * key_len];
  tinyengine_status_fp status = STATE_SUCCESS_fp;
  clock_t start_tick;
  clock_t end_tick;
  int run;

  fill_tensor(query, query_len * num_heads * head_dim, 0.002f, 0.0f);
  fill_tensor(key, key_len * num_heads * head_dim, 0.002f, 0.0f);
  fill_tensor(value, key_len * num_heads * head_dim, 0.002f, 0.0f);

  start_tick = clock();
  for (run = 0; run < runs; ++run) {
    status = qkv_attention_fp(query, key, value, query_len, key_len, num_heads, head_dim, true,
                              output_data, score_buffer);
  }
  end_tick = clock();

  print_csv_row("operator", "qkv_attention_fp", query_len, num_heads * head_dim, 0, 0, 0, 0, num_heads,
                num_heads, 0, head_dim, 0, 0, DEFAULT_DTYPE_BYTES, runs, sizeof(score_buffer), 0,
                elapsed_ms(start_tick, end_tick), checksum_fp(output_data, query_len * num_heads * head_dim),
                status == STATE_SUCCESS_fp ? "ok" : "fail");
}

static void benchmark_embedding_lookup(int runs) {
  const uint16_t seq_len = 8;
  const uint16_t hidden_size = 64;
  const uint16_t vocab_size = 32;
  uint32_t token_ids[seq_len];
  float embedding_table[vocab_size * hidden_size];
  float output_data[seq_len * hidden_size];
  tinyengine_status_fp status = STATE_SUCCESS_fp;
  clock_t start_tick;
  clock_t end_tick;
  int run;

  fill_u32(token_ids, seq_len, vocab_size);
  fill_tensor(embedding_table, vocab_size * hidden_size, 0.001f, 0.0f);

  start_tick = clock();
  for (run = 0; run < runs; ++run) {
    status = embedding_lookup_fp(seq_len, hidden_size, token_ids, embedding_table, output_data);
  }
  end_tick = clock();

  print_csv_row("operator", "embedding_lookup_fp", seq_len, hidden_size, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                vocab_size, DEFAULT_DTYPE_BYTES, runs, 0, 0, elapsed_ms(start_tick, end_tick),
                checksum_fp(output_data, seq_len * hidden_size), status == STATE_SUCCESS_fp ? "ok" : "fail");
}

static void benchmark_argmax(int runs) {
  const uint16_t size = 512;
  float input_data[size];
  uint16_t output_index = 0;
  float output_value = 0.0f;
  tinyengine_status_fp status = STATE_SUCCESS_fp;
  clock_t start_tick;
  clock_t end_tick;
  int run;

  fill_tensor(input_data, size, 0.01f, 0.0f);
  input_data[size - 1] = 10.0f;

  start_tick = clock();
  for (run = 0; run < runs; ++run) {
    status = argmax_fp(size, input_data, &output_index, &output_value);
  }
  end_tick = clock();

  print_csv_row("operator", "argmax_fp", size, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, DEFAULT_DTYPE_BYTES, runs,
                0, 0, elapsed_ms(start_tick, end_tick), output_value + (float)output_index,
                status == STATE_SUCCESS_fp ? "ok" : "fail");
}

static void benchmark_mm_projector(int runs) {
  mm_projector_fp_params params;
  mm_projector_fp_config config;
  const uint16_t num_image_tokens = 16;
  const uint16_t vision_hidden = 128;
  const uint16_t hidden_size = 64;
  const size_t input_floats = (size_t)num_image_tokens * vision_hidden;
  const size_t output_floats = (size_t)num_image_tokens * hidden_size;
  const size_t fc1_weights = (size_t)hidden_size * vision_hidden;
  const size_t fc2_weights = (size_t)hidden_size * hidden_size;
  const size_t workspace_floats =
      mm_projector_fp_get_workspace_floats(&(mm_projector_fp_config){num_image_tokens, vision_hidden, hidden_size});
  float* input_data = (float*)malloc(sizeof(float) * input_floats);
  float* output_data = (float*)malloc(sizeof(float) * output_floats);
  float* workspace = (float*)malloc(sizeof(float) * workspace_floats);
  float* fc1_weight = (float*)malloc(sizeof(float) * fc1_weights);
  float* fc1_bias = (float*)malloc(sizeof(float) * hidden_size);
  float* fc2_weight = (float*)malloc(sizeof(float) * fc2_weights);
  float* fc2_bias = (float*)malloc(sizeof(float) * hidden_size);
  tinyengine_status_fp status = STATE_SUCCESS_fp;
  clock_t start_tick;
  clock_t end_tick;
  int run;

  if (input_data == NULL || output_data == NULL || workspace == NULL || fc1_weight == NULL || fc1_bias == NULL ||
      fc2_weight == NULL || fc2_bias == NULL) {
    print_csv_row("wrapper", "mm_projector_fp", 0, hidden_size, 0, vision_hidden, num_image_tokens, 0, 0, 0, 0,
                  0, 0, 0, DEFAULT_DTYPE_BYTES, runs, workspace_floats * sizeof(float), 0, 0.0, 0.0f, "alloc_fail");
    goto cleanup;
  }

  fill_tensor(input_data, input_floats, 0.01f, 0.0f);
  fill_tensor(fc1_weight, fc1_weights, 0.001f, 0.0f);
  fill_tensor(fc1_bias, hidden_size, 0.0f, 0.0f);
  fill_tensor(fc2_weight, fc2_weights, 0.001f, 0.0f);
  fill_tensor(fc2_bias, hidden_size, 0.0f, 0.0f);

  config.num_image_tokens = num_image_tokens;
  config.vision_hidden = vision_hidden;
  config.hidden_size = hidden_size;

  params.config = config;
  params.fc1_weight = fc1_weight;
  params.fc1_bias = fc1_bias;
  params.fc2_weight = fc2_weight;
  params.fc2_bias = fc2_bias;

  start_tick = clock();
  for (run = 0; run < runs; ++run) {
    status = mm_projector_fp_run(&params, input_data, output_data, workspace);
  }
  end_tick = clock();

  print_csv_row("wrapper", "mm_projector_fp", 0, hidden_size, 0, vision_hidden, num_image_tokens, 0, 0, 0, 0, 0,
                0, 0, DEFAULT_DTYPE_BYTES, runs, mm_projector_fp_get_workspace_bytes(&config), 0,
                elapsed_ms(start_tick, end_tick), checksum_fp(output_data, output_floats),
                status == STATE_SUCCESS_fp ? "ok" : "fail");

cleanup:
  free(input_data);
  free(output_data);
  free(workspace);
  free(fc1_weight);
  free(fc1_bias);
  free(fc2_weight);
  free(fc2_bias);
}

static void benchmark_lm_head_last_token(int runs) {
  const uint16_t hidden_size = 128;
  size_t i;

  for (i = 0; i < sizeof(k_lm_head_vocab_sizes) / sizeof(k_lm_head_vocab_sizes[0]); ++i) {
    const uint32_t vocab_size = k_lm_head_vocab_sizes[i];
    const size_t weight_floats = (size_t)vocab_size * hidden_size;
    float* last_hidden_state = (float*)malloc(sizeof(float) * hidden_size);
    float* weight = (float*)malloc(sizeof(float) * weight_floats);
    float* bias = (float*)malloc(sizeof(float) * vocab_size);
    float* logits = (float*)malloc(sizeof(float) * vocab_size);
    lm_head_last_token_fp_params params;
    tinyengine_status_fp status = STATE_SUCCESS_fp;
    clock_t start_tick;
    clock_t end_tick;
    int run;

    if (last_hidden_state == NULL || weight == NULL || bias == NULL || logits == NULL) {
      print_csv_row("wrapper", "lm_head_last_token_fp", 1, hidden_size, 0, 0, 0, 1, 0, 0, 0, 0, 0, vocab_size,
                    DEFAULT_DTYPE_BYTES, runs, 0, 0, 0.0, 0.0f, "alloc_fail");
      free(last_hidden_state);
      free(weight);
      free(bias);
      free(logits);
      continue;
    }

    fill_tensor(last_hidden_state, hidden_size, 0.01f, 0.0f);
    fill_tensor(weight, weight_floats, 0.001f, 0.0f);
    fill_tensor(bias, vocab_size, 0.0f, 0.0f);

    params.config.hidden_size = hidden_size;
    params.config.vocab_size = vocab_size;
    params.weight = weight;
    params.bias = bias;

    start_tick = clock();
    for (run = 0; run < runs; ++run) {
      status = lm_head_last_token_fp_run(&params, last_hidden_state, logits, NULL);
    }
    end_tick = clock();

    print_csv_row("wrapper", "lm_head_last_token_fp", 1, hidden_size, 0, 0, 0, 1, 0, 0, 0, 0, 0, vocab_size,
                  DEFAULT_DTYPE_BYTES, runs, 0, 0, elapsed_ms(start_tick, end_tick), checksum_fp(logits, vocab_size),
                  status == STATE_SUCCESS_fp ? "ok" : "fail");

    free(last_hidden_state);
    free(weight);
    free(bias);
    free(logits);
  }
}

static void benchmark_kv_cache(int runs) {
  size_t i;

  for (i = 0; i < sizeof(k_kv_cache_configs) / sizeof(k_kv_cache_configs[0]); ++i) {
    const kv_cache_bench_entry* entry = &k_kv_cache_configs[i];
    const kv_cache_fp_config* config = &entry->config;
    const uint16_t token_size = config->num_kv_heads * config->head_dim;
    const size_t layer_stride = (size_t)config->max_seq_len * token_size;
    const size_t key_cache_floats = (size_t)config->num_layers * layer_stride;
    float* key_cache = (float*)malloc(sizeof(float) * key_cache_floats);
    float* value_cache = (float*)malloc(sizeof(float) * key_cache_floats);
    float* key_token = (float*)malloc(sizeof(float) * token_size);
    float* value_token = (float*)malloc(sizeof(float) * token_size);
    float* key_read = (float*)malloc(sizeof(float) * (size_t)entry->seq_len * token_size);
    float* value_read = (float*)malloc(sizeof(float) * (size_t)entry->seq_len * token_size);
    tinyengine_status_fp status = STATE_SUCCESS_fp;
    clock_t start_tick;
    clock_t end_tick;
    double append_ms;
    double read_ms;
    int run;
    uint16_t layer_idx;
    uint16_t seq_pos;

    if (key_cache == NULL || value_cache == NULL || key_token == NULL || value_token == NULL || key_read == NULL ||
        value_read == NULL) {
      print_csv_row("wrapper", "kv_cache_alloc", entry->seq_len, 0, 0, 0, 0, 0, 0, config->num_kv_heads, 0,
                    config->head_dim, config->num_layers, 0, DEFAULT_DTYPE_BYTES, runs, 0,
                    kv_cache_fp_get_cache_bytes(config), 0.0, 0.0f, "alloc_fail");
      free(key_cache);
      free(value_cache);
      free(key_token);
      free(value_token);
      free(key_read);
      free(value_read);
      continue;
    }

    fill_tensor(key_cache, key_cache_floats, 0.0f, 0.0f);
    fill_tensor(value_cache, key_cache_floats, 0.0f, 0.0f);
    fill_tensor(key_token, token_size, 0.001f, 0.0f);
    fill_tensor(value_token, token_size, 0.001f, 0.0f);

    print_csv_row("wrapper", "kv_cache_size", entry->seq_len, 0, 0, 0, 0, 0, 0, config->num_kv_heads, 0,
                  config->head_dim, config->num_layers, 0, DEFAULT_DTYPE_BYTES, runs, 0,
                  kv_cache_fp_get_cache_bytes(config), 0.0, 0.0f, "ok");

    start_tick = clock();
    for (run = 0; run < runs; ++run) {
      for (layer_idx = 0; layer_idx < config->num_layers; ++layer_idx) {
        for (seq_pos = 0; seq_pos < entry->seq_len; ++seq_pos) {
          status = kv_cache_append_fp(config, key_cache, value_cache, layer_idx, seq_pos, key_token, value_token);
        }
      }
    }
    end_tick = clock();
    append_ms = elapsed_ms(start_tick, end_tick);

    print_csv_row("wrapper", "kv_cache_append_fp", entry->seq_len, 0, 0, 0, 0, 0, 0, config->num_kv_heads, 0,
                  config->head_dim, config->num_layers, 0, DEFAULT_DTYPE_BYTES, runs, 0,
                  kv_cache_fp_get_cache_bytes(config), append_ms,
                  checksum_fp(key_cache, key_cache_floats) + checksum_fp(value_cache, key_cache_floats),
                  status == STATE_SUCCESS_fp ? "ok" : "fail");

    start_tick = clock();
    for (run = 0; run < runs; ++run) {
      for (layer_idx = 0; layer_idx < config->num_layers; ++layer_idx) {
        status = kv_cache_read_fp(config, key_cache, value_cache, layer_idx, entry->seq_len, key_read, value_read);
      }
    }
    end_tick = clock();
    read_ms = elapsed_ms(start_tick, end_tick);

    print_csv_row("wrapper", "kv_cache_read_fp", entry->seq_len, 0, 0, 0, 0, 0, 0, config->num_kv_heads, 0,
                  config->head_dim, config->num_layers, 0, DEFAULT_DTYPE_BYTES, runs, 0,
                  kv_cache_fp_get_cache_bytes(config), read_ms,
                  checksum_fp(key_read, (size_t)entry->seq_len * token_size) +
                      checksum_fp(value_read, (size_t)entry->seq_len * token_size),
                  status == STATE_SUCCESS_fp ? "ok" : "fail");

    free(key_cache);
    free(value_cache);
    free(key_token);
    free(value_token);
    free(key_read);
    free(value_read);
  }
}

static void benchmark_image_text_fusion(int runs) {
  const uint16_t text_tokens = DEFAULT_TEXT_TOKENS;
  const uint16_t hidden_size = 128;
  size_t i;

  for (i = 0; i < sizeof(k_image_token_counts) / sizeof(k_image_token_counts[0]); ++i) {
    const uint16_t image_tokens = k_image_token_counts[i];
    const image_text_fusion_fp_config config = {image_tokens, text_tokens, hidden_size};
    const size_t image_floats = (size_t)image_tokens * hidden_size;
    const size_t text_floats = (size_t)text_tokens * hidden_size;
    const size_t fused_floats = (size_t)(image_tokens + text_tokens) * hidden_size;
    float* image_embeds = (float*)malloc(sizeof(float) * image_floats);
    float* text_embeds = (float*)malloc(sizeof(float) * text_floats);
    float* fused_output = (float*)malloc(sizeof(float) * fused_floats);
    tinyengine_status_fp status = STATE_SUCCESS_fp;
    clock_t start_tick;
    clock_t end_tick;
    int run;

    if (image_embeds == NULL || text_embeds == NULL || fused_output == NULL) {
      print_csv_row("wrapper", "image_text_fusion_fp", image_tokens + text_tokens, hidden_size, 0, 0, image_tokens,
                    text_tokens, 0, 0, 0, 0, 0, 0, DEFAULT_DTYPE_BYTES, runs, 0, 0, 0.0, 0.0f, "alloc_fail");
      free(image_embeds);
      free(text_embeds);
      free(fused_output);
      continue;
    }

    fill_tensor(image_embeds, image_floats, 0.01f, 0.0f);
    fill_tensor(text_embeds, text_floats, 0.01f, 0.0f);

    start_tick = clock();
    for (run = 0; run < runs; ++run) {
      status = image_text_fusion_fp_run(&config, image_embeds, text_embeds, fused_output, NULL);
    }
    end_tick = clock();

    print_csv_row("wrapper", "image_text_fusion_fp", image_tokens + text_tokens, hidden_size, 0, 0, image_tokens,
                  text_tokens, 0, 0, 0, 0, 0, 0, DEFAULT_DTYPE_BYTES, runs, 0, 0,
                  elapsed_ms(start_tick, end_tick), checksum_fp(fused_output, fused_floats),
                  status == STATE_SUCCESS_fp ? "ok" : "fail");

    free(image_embeds);
    free(text_embeds);
    free(fused_output);
  }
}

static void benchmark_qwen_block_entry(const qwen_bench_entry* entry, int runs) {
  const qwen_block_fp_config* config = &entry->config;
  const uint16_t head_dim = config->hidden_size / config->num_attention_heads;
  const uint16_t kv_hidden = config->num_key_value_heads * head_dim;
  const size_t hidden_floats = (size_t)config->seq_len * config->hidden_size;
  const size_t q_proj_weights = (size_t)config->hidden_size * config->hidden_size;
  const size_t kv_proj_weights = (size_t)kv_hidden * config->hidden_size;
  const size_t down_proj_weights = (size_t)config->hidden_size * config->intermediate_size;
  const size_t workspace_floats = qwen_block_fp_get_workspace_floats(config);
  qwen_block_fp_params params;
  float* input_data = (float*)malloc(sizeof(float) * hidden_floats);
  float* output_data = (float*)malloc(sizeof(float) * hidden_floats);
  float* workspace = (float*)malloc(sizeof(float) * workspace_floats);
  uint32_t* position_ids = (uint32_t*)malloc(sizeof(uint32_t) * config->seq_len);
  float* rope_cos = (float*)malloc(sizeof(float) * config->seq_len * config->rotary_dim);
  float* rope_sin = (float*)malloc(sizeof(float) * config->seq_len * config->rotary_dim);
  float* input_norm_weight = (float*)malloc(sizeof(float) * config->hidden_size);
  float* post_attn_norm_weight = (float*)malloc(sizeof(float) * config->hidden_size);
  float* q_proj_weight = (float*)malloc(sizeof(float) * q_proj_weights);
  float* q_proj_bias = (float*)malloc(sizeof(float) * config->hidden_size);
  float* k_proj_weight = (float*)malloc(sizeof(float) * kv_proj_weights);
  float* k_proj_bias = (float*)malloc(sizeof(float) * kv_hidden);
  float* v_proj_weight = (float*)malloc(sizeof(float) * kv_proj_weights);
  float* v_proj_bias = (float*)malloc(sizeof(float) * kv_hidden);
  float* o_proj_weight = (float*)malloc(sizeof(float) * q_proj_weights);
  float* o_proj_bias = (float*)malloc(sizeof(float) * config->hidden_size);
  float* gate_proj_weight = (float*)malloc(sizeof(float) * config->intermediate_size * config->hidden_size);
  float* gate_proj_bias = (float*)malloc(sizeof(float) * config->intermediate_size);
  float* up_proj_weight = (float*)malloc(sizeof(float) * config->intermediate_size * config->hidden_size);
  float* up_proj_bias = (float*)malloc(sizeof(float) * config->intermediate_size);
  float* down_proj_weight = (float*)malloc(sizeof(float) * down_proj_weights);
  float* down_proj_bias = (float*)malloc(sizeof(float) * config->hidden_size);
  tinyengine_status_fp status = STATE_SUCCESS_fp;
  clock_t start_tick;
  clock_t end_tick;
  int run;

  if (input_data == NULL || output_data == NULL || workspace == NULL || position_ids == NULL || rope_cos == NULL ||
      rope_sin == NULL || input_norm_weight == NULL || post_attn_norm_weight == NULL || q_proj_weight == NULL ||
      q_proj_bias == NULL || k_proj_weight == NULL || k_proj_bias == NULL || v_proj_weight == NULL ||
      v_proj_bias == NULL || o_proj_weight == NULL || o_proj_bias == NULL || gate_proj_weight == NULL ||
      gate_proj_bias == NULL || up_proj_weight == NULL || up_proj_bias == NULL || down_proj_weight == NULL ||
      down_proj_bias == NULL) {
      print_csv_row("wrapper", entry->name, config->seq_len, config->hidden_size, config->intermediate_size, 0, 0,
                    config->seq_len, config->num_attention_heads, config->num_key_value_heads, config->rotary_dim,
                    head_dim, 0, 0, DEFAULT_DTYPE_BYTES, runs, qwen_block_fp_get_workspace_bytes(config), 0, 0.0,
                    0.0f, "alloc_fail");
      goto cleanup;
  }

  fill_tensor(input_data, hidden_floats, 0.01f, 0.0f);
  fill_u32(position_ids, config->seq_len, config->seq_len);
  fill_rope_tables(rope_cos, rope_sin, config->seq_len, config->rotary_dim);
  fill_tensor(input_norm_weight, config->hidden_size, 0.0f, 1.0f);
  fill_tensor(post_attn_norm_weight, config->hidden_size, 0.0f, 1.0f);
  fill_tensor(q_proj_weight, q_proj_weights, 0.001f, 0.0f);
  fill_tensor(q_proj_bias, config->hidden_size, 0.0f, 0.0f);
  fill_tensor(k_proj_weight, kv_proj_weights, 0.001f, 0.0f);
  fill_tensor(k_proj_bias, kv_hidden, 0.0f, 0.0f);
  fill_tensor(v_proj_weight, kv_proj_weights, 0.001f, 0.0f);
  fill_tensor(v_proj_bias, kv_hidden, 0.0f, 0.0f);
  fill_tensor(o_proj_weight, q_proj_weights, 0.001f, 0.0f);
  fill_tensor(o_proj_bias, config->hidden_size, 0.0f, 0.0f);
  fill_tensor(gate_proj_weight, (size_t)config->intermediate_size * config->hidden_size, 0.001f, 0.0f);
  fill_tensor(gate_proj_bias, config->intermediate_size, 0.0f, 0.0f);
  fill_tensor(up_proj_weight, (size_t)config->intermediate_size * config->hidden_size, 0.001f, 0.0f);
  fill_tensor(up_proj_bias, config->intermediate_size, 0.0f, 0.0f);
  fill_tensor(down_proj_weight, down_proj_weights, 0.001f, 0.0f);
  fill_tensor(down_proj_bias, config->hidden_size, 0.0f, 0.0f);

  params.config = *config;
  params.position_ids = position_ids;
  params.rope_cos = rope_cos;
  params.rope_sin = rope_sin;
  params.input_norm_weight = input_norm_weight;
  params.post_attn_norm_weight = post_attn_norm_weight;
  params.q_proj_weight = q_proj_weight;
  params.q_proj_bias = q_proj_bias;
  params.k_proj_weight = k_proj_weight;
  params.k_proj_bias = k_proj_bias;
  params.v_proj_weight = v_proj_weight;
  params.v_proj_bias = v_proj_bias;
  params.o_proj_weight = o_proj_weight;
  params.o_proj_bias = o_proj_bias;
  params.gate_proj_weight = gate_proj_weight;
  params.gate_proj_bias = gate_proj_bias;
  params.up_proj_weight = up_proj_weight;
  params.up_proj_bias = up_proj_bias;
  params.down_proj_weight = down_proj_weight;
  params.down_proj_bias = down_proj_bias;

  start_tick = clock();
  for (run = 0; run < runs; ++run) {
    status = qwen_block_fp_run(&params, input_data, output_data, workspace);
  }
  end_tick = clock();

  print_csv_row("wrapper", entry->name, config->seq_len, config->hidden_size, config->intermediate_size, 0, 0,
                config->seq_len, config->num_attention_heads, config->num_key_value_heads, config->rotary_dim,
                head_dim, 0, 0, DEFAULT_DTYPE_BYTES, runs, qwen_block_fp_get_workspace_bytes(config), 0,
                elapsed_ms(start_tick, end_tick), checksum_fp(output_data, hidden_floats),
                status == STATE_SUCCESS_fp ? "ok" : "fail");

cleanup:
  free(input_data);
  free(output_data);
  free(workspace);
  free(position_ids);
  free(rope_cos);
  free(rope_sin);
  free(input_norm_weight);
  free(post_attn_norm_weight);
  free(q_proj_weight);
  free(q_proj_bias);
  free(k_proj_weight);
  free(k_proj_bias);
  free(v_proj_weight);
  free(v_proj_bias);
  free(o_proj_weight);
  free(o_proj_bias);
  free(gate_proj_weight);
  free(gate_proj_bias);
  free(up_proj_weight);
  free(up_proj_bias);
  free(down_proj_weight);
  free(down_proj_bias);
}

int main(void) {
  size_t i;

  print_csv_header();

  benchmark_rmsnorm(DEFAULT_RUNS);
  benchmark_rotary_embedding(DEFAULT_RUNS);
  benchmark_repeat_kv(DEFAULT_RUNS);
  benchmark_silu(DEFAULT_RUNS);
  benchmark_gelu(DEFAULT_RUNS);
  benchmark_qkv_attention(DEFAULT_RUNS);
  benchmark_embedding_lookup(DEFAULT_RUNS);
  benchmark_argmax(DEFAULT_RUNS);
  benchmark_mm_projector(DEFAULT_RUNS);
  benchmark_lm_head_last_token(DEFAULT_RUNS);
  benchmark_kv_cache(DEFAULT_RUNS);
  benchmark_image_text_fusion(DEFAULT_RUNS);

  for (i = 0; i < sizeof(k_qwen_configs) / sizeof(k_qwen_configs[0]); ++i) {
    benchmark_qwen_block_entry(&k_qwen_configs[i], DEFAULT_RUNS);
  }

  return 0;
}
