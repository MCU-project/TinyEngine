## STM32N6570-DK Porting Preparation

## 1. 목적

현재까지의 `LLaVA/Qwen` profiling framework는 PC/host 환경에서 FP32 operator와 wrapper를 실행하고, CSV 형식으로 latency / workspace / cache 관련 정보를 출력하는 단계까지 완성되었다.

이제 다음 단계의 목표는 전체 `LLaVA-OneVision-Qwen2-0.5B` 모델을 바로 STM32N6570-DK에 올리는 것이 아니라, 먼저 STM32N6570-DK에서 실행 가능한 작은 microbenchmark entry를 분리하여 `LLaVA-style` 연산 경로가 실제 MCU에서 build / flash / run 가능한지 확인하는 것이다.

즉, 이번 단계의 핵심 목적은 다음과 같다.

```text
Host-side profiling framework
  ↓
STM32N6570-DK용 toy microbenchmark 분리
  ↓
operator / wrapper 단위 MCU 실행 확인
  ↓
NAS/KD 축소 모델을 올리기 위한 runtime path 검증

이번 단계에서는 원본 LLaVA 0.5B 전체 forward를 구현하지 않는다.
또한 실제 checkpoint weight loader, quantization, kernel optimization도 아직 고려하지 않는다.

2. 현재 Host-side Framework 상태

현재 host-side profiling framework에는 다음 구성 요소가 이미 포함되어 있다.

1. FP32 LLaVA/Qwen operator benchmark
2. Qwen decoder block wrapper
3. mm_projector wrapper
4. lm_head last-token benchmark
5. KV cache size / append / read benchmark
6. image-text fusion benchmark
7. CSV 출력 형식
8. memory estimator
9. MCU candidate config sweep script

주요 host benchmark driver는 다음 파일이다.

examples/llava_microbench/main.c

이 파일은 host 환경에서 다음 benchmark entry들을 CSV row로 출력한다.

rmsnorm_fp
rotary_embedding_fp
repeat_kv_fp
silu_fp
gelu_fp
qkv_attention_fp
embedding_lookup_fp
argmax_fp
mm_projector_fp
lm_head_last_token_fp
kv_cache_size
kv_cache_append_fp
kv_cache_read_fp
image_text_fusion_fp
qwen_block_fp
3. 이번 단계에서 추가할 STM32용 구조

STM32N6570-DK에서 실행할 수 있는 microbenchmark를 분리하기 위해 다음 파일 구조를 추가한다.

examples/llava_microbench/
  main.c                    # 기존 host용 benchmark driver
  main_stm32n6.c             # STM32N6570-DK용 toy benchmark entry
  benchmark_common.h         # host / STM32 공통 초기화 및 checksum helper
  benchmark_common.c
  benchmark_timer.h          # 공통 timer interface
  benchmark_timer_stm32.c    # STM32용 timer implementation

기존 host용 main.c와 STM32용 main_stm32n6.c에서 중복되는 초기화 코드는 benchmark_common.c/h로 분리한다.

4. STM32용 main_stm32n6.c의 역할

main_stm32n6.c는 전체 LLaVA forward가 아니라, STM32N6570-DK에서 실행 가능한 작은 toy benchmark만 수행한다.

초기 목표는 다음 operator / wrapper가 MCU에서 정상 실행되는지 확인하는 것이다.

1. rmsnorm_fp
2. gelu_fp
3. silu_fp
4. mm_projector_fp
5. image_text_fusion_fp
6. qwen_block_fp
7. lm_head_last_token_fp

각 benchmark는 다음 항목을 출력해야 한다.

- benchmark name
- variant name
- seq_len
- hidden_size
- intermediate_size
- vision_hidden
- num_image_tokens
- text_tokens
- vocab_size
- workspace_bytes
- cycles
- latency_ms
- checksum
- status
5. Toy Config

STM32용 benchmark에서는 원본 LLaVA 0.5B shape를 사용하지 않는다.

처음에는 아래와 같은 작은 toy config만 사용한다.

seq_len = 4
hidden_size = 32
intermediate_size = 64
num_heads = 4
num_kv_heads = 2
rotary_dim = 8
head_dim = 8
vision_hidden = 32
num_image_tokens = 4
text_tokens = 4
vocab_size = 128

이 toy config의 목적은 성능 측정이 아니라 다음을 확인하는 것이다.

1. STM32CubeIDE에서 build 되는가?
2. firmware에 새 source file이 포함되는가?
3. flash 후 hard fault 없이 실행되는가?
4. UART / printf 출력이 가능한가?
5. checksum이 NaN / Inf가 아닌가?
6. latency 측정값이 출력되는가?
7. workspace_bytes가 출력되는가?

toy config가 성공한 뒤에는 config table을 이용해 점진적으로 크기를 키운다.

hidden_size: 32 → 64 → 128 → 256
seq_len: 4 → 8 → 16 → 32
num_image_tokens: 4 → 16 → 32 → 64
vocab_size: 128 → 1024 → 4096
6. Timer Wrapper

STM32N6570-DK에서 latency를 측정하기 위해 공통 timer interface를 추가한다.

benchmark_timer.h

benchmark_timer.h는 host / STM32 양쪽에서 공통으로 사용할 timer API를 정의한다.

예상 API:

void benchmark_timer_init(void);
uint32_t benchmark_timer_now(void);
float benchmark_timer_elapsed_ms(uint32_t start, uint32_t end);
benchmark_timer_stm32.c

STM32용 구현에서는 가능하면 DWT cycle counter를 사용한다.

우선순위:
1. DWT cycle counter
2. 사용 불가능하면 HAL_GetTick fallback

DWT cycle counter를 사용할 수 있으면 cycle 단위 latency를 출력하고, 시스템 클럭 기준으로 ms 단위 latency도 계산한다.

DWT 사용이 어려운 환경에서는 HAL_GetTick() 기반 fallback을 사용한다.

7. CSV 출력 형식

STM32용 benchmark 출력도 host benchmark와 동일하게 CSV 형식을 사용한다.

예상 CSV header:

benchmark,variant,seq_len,hidden_size,intermediate_size,vision_hidden,num_image_tokens,text_tokens,num_heads,num_kv_heads,rotary_dim,head_dim,vocab_size,runs,workspace_bytes,cycles,latency_ms,checksum,status

예상 출력 예시:

benchmark,variant,seq_len,hidden_size,intermediate_size,vision_hidden,num_image_tokens,text_tokens,num_heads,num_kv_heads,rotary_dim,head_dim,vocab_size,runs,workspace_bytes,cycles,latency_ms,checksum,status
operator,rmsnorm_fp,4,32,0,0,0,0,0,0,0,0,0,10,0,12345,0.123,0.9821,ok
wrapper,mm_projector_fp,0,32,0,32,4,0,0,0,0,0,0,10,512,23456,0.234,1.2041,ok
wrapper,qwen_block_fp,4,32,64,0,0,0,4,2,8,8,0,10,5440,98765,0.987,-1.1881,ok
wrapper,lm_head_last_token_fp,1,32,0,0,0,0,0,0,0,0,128,10,0,45678,0.456,2.0312,ok
8. Checksum 및 Sanity Check

각 benchmark는 output tensor에 대해 간단한 checksum을 계산한다.

목적은 정확도 검증이 아니라, MCU에서 실행 중 다음 문제가 발생했는지 빠르게 확인하는 것이다.

1. NaN 발생
2. Inf 발생
3. output이 전부 0으로 고정
4. hard fault 또는 memory corruption
5. host와 완전히 다른 비정상 출력

checksum helper는 benchmark_common.c/h에 둔다.

예상 helper:

float checksum_fp32(const float *data, int length);
int has_nan_or_inf_fp32(const float *data, int length);
9. Memory Allocation Policy

STM32용 benchmark에서는 malloc 사용을 피한다.

모든 buffer는 다음 중 하나로 관리한다.

1. static buffer
2. 전역 workspace buffer
3. 명시적으로 전달되는 scratch buffer

예상 방식:

#define BENCHMARK_WORKSPACE_FLOATS 16384
static float g_workspace[BENCHMARK_WORKSPACE_FLOATS];

static float g_input[...];
static float g_output[...];
static float g_weight[...];

이 방식의 목적은 다음과 같다.

1. MCU에서 heap 사용으로 인한 불안정성 제거
2. peak workspace 크기 추적 용이
3. SRAM 사용량을 명확하게 파악
4. 이후 PSRAM / SRAM 배치 전략 수립 가능
10. STM32 포팅 단계에서 아직 하지 않는 것

이번 단계에서는 다음 항목을 의도적으로 제외한다.

1. 원본 LLaVA 0.5B 전체 forward
2. 실제 LLaVA checkpoint weight loading
3. SigLIP vision tower 전체 포팅
4. NAS/KD 축소 모델의 실제 weight 적용
5. quantization
6. INT8 / INT4 kernel
7. optimized attention kernel
8. optimized GEMM / GEMV kernel
9. end-to-end text generation

이번 단계의 목적은 오직 다음이다.

STM32N6570-DK에서 LLaVA-style operator path가 실행 가능한지 확인한다.
11. STM32용 최소 성공 기준

STM32N6570-DK 포팅 준비 단계의 최소 성공 기준은 다음과 같다.

1. main_stm32n6.c가 STM32CubeIDE project에서 build 됨
2. llm_ops_fp.c, mm_projector_fp.c, qwen_block_fp.c 등이 link 됨
3. flash 후 UART / printf 출력 확인
4. rmsnorm_fp toy benchmark 실행 성공
5. gelu_fp / silu_fp toy benchmark 실행 성공
6. mm_projector_fp toy benchmark 실행 성공
7. image_text_fusion_fp toy benchmark 실행 성공
8. qwen_block_fp toy benchmark 실행 성공
9. lm_head_last_token_fp toy benchmark 실행 성공
10. 각 benchmark에서 checksum과 latency 출력
11. NaN / Inf / hard fault 없음
12. 이후 확장 계획

STM32 toy benchmark가 성공하면 다음 순서로 확장한다.

1. hidden_size / seq_len / vocab_size scaling test
2. qwen_block_fp 내부 stage별 latency split
3. KV cache append/read 실제 MCU latency 측정
4. image token 수 증가에 따른 projector/fusion latency 측정
5. NAS/KD 팀이 제공한 축소 config 반영
6. 축소 모델 weight를 C array 또는 binary blob 형태로 연결
7. TinyEngine runtime path에서 mini LLaVA forward 실행
8. 최종적으로 STM32N6570-DK에서 token 1개 생성 경로 검증
13. 역할 분리

이번 프로젝트에서 NAS/KD 팀과 MCU runtime 담당의 역할은 다음과 같이 나뉜다.

NAS/KD 팀
1. LLaVA 0.5B를 얼마나 줄일지 결정
2. hidden size / layer 수 / image token 수 / vocab size 후보 선정
3. distillation 또는 NAS로 축소 모델 생성
4. 축소 모델 weight 제공
MCU Runtime / TinyEngine 담당
1. LLaVA-style operator path를 TinyEngine에 구현
2. STM32N6570-DK에서 operator 단위 실행 확인
3. wrapper 단위 실행 확인
4. latency / workspace / checksum 출력
5. hard fault, memory overflow, build/link 문제 확인
6. NAS/KD 축소 모델이 실제 MCU에서 실행 가능한지 검증

즉, 현재 단계에서 MCU runtime 담당의 목표는 다음이다.

NAS/KD 팀이 만든 축소 LLaVA-style 모델을
STM32N6570-DK + TinyEngine 위에서 실행할 수 있는 runtime path를 준비한다.
14. Summary

현재 host-side profiling framework는 이미 다음을 지원한다.

- FP32 LLaVA/Qwen operator benchmark
- Qwen block benchmark
- mm_projector benchmark
- lm_head last-token benchmark
- KV cache benchmark
- image-text fusion benchmark
- memory estimator
- MCU candidate sweep

다음 단계는 이 host-side framework를 STM32N6570-DK 포팅 준비 단계로 확장하는 것이다.

이를 위해 main_stm32n6.c, benchmark_common.c/h, benchmark_timer.h, benchmark_timer_stm32.c를 추가하고, 작은 toy config를 이용해 STM32N6570-DK에서 operator / wrapper가 실제로 실행되는지 확인한다.

이 단계는 원본 LLaVA 0.5B 전체를 올리는 단계가 아니다.
목표는 NAS/KD 팀이 만든 축소 LLaVA-style 모델을 나중에 MCU에 올릴 수 있도록, TinyEngine 기반 runtime path가 실제 보드에서 동작하는지 검증하는 것이다.