#!/usr/bin/env python3

import ctypes as C
import math
import subprocess
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
TMP_DIR = ROOT / "tests" / ".tmp"
TMP_DIR.mkdir(exist_ok=True)

ORIG_LIB = TMP_DIR / "libllava_fp_original.dylib"
VERIFIED_LIB = TMP_DIR / "libllava_fp_verified.dylib"

COMMON_SOURCES = [
    "TinyEngine/src/kernels/fp_backward_op/add_fp.c",
    "TinyEngine/src/kernels/fp_backward_op/mul_fp.c",
    "TinyEngine/src/kernels/fp_backward_op/image_text_fusion_fp.c",
    "TinyEngine/src/kernels/fp_backward_op/kv_cache_fp.c",
    "TinyEngine/src/kernels/fp_backward_op/llm_ops_fp.c",
    "TinyEngine/src/kernels/fp_backward_op/lm_head_last_token_fp.c",
    "TinyEngine/src/kernels/fp_backward_op/mm_projector_fp.c",
    "TinyEngine/src/kernels/fp_backward_op/qwen_block_fp.c",
]

VERIFIED_SOURCES = COMMON_SOURCES + [
    "TinyEngine/src/kernels/fp_backward_op/llm_ops_fp_verified.c",
    "TinyEngine/src/kernels/fp_backward_op/mm_projector_fp_verified.c",
    "TinyEngine/src/kernels/fp_backward_op/qwen_block_fp_verified.c",
]


def build_shared_lib(output_path: Path, sources):
    cmd = [
        "clang",
        "-shared",
        "-fPIC",
        "-std=c11",
        "-Wall",
        "-Wextra",
        "-pedantic",
        "-I",
        str(ROOT / "TinyEngine/include"),
        "-lm",
        "-o",
        str(output_path),
    ] + [str(ROOT / src) for src in sources]
    subprocess.run(cmd, check=True, cwd=ROOT)


build_shared_lib(ORIG_LIB, COMMON_SOURCES)
build_shared_lib(VERIFIED_LIB, VERIFIED_SOURCES)


def ptr_f(tensor):
    return tensor.contiguous().view(-1).numpy().ctypes.data_as(C.POINTER(C.c_float))


def ptr_u32(tensor):
    return tensor.contiguous().view(-1).numpy().ctypes.data_as(C.POINTER(C.c_uint32))


def maxdiff(a, b):
    return float((a - b).abs().max().item())


class QwenBlockConfig(C.Structure):
    _fields_ = [
        ("seq_len", C.c_uint16),
        ("hidden_size", C.c_uint16),
        ("intermediate_size", C.c_uint16),
        ("num_attention_heads", C.c_uint16),
        ("num_key_value_heads", C.c_uint16),
        ("rotary_dim", C.c_uint16),
        ("rms_norm_eps", C.c_float),
        ("causal", C.c_bool),
    ]


class QwenBlockParams(C.Structure):
    _fields_ = [
        ("config", QwenBlockConfig),
        ("position_ids", C.POINTER(C.c_uint32)),
        ("rope_cos", C.POINTER(C.c_float)),
        ("rope_sin", C.POINTER(C.c_float)),
        ("input_norm_weight", C.POINTER(C.c_float)),
        ("post_attn_norm_weight", C.POINTER(C.c_float)),
        ("q_proj_weight", C.POINTER(C.c_float)),
        ("q_proj_bias", C.POINTER(C.c_float)),
        ("k_proj_weight", C.POINTER(C.c_float)),
        ("k_proj_bias", C.POINTER(C.c_float)),
        ("v_proj_weight", C.POINTER(C.c_float)),
        ("v_proj_bias", C.POINTER(C.c_float)),
        ("o_proj_weight", C.POINTER(C.c_float)),
        ("o_proj_bias", C.POINTER(C.c_float)),
        ("gate_proj_weight", C.POINTER(C.c_float)),
        ("gate_proj_bias", C.POINTER(C.c_float)),
        ("up_proj_weight", C.POINTER(C.c_float)),
        ("up_proj_bias", C.POINTER(C.c_float)),
        ("down_proj_weight", C.POINTER(C.c_float)),
        ("down_proj_bias", C.POINTER(C.c_float)),
    ]


class MMProjectorConfig(C.Structure):
    _fields_ = [
        ("num_image_tokens", C.c_uint16),
        ("vision_hidden", C.c_uint16),
        ("hidden_size", C.c_uint16),
    ]


class MMProjectorParams(C.Structure):
    _fields_ = [
        ("config", MMProjectorConfig),
        ("fc1_weight", C.POINTER(C.c_float)),
        ("fc1_bias", C.POINTER(C.c_float)),
        ("fc2_weight", C.POINTER(C.c_float)),
        ("fc2_bias", C.POINTER(C.c_float)),
    ]


class LMHeadConfig(C.Structure):
    _fields_ = [("hidden_size", C.c_uint16), ("vocab_size", C.c_uint32)]


class LMHeadParams(C.Structure):
    _fields_ = [
        ("config", LMHeadConfig),
        ("weight", C.POINTER(C.c_float)),
        ("bias", C.POINTER(C.c_float)),
    ]


class KVCacheConfig(C.Structure):
    _fields_ = [
        ("num_layers", C.c_uint16),
        ("max_seq_len", C.c_uint16),
        ("num_kv_heads", C.c_uint16),
        ("head_dim", C.c_uint16),
    ]


class ImageTextFusionConfig(C.Structure):
    _fields_ = [
        ("image_tokens", C.c_uint16),
        ("text_tokens", C.c_uint16),
        ("hidden_size", C.c_uint16),
    ]


def configure_common_symbols(lib):
    lib.rmsnorm_fp.argtypes = [C.POINTER(C.c_float), C.POINTER(C.c_float), C.c_uint16, C.c_uint16, C.c_float, C.POINTER(C.c_float)]
    lib.silu_fp.argtypes = [C.c_uint16, C.POINTER(C.c_float), C.POINTER(C.c_float)]
    lib.gelu_fp.argtypes = [C.c_uint16, C.POINTER(C.c_float), C.POINTER(C.c_float)]
    lib.repeat_kv_fp.argtypes = [C.POINTER(C.c_float), C.c_uint16, C.c_uint16, C.c_uint16, C.c_uint16, C.POINTER(C.c_float)]
    lib.rotary_embedding_fp.argtypes = [C.POINTER(C.c_float), C.POINTER(C.c_float), C.c_uint16, C.c_uint16, C.c_uint16, C.c_uint16, C.POINTER(C.c_uint32), C.POINTER(C.c_float), C.POINTER(C.c_float), C.c_uint16]
    lib.qkv_attention_fp.argtypes = [C.POINTER(C.c_float), C.POINTER(C.c_float), C.POINTER(C.c_float), C.c_uint16, C.c_uint16, C.c_uint16, C.c_uint16, C.c_bool, C.POINTER(C.c_float), C.POINTER(C.c_float)]
    lib.embedding_lookup_fp.argtypes = [C.c_uint16, C.c_uint16, C.POINTER(C.c_uint32), C.POINTER(C.c_float), C.POINTER(C.c_float)]
    lib.argmax_fp.argtypes = [C.c_uint16, C.POINTER(C.c_float), C.POINTER(C.c_uint16), C.POINTER(C.c_float)]
    lib.mm_projector_fp_get_workspace_floats.argtypes = [C.POINTER(MMProjectorConfig)]
    lib.mm_projector_fp_get_workspace_floats.restype = C.c_size_t
    lib.mm_projector_fp_run.argtypes = [C.POINTER(MMProjectorParams), C.POINTER(C.c_float), C.POINTER(C.c_float), C.POINTER(C.c_float)]
    lib.lm_head_last_token_fp_run.argtypes = [C.POINTER(LMHeadParams), C.POINTER(C.c_float), C.POINTER(C.c_float), C.POINTER(C.c_float)]
    lib.kv_cache_append_fp.argtypes = [C.POINTER(KVCacheConfig), C.POINTER(C.c_float), C.POINTER(C.c_float), C.c_uint16, C.c_uint16, C.POINTER(C.c_float), C.POINTER(C.c_float)]
    lib.kv_cache_read_fp.argtypes = [C.POINTER(KVCacheConfig), C.POINTER(C.c_float), C.POINTER(C.c_float), C.c_uint16, C.c_uint16, C.POINTER(C.c_float), C.POINTER(C.c_float)]
    lib.image_text_fusion_fp_run.argtypes = [C.POINTER(ImageTextFusionConfig), C.POINTER(C.c_float), C.POINTER(C.c_float), C.POINTER(C.c_float), C.POINTER(C.c_float)]
    lib.qwen_block_fp_get_workspace_floats.argtypes = [C.POINTER(QwenBlockConfig)]
    lib.qwen_block_fp_get_workspace_floats.restype = C.c_size_t
    lib.qwen_block_fp_run.argtypes = [C.POINTER(QwenBlockParams), C.POINTER(C.c_float), C.POINTER(C.c_float), C.POINTER(C.c_float)]


def configure_verified_symbols(lib):
    lib.gelu_exact_fp_verified.argtypes = [C.c_size_t, C.POINTER(C.c_float), C.POINTER(C.c_float)]
    lib.rotary_embedding_qwen_fp.argtypes = [C.POINTER(C.c_float), C.POINTER(C.c_float), C.c_uint16, C.c_uint16, C.c_uint16, C.c_uint16, C.POINTER(C.c_uint32), C.POINTER(C.c_float), C.POINTER(C.c_float), C.c_uint16]
    lib.mm_projector_fp_verified_get_workspace_floats.argtypes = [C.POINTER(MMProjectorConfig)]
    lib.mm_projector_fp_verified_get_workspace_floats.restype = C.c_size_t
    lib.mm_projector_fp_verified_run.argtypes = [C.POINTER(MMProjectorParams), C.POINTER(C.c_float), C.POINTER(C.c_float), C.POINTER(C.c_float)]
    lib.mm_projector_fp_verified_run_checked.argtypes = [C.POINTER(MMProjectorParams), C.POINTER(C.c_float), C.POINTER(C.c_float), C.POINTER(C.c_float), C.c_size_t, C.c_bool]
    lib.qwen_block_fp_verified_get_workspace_floats.argtypes = [C.POINTER(QwenBlockConfig)]
    lib.qwen_block_fp_verified_get_workspace_floats.restype = C.c_size_t
    lib.qwen_block_fp_verified_run.argtypes = [C.POINTER(QwenBlockParams), C.POINTER(C.c_float), C.POINTER(C.c_float), C.POINTER(C.c_float)]
    lib.qwen_block_fp_verified_run_checked.argtypes = [C.POINTER(QwenBlockParams), C.POINTER(C.c_float), C.POINTER(C.c_float), C.POINTER(C.c_float), C.c_size_t, C.c_uint32]


orig = C.CDLL(str(ORIG_LIB))
verified = C.CDLL(str(VERIFIED_LIB))
configure_common_symbols(orig)
configure_common_symbols(verified)
configure_verified_symbols(verified)


def rmsnorm_ref(x, w, eps):
    return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps) * w


def linear_ref(x, w, b=None):
    y = x @ w.t()
    return y + (b if b is not None else 0)


def repeat_kv_ref(x, n_rep):
    return x.repeat_interleave(n_rep, dim=1)


def rope_qwen_ref(x, cos, sin, rotary_dim):
    out = x.clone()
    half = rotary_dim // 2
    x1 = x[..., :half]
    x2 = x[..., half:rotary_dim]
    cos_half = cos[..., :half]
    sin_half = sin[..., :half]
    out[..., :half] = x1 * cos_half - x2 * sin_half
    out[..., half:rotary_dim] = x2 * cos_half + x1 * sin_half
    return out


def attention_ref(q, k, v, causal):
    scores = (q.permute(1, 0, 2) @ k.permute(1, 2, 0)) / math.sqrt(q.shape[-1])
    if causal:
      mask = torch.triu(torch.ones_like(scores, dtype=torch.bool), diagonal=1)
      scores = scores.masked_fill(mask, float("-inf"))
    probs = torch.softmax(scores, dim=-1)
    return (probs @ v.permute(1, 0, 2)).permute(1, 0, 2)


def make_u32_positions(seq_len):
    pos64 = torch.arange(seq_len, dtype=torch.int64)
    pos32 = pos64.to(torch.int32).to(torch.uint32)
    return pos64, pos32


def make_rope_tables(seq_len, rotary_dim):
    if rotary_dim == 0:
        return torch.empty(seq_len, 0), torch.empty(seq_len, 0)
    half = rotary_dim // 2
    cos_half = torch.randn(seq_len, half)
    sin_half = torch.randn(seq_len, half)
    return torch.cat([cos_half, cos_half], dim=-1).contiguous(), torch.cat([sin_half, sin_half], dim=-1).contiguous()


def build_qwen_params(cfg, weights, pos32, cos, sin):
    return QwenBlockParams(
        cfg,
        ptr_u32(pos32),
        ptr_f(cos),
        ptr_f(sin),
        ptr_f(weights["input_norm_weight"]),
        ptr_f(weights["post_attn_norm_weight"]),
        ptr_f(weights["q_proj_weight"]),
        ptr_f(weights["q_proj_bias"]),
        ptr_f(weights["k_proj_weight"]),
        ptr_f(weights["k_proj_bias"]),
        ptr_f(weights["v_proj_weight"]),
        ptr_f(weights["v_proj_bias"]),
        ptr_f(weights["o_proj_weight"]),
        ptr_f(weights["o_proj_bias"]),
        ptr_f(weights["gate_proj_weight"]),
        ptr_f(weights["gate_proj_bias"]),
        ptr_f(weights["up_proj_weight"]),
        ptr_f(weights["up_proj_bias"]),
        ptr_f(weights["down_proj_weight"]),
        ptr_f(weights["down_proj_bias"]),
    )


def qwen_block_ref(x, cfg, weights, pos64, cos, sin):
    head_dim = cfg.hidden_size // cfg.num_attention_heads
    kv_hidden = cfg.num_key_value_heads * head_dim

    n1 = rmsnorm_ref(x, weights["input_norm_weight"], cfg.rms_norm_eps)
    q = linear_ref(n1, weights["q_proj_weight"], weights["q_proj_bias"]).view(cfg.seq_len, cfg.num_attention_heads, head_dim)
    k = linear_ref(n1, weights["k_proj_weight"], weights["k_proj_bias"]).view(cfg.seq_len, cfg.num_key_value_heads, head_dim)
    v = linear_ref(n1, weights["v_proj_weight"], weights["v_proj_bias"]).view(cfg.seq_len, cfg.num_key_value_heads, head_dim)

    if cfg.rotary_dim > 0:
        rope_cos = cos[pos64].view(cfg.seq_len, 1, cfg.rotary_dim)
        rope_sin = sin[pos64].view(cfg.seq_len, 1, cfg.rotary_dim)
        q = rope_qwen_ref(q, rope_cos, rope_sin, cfg.rotary_dim)
        k = rope_qwen_ref(k, rope_cos, rope_sin, cfg.rotary_dim)

    k_rep = repeat_kv_ref(k, cfg.num_attention_heads // cfg.num_key_value_heads)
    v_rep = repeat_kv_ref(v, cfg.num_attention_heads // cfg.num_key_value_heads)
    attn = attention_ref(q, k_rep, v_rep, bool(cfg.causal)).reshape(cfg.seq_len, cfg.hidden_size)
    attn_res = linear_ref(attn, weights["o_proj_weight"], weights["o_proj_bias"]) + x
    n2 = rmsnorm_ref(attn_res, weights["post_attn_norm_weight"], cfg.rms_norm_eps)
    gate = torch.nn.functional.silu(linear_ref(n2, weights["gate_proj_weight"], weights["gate_proj_bias"]))
    up = linear_ref(n2, weights["up_proj_weight"], weights["up_proj_bias"])
    mlp = linear_ref(gate * up, weights["down_proj_weight"], weights["down_proj_bias"])
    return attn_res + mlp


def qwen_weight_pack(cfg):
    head_dim = cfg.hidden_size // cfg.num_attention_heads
    kv_hidden = cfg.num_key_value_heads * head_dim
    shapes = {
        "input_norm_weight": (cfg.hidden_size,),
        "post_attn_norm_weight": (cfg.hidden_size,),
        "q_proj_weight": (cfg.hidden_size, cfg.hidden_size),
        "q_proj_bias": (cfg.hidden_size,),
        "k_proj_weight": (kv_hidden, cfg.hidden_size),
        "k_proj_bias": (kv_hidden,),
        "v_proj_weight": (kv_hidden, cfg.hidden_size),
        "v_proj_bias": (kv_hidden,),
        "o_proj_weight": (cfg.hidden_size, cfg.hidden_size),
        "o_proj_bias": (cfg.hidden_size,),
        "gate_proj_weight": (cfg.intermediate_size, cfg.hidden_size),
        "gate_proj_bias": (cfg.intermediate_size,),
        "up_proj_weight": (cfg.intermediate_size, cfg.hidden_size),
        "up_proj_bias": (cfg.intermediate_size,),
        "down_proj_weight": (cfg.hidden_size, cfg.intermediate_size),
        "down_proj_bias": (cfg.hidden_size,),
    }
    return {name: torch.randn(*shape) for name, shape in shapes.items()}


def check_close(name, diff, tol=1e-4):
    print(f"{name}: max_abs_diff={diff:.8f}")
    if diff > tol:
        raise AssertionError(f"{name} exceeded tolerance {tol}: got {diff}")


def check_expected_large(name, diff, min_diff):
    print(f"{name}: max_abs_diff={diff:.8f}")
    if diff < min_diff:
        raise AssertionError(f"{name} was expected to fail by at least {min_diff}, got {diff}")


torch.manual_seed(0)

# Original implementation sanity checks that should pass.
x = torch.randn(5, 7)
w = torch.randn(7)
out = torch.empty_like(x)
orig.rmsnorm_fp(ptr_f(x), ptr_f(w), 5, 7, 1e-6, ptr_f(out))
check_close("original.rmsnorm_fp", maxdiff(out, rmsnorm_ref(x, w, 1e-6)), 1e-5)

v = torch.randn(31)
out = torch.empty_like(v)
orig.silu_fp(v.numel(), ptr_f(v), ptr_f(out))
check_close("original.silu_fp", maxdiff(out, torch.nn.functional.silu(v)), 1e-6)

inp = torch.randn(4, 2, 6)
out = torch.empty(4, 6, 6)
orig.repeat_kv_fp(ptr_f(inp), 4, 2, 6, 3, ptr_f(out))
check_close("original.repeat_kv_fp", maxdiff(out, repeat_kv_ref(inp, 3)), 1e-6)

q = torch.randn(5, 3, 4)
k = torch.randn(5, 3, 4)
v = torch.randn(5, 3, 4)
out = torch.empty_like(q)
scores = torch.empty(5, 5)
orig.qkv_attention_fp(ptr_f(q), ptr_f(k), ptr_f(v), 5, 5, 3, 4, True, ptr_f(out), ptr_f(scores))
check_close("original.qkv_attention_fp", maxdiff(out, attention_ref(q, k, v, True)), 1e-5)

ids = torch.tensor([3, 1, 12, 0], dtype=torch.int64)
ids_u32 = ids.to(torch.int32).to(torch.uint32)
emb = torch.randn(13, 11)
out = torch.empty(4, 11)
orig.embedding_lookup_fp(4, 11, ptr_u32(ids_u32), ptr_f(emb), ptr_f(out))
check_close("original.embedding_lookup_fp", maxdiff(out, emb[ids]), 0.0)

arr = torch.randn(19)
idx = C.c_uint16()
val = C.c_float()
orig.argmax_fp(arr.numel(), ptr_f(arr), C.byref(idx), C.byref(val))
check_close("original.argmax_fp.index", 0.0 if idx.value == int(arr.argmax()) else 1.0, 0.0)
check_close("original.argmax_fp.value", abs(val.value - float(arr.max())), 1e-6)

cfg_lm = LMHeadConfig(9, 17)
last = torch.randn(cfg_lm.hidden_size)
W = torch.randn(cfg_lm.vocab_size, cfg_lm.hidden_size)
b = torch.randn(cfg_lm.vocab_size)
params_lm = LMHeadParams(cfg_lm, ptr_f(W), ptr_f(b))
out = torch.empty(cfg_lm.vocab_size)
orig.lm_head_last_token_fp_run(C.byref(params_lm), ptr_f(last), ptr_f(out), C.POINTER(C.c_float)())
check_close("original.lm_head_last_token_fp", maxdiff(out, linear_ref(last.unsqueeze(0), W, b).squeeze(0)), 1e-5)

cfg_kv = KVCacheConfig(2, 5, 3, 4)
stride = cfg_kv.max_seq_len * cfg_kv.num_kv_heads * cfg_kv.head_dim
k_cache = torch.zeros(cfg_kv.num_layers * stride)
v_cache = torch.zeros_like(k_cache)
exp_k = []
exp_v = []
for pos in range(4):
    token_k = torch.randn(cfg_kv.num_kv_heads * cfg_kv.head_dim)
    token_v = torch.randn(cfg_kv.num_kv_heads * cfg_kv.head_dim)
    exp_k.append(token_k)
    exp_v.append(token_v)
    orig.kv_cache_append_fp(C.byref(cfg_kv), ptr_f(k_cache), ptr_f(v_cache), 1, pos, ptr_f(token_k), ptr_f(token_v))
out_k = torch.empty(4 * cfg_kv.num_kv_heads * cfg_kv.head_dim)
out_v = torch.empty_like(out_k)
orig.kv_cache_read_fp(C.byref(cfg_kv), ptr_f(k_cache), ptr_f(v_cache), 1, 4, ptr_f(out_k), ptr_f(out_v))
check_close("original.kv_cache.key", maxdiff(out_k, torch.cat(exp_k)), 0.0)
check_close("original.kv_cache.value", maxdiff(out_v, torch.cat(exp_v)), 0.0)

cfg_fuse = ImageTextFusionConfig(3, 2, 5)
img = torch.randn(cfg_fuse.image_tokens, cfg_fuse.hidden_size)
txt = torch.randn(cfg_fuse.text_tokens, cfg_fuse.hidden_size)
out = torch.empty(cfg_fuse.image_tokens + cfg_fuse.text_tokens, cfg_fuse.hidden_size)
orig.image_text_fusion_fp_run(C.byref(cfg_fuse), ptr_f(img), ptr_f(txt), ptr_f(out), C.POINTER(C.c_float)())
check_close("original.image_text_fusion_fp", maxdiff(out, torch.cat([img, txt], dim=0)), 0.0)

# Known original mismatches.
seq_len = 4
num_q_heads = 4
num_k_heads = 2
head_dim = 8
rotary_dim = 8
q = torch.randn(seq_len, num_q_heads, head_dim)
k = torch.randn(seq_len, num_k_heads, head_dim)
pos64, pos32 = make_u32_positions(seq_len)
cos, sin = make_rope_tables(seq_len, rotary_dim)
q_out = q.clone()
k_out = k.clone()
orig.rotary_embedding_fp(ptr_f(q_out), ptr_f(k_out), seq_len, num_q_heads, num_k_heads, head_dim, ptr_u32(pos32), ptr_f(cos), ptr_f(sin), rotary_dim)
rope_ref = rope_qwen_ref(q, cos[pos64].view(seq_len, 1, rotary_dim), sin[pos64].view(seq_len, 1, rotary_dim), rotary_dim)
check_expected_large("original.rotary_embedding_fp_vs_qwen_torch", maxdiff(q_out, rope_ref), 1.0)

cfg_mm_large = MMProjectorConfig(257, 4, 257)
fc1_in = torch.randn(cfg_mm_large.num_image_tokens, cfg_mm_large.vision_hidden)
fc1_w = torch.randn(cfg_mm_large.hidden_size, cfg_mm_large.vision_hidden)
fc1_b = torch.randn(cfg_mm_large.hidden_size)
fc2_w = torch.randn(cfg_mm_large.hidden_size, cfg_mm_large.hidden_size)
fc2_b = torch.randn(cfg_mm_large.hidden_size)
mm_params = MMProjectorParams(cfg_mm_large, ptr_f(fc1_w), ptr_f(fc1_b), ptr_f(fc2_w), ptr_f(fc2_b))
mm_out = torch.empty(cfg_mm_large.num_image_tokens, cfg_mm_large.hidden_size)
mm_ws = torch.empty(orig.mm_projector_fp_get_workspace_floats(C.byref(cfg_mm_large)))
orig.mm_projector_fp_run(C.byref(mm_params), ptr_f(fc1_in), ptr_f(mm_out), ptr_f(mm_ws))
mm_ref = linear_ref(torch.nn.functional.gelu(linear_ref(fc1_in, fc1_w, fc1_b)), fc2_w, fc2_b)
check_expected_large("original.mm_projector_fp_large_vs_torch", maxdiff(mm_out, mm_ref), 1e-2)

cfg_q_orig = QwenBlockConfig(8, 64, 96, 4, 1, 16, 1e-6, True)
weights = qwen_weight_pack(cfg_q_orig)
x = torch.randn(cfg_q_orig.seq_len, cfg_q_orig.hidden_size)
pos64, pos32 = make_u32_positions(cfg_q_orig.seq_len)
cos, sin = make_rope_tables(cfg_q_orig.seq_len, cfg_q_orig.rotary_dim)
params = build_qwen_params(cfg_q_orig, weights, pos32, cos, sin)
q_out = torch.empty_like(x)
ws = torch.empty(orig.qwen_block_fp_get_workspace_floats(C.byref(cfg_q_orig)))
orig.qwen_block_fp_run(C.byref(params), ptr_f(x), ptr_f(q_out), ptr_f(ws))
q_ref = qwen_block_ref(x, cfg_q_orig, weights, pos64, cos, sin)
check_expected_large("original.qwen_block_fp_vs_qwen_torch", maxdiff(q_out, q_ref), 1.0)

# Verified operators and wrappers.
v = torch.randn(4097)
out = torch.empty_like(v)
verified.gelu_exact_fp_verified(v.numel(), ptr_f(v), ptr_f(out))
check_close("verified.gelu_exact_fp_verified", maxdiff(out, torch.nn.functional.gelu(v)), 1e-6)

q = torch.randn(seq_len, num_q_heads, head_dim)
k = torch.randn(seq_len, num_k_heads, head_dim)
rotary_pos64, rotary_pos32 = make_u32_positions(seq_len)
rotary_cos, rotary_sin = make_rope_tables(seq_len, rotary_dim)
q_out = q.clone()
k_out = k.clone()
verified.rotary_embedding_qwen_fp(ptr_f(q_out), ptr_f(k_out), seq_len, num_q_heads, num_k_heads, head_dim, ptr_u32(rotary_pos32), ptr_f(rotary_cos), ptr_f(rotary_sin), rotary_dim)
check_close("verified.rotary_embedding_qwen_fp", maxdiff(q_out, rope_qwen_ref(q, rotary_cos[rotary_pos64].view(seq_len, 1, rotary_dim), rotary_sin[rotary_pos64].view(seq_len, 1, rotary_dim), rotary_dim)), 1e-5)

cfg_mm_small = MMProjectorConfig(6, 5, 7)
fc1_in = torch.randn(cfg_mm_small.num_image_tokens, cfg_mm_small.vision_hidden)
fc1_w = torch.randn(cfg_mm_small.hidden_size, cfg_mm_small.vision_hidden)
fc1_b = torch.randn(cfg_mm_small.hidden_size)
fc2_w = torch.randn(cfg_mm_small.hidden_size, cfg_mm_small.hidden_size)
fc2_b = torch.randn(cfg_mm_small.hidden_size)
mm_params = MMProjectorParams(cfg_mm_small, ptr_f(fc1_w), ptr_f(fc1_b), ptr_f(fc2_w), ptr_f(fc2_b))
mm_out = torch.empty(cfg_mm_small.num_image_tokens, cfg_mm_small.hidden_size)
mm_ws = torch.empty(verified.mm_projector_fp_verified_get_workspace_floats(C.byref(cfg_mm_small)))
verified.mm_projector_fp_verified_run(C.byref(mm_params), ptr_f(fc1_in), ptr_f(mm_out), ptr_f(mm_ws))
mm_ref = linear_ref(torch.nn.functional.gelu(linear_ref(fc1_in, fc1_w, fc1_b)), fc2_w, fc2_b)
check_close("verified.mm_projector_fp_verified.small", maxdiff(mm_out, mm_ref), 1e-5)

status = verified.mm_projector_fp_verified_run_checked(C.byref(mm_params), ptr_f(fc1_in), ptr_f(mm_out), ptr_f(mm_ws), 1, True)
check_close("verified.mm_projector_fp_verified.small_workspace_check", 0.0 if status == 1 else 1.0, 0.0)

cfg_mm_large = MMProjectorConfig(257, 4, 257)
fc1_in = torch.randn(cfg_mm_large.num_image_tokens, cfg_mm_large.vision_hidden)
fc1_w = torch.randn(cfg_mm_large.hidden_size, cfg_mm_large.vision_hidden)
fc1_b = torch.randn(cfg_mm_large.hidden_size)
fc2_w = torch.randn(cfg_mm_large.hidden_size, cfg_mm_large.hidden_size)
fc2_b = torch.randn(cfg_mm_large.hidden_size)
mm_params = MMProjectorParams(cfg_mm_large, ptr_f(fc1_w), ptr_f(fc1_b), ptr_f(fc2_w), ptr_f(fc2_b))
mm_out = torch.empty(cfg_mm_large.num_image_tokens, cfg_mm_large.hidden_size)
mm_ws = torch.empty(verified.mm_projector_fp_verified_get_workspace_floats(C.byref(cfg_mm_large)))
verified.mm_projector_fp_verified_run(C.byref(mm_params), ptr_f(fc1_in), ptr_f(mm_out), ptr_f(mm_ws))
mm_ref = linear_ref(torch.nn.functional.gelu(linear_ref(fc1_in, fc1_w, fc1_b)), fc2_w, fc2_b)
check_close("verified.mm_projector_fp_verified.large", maxdiff(mm_out, mm_ref), 1e-4)

cfg_q_verified = QwenBlockConfig(8, 64, 96, 4, 1, 16, 1e-6, True)
weights = qwen_weight_pack(cfg_q_verified)
x = torch.randn(cfg_q_verified.seq_len, cfg_q_verified.hidden_size)
pos64, pos32 = make_u32_positions(cfg_q_verified.seq_len)
cos, sin = make_rope_tables(cfg_q_verified.seq_len, cfg_q_verified.rotary_dim)
params = build_qwen_params(cfg_q_verified, weights, pos32, cos, sin)
q_out = torch.empty_like(x)
ws = torch.empty(verified.qwen_block_fp_verified_get_workspace_floats(C.byref(cfg_q_verified)))
verified.qwen_block_fp_verified_run(C.byref(params), ptr_f(x), ptr_f(q_out), ptr_f(ws))
q_ref = qwen_block_ref(x, cfg_q_verified, weights, pos64, cos, sin)
check_close("verified.qwen_block_fp_verified.qwen_rope", maxdiff(q_out, q_ref), 1e-3)

status = verified.qwen_block_fp_verified_run_checked(C.byref(params), ptr_f(x), ptr_f(q_out), ptr_f(ws), ws.numel(), 1)
check_close("verified.qwen_block_fp_verified.rope_bounds_check", 0.0 if status == 1 else 1.0, 0.0)

cfg_q_large = QwenBlockConfig(257, 256, 256, 4, 1, 0, 1e-6, True)
weights = qwen_weight_pack(cfg_q_large)
weights = {name: tensor * 0.05 for name, tensor in weights.items()}
x = torch.randn(cfg_q_large.seq_len, cfg_q_large.hidden_size) * 0.05
pos64, pos32 = make_u32_positions(cfg_q_large.seq_len)
cos = torch.empty(cfg_q_large.seq_len, 0)
sin = torch.empty(cfg_q_large.seq_len, 0)
params = build_qwen_params(cfg_q_large, weights, pos32, cos, sin)
q_out = torch.empty_like(x)
ws = torch.empty(verified.qwen_block_fp_verified_get_workspace_floats(C.byref(cfg_q_large)))
verified.qwen_block_fp_verified_run(C.byref(params), ptr_f(x), ptr_f(q_out), ptr_f(ws))
q_ref = qwen_block_ref(x, cfg_q_large, weights, pos64, cos, sin)
check_close("verified.qwen_block_fp_verified.large_no_rope", maxdiff(q_out, q_ref), 1e-3)

print("ALL VERIFIED CHECKS PASSED")
