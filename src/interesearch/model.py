"""The target model class, vendored with torch and pyyaml as the only dependencies.

Source: goodfire-ai/param-decomp, tag `vpd-paper` (commit 74146b555f3eb3cd2bc639c3789fd3b7a099094a),
file `param_decomp/pretrain/models/llama_simple_mlp.py` (MIT licence, Copyright (c) 2025 Goodfire, see NOTICE).
Changes: removed the param_decomp/jaxtyping imports, the optimizer and generation helpers and the
W&B loading path. The config is a dataclass read from the run's `model_config.yaml`. The forward computation is
copied unchanged (RoPE with `rotate_every_two`, manual causal attention since the target config has
`flash_attention: false`, tanh-GELU, RMSNorm in fp32, tied `wte`/`lm_head`).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, fields
from pathlib import Path

import torch
import torch.nn as nn
import yaml
from torch import Tensor
from torch.nn import functional as F


@dataclass
class LlamaSimpleMLPConfig:
    model_type: str = "LlamaSimpleMLP"
    block_size: int = 1024
    vocab_size: int = 50257
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    n_intermediate: int = 768 * 4
    mlp_bias: bool = False
    attn_bias: bool = False
    rotary_adjacent_pairs: bool = False
    rotary_dim: int = 768 // 12
    rotary_base: int = 10000
    n_ctx: int = 1024
    n_key_value_heads: int = 12 // 4
    use_grouped_query_attention: bool = True
    flash_attention: bool = True
    rms_norm_eps: float = 1e-6

    @classmethod
    def from_yaml(cls, path: str | Path) -> LlamaSimpleMLPConfig:
        d = yaml.safe_load(Path(path).read_text())
        known = {f.name for f in fields(cls)}
        unknown = set(d) - known
        if unknown:
            raise ValueError(f"unknown model config keys: {sorted(unknown)}")
        return cls(**d)


class CausalSelfAttention(nn.Module):
    def __init__(self, config: LlamaSimpleMLPConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.use_grouped_query_attention = config.use_grouped_query_attention
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head
        self.n_key_value_heads = config.n_key_value_heads
        self.repeat_kv_heads = config.n_head // config.n_key_value_heads
        self.rotary_dim = self.head_dim  # as in the original: config.rotary_dim is not used
        self.rotary_adjacent_pairs = config.rotary_adjacent_pairs
        self.rotary_base = config.rotary_base
        self.n_ctx = config.n_ctx
        self.flash_attention = config.flash_attention
        if self.use_grouped_query_attention:
            self.q_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.attn_bias)
            self.k_proj = nn.Linear(config.n_embd, self.n_key_value_heads * self.head_dim, bias=config.attn_bias)
            self.v_proj = nn.Linear(config.n_embd, self.n_key_value_heads * self.head_dim, bias=config.attn_bias)
        else:
            self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.attn_bias)
        self.o_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.attn_bias)

        self.register_buffer(
            "bias",
            torch.tril(torch.ones(config.block_size, config.block_size)).view(1, 1, config.block_size, config.block_size),
            persistent=False,
        )
        sin, cos = self.calculate_sin_cos_rotary(self.rotary_dim, self.n_ctx, base=self.rotary_base)
        self.register_buffer("rotary_sin", sin, persistent=False)
        self.register_buffer("rotary_cos", cos, persistent=False)

    def calculate_sin_cos_rotary(
        self, rotary_dim: int, n_ctx: int, base: int = 10000, dtype: torch.dtype = torch.float32
    ) -> tuple[Tensor, Tensor]:
        high_precision = torch.float32 if dtype != torch.float64 else torch.float64
        pos = torch.arange(n_ctx, dtype=high_precision)
        dim = torch.arange(rotary_dim // 2, dtype=high_precision)
        freq = base ** (dim / (rotary_dim / 2))
        if self.rotary_adjacent_pairs:
            freq = freq.unsqueeze(1).repeat(1, 2).flatten()
        else:
            freq = freq.repeat(2)
        angles = pos[:, None] / freq[None, :]
        return torch.sin(angles).to(dtype), torch.cos(angles).to(dtype)

    def rotate_every_two(self, x: Tensor) -> Tensor:
        x_rot = x.clone()
        if self.rotary_adjacent_pairs:
            x_rot[..., ::2] = -x[..., 1::2]
            x_rot[..., 1::2] = x[..., ::2]
        else:
            n = x.shape[-1] // 2
            x_rot[..., :n] = -x[..., n:]
            x_rot[..., n:] = x[..., :n]
        return x_rot

    def apply_rotary_pos_emb(self, q: Tensor, k: Tensor, cos: Tensor, sin: Tensor) -> tuple[Tensor, Tensor]:
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)
        q_rot = q[..., : self.rotary_dim]
        k_rot = k[..., : self.rotary_dim]
        q_rotated = (q_rot * cos) + (self.rotate_every_two(q_rot) * sin)
        k_rotated = (k_rot * cos) + (self.rotate_every_two(k_rot) * sin)
        if self.rotary_dim < self.head_dim:
            q_embed = torch.cat((q_rotated, q[..., self.rotary_dim :]), dim=-1)
            k_embed = torch.cat((k_rotated, k[..., self.rotary_dim :]), dim=-1)
        else:
            q_embed, k_embed = q_rotated, k_rotated
        return q_embed.to(q.dtype), k_embed.to(k.dtype)

    def forward(self, x: Tensor) -> Tensor:
        B, T, C = x.size()
        if self.use_grouped_query_attention:
            q = self.q_proj(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
            k = self.k_proj(x).view(B, T, self.n_key_value_heads, self.head_dim).transpose(1, 2)
            v = self.v_proj(x).view(B, T, self.n_key_value_heads, self.head_dim).transpose(1, 2)
        else:
            q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
            q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
            k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
            v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        position_ids = torch.arange(0, T, dtype=torch.long, device=x.device).unsqueeze(0)
        position_ids = position_ids.clamp(max=self.n_ctx - 1)
        cos = self.rotary_cos[position_ids].to(q.dtype)
        sin = self.rotary_sin[position_ids].to(q.dtype)
        q, k = self.apply_rotary_pos_emb(q, k, cos, sin)

        if self.use_grouped_query_attention and self.repeat_kv_heads > 1:
            k = k.repeat_interleave(self.repeat_kv_heads, dim=1)
            v = v.repeat_interleave(self.repeat_kv_heads, dim=1)

        if self.flash_attention:
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=True)
        else:
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
            att = F.softmax(att, dim=-1)
            y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.o_proj(y)


class NewGELU(nn.Module):
    def forward(self, input: Tensor) -> Tensor:
        return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))


class MLP(nn.Module):
    def __init__(self, config: LlamaSimpleMLPConfig):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, config.n_intermediate, bias=config.mlp_bias)
        self.gelu = NewGELU()
        self.down_proj = nn.Linear(config.n_intermediate, config.n_embd, bias=config.mlp_bias)

    def forward(self, x: Tensor) -> Tensor:
        return self.down_proj(self.gelu(self.c_fc(x)))


class LlamaRMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: Tensor) -> Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class Block(nn.Module):
    def __init__(self, config: LlamaSimpleMLPConfig):
        super().__init__()
        self.rms_1 = LlamaRMSNorm(config.n_embd, eps=config.rms_norm_eps)
        self.attn = CausalSelfAttention(config)
        self.rms_2 = LlamaRMSNorm(config.n_embd, eps=config.rms_norm_eps)
        self.mlp = MLP(config)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attn(self.rms_1(x))
        x = x + self.mlp(self.rms_2(x))
        return x


class LlamaSimpleMLP(nn.Module):
    """Same module names and state-dict keys as the original (`h.N.attn.q_proj`, `h.N.mlp.c_fc`, ...)."""

    def __init__(self, config: LlamaSimpleMLPConfig):
        super().__init__()
        self.config = config
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.h = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.ln_f = LlamaRMSNorm(config.n_embd, eps=config.rms_norm_eps)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.wte.weight = self.lm_head.weight  # tied, as in the original

    def forward(self, idx: Tensor, targets: Tensor | None = None) -> tuple[Tensor, Tensor | None]:
        """Returns (logits [B, T, vocab], mean CE or None). Same signature as the original's default call."""
        _b, t = idx.size()
        assert t <= self.config.block_size, f"sequence length {t} > block size {self.config.block_size}"
        x = self.wte(idx)
        for block in self.h:
            x = block(x)
        logits = self.lm_head(self.ln_f(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.long().reshape(-1), ignore_index=-1)
        return logits, loss
