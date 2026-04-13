import copy
import math
from typing import Optional

import torch
from torch import nn
from transformers import LlamaConfig
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
from transformers.models.llama.modeling_llama import LlamaRMSNorm


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(states: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    return (states * cos) + (rotate_half(states) * sin)


def build_key_mask(
    attention_mask: torch.Tensor,
    window_size: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    min_value = torch.finfo(dtype).min
    key_mask = torch.where(attention_mask == 0, min_value, 0.0)
    key_mask = key_mask.unsqueeze(1).unsqueeze(2)
    return key_mask.expand(-1, 1, window_size, -1).to(device)


class RotaryEmbedding(nn.Module):
    def __init__(self, config: LlamaConfig):
        super().__init__()
        if getattr(config, "rope_scaling", None):
            rope_type = config.rope_scaling.get("rope_type", config.rope_scaling.get("type"))
        else:
            rope_type = "default"
        rope_init_fn = ROPE_INIT_FUNCTIONS[rope_type]
        self.inv_freq, self.attention_scaling = rope_init_fn(config, "cpu")

    @torch.no_grad()
    def forward(self, hidden_states: torch.Tensor, position_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        inv_freq = self.inv_freq.to(position_ids.device)[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        position_ids = position_ids[:, None, :].float()
        device_type = hidden_states.device.type
        device_type = device_type if device_type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq @ position_ids).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling
        return cos.to(dtype=hidden_states.dtype), sin.to(dtype=hidden_states.dtype)


class PrunedLlamaAttention(nn.Module):
    def __init__(self, base_config: LlamaConfig, num_heads: int = 16):
        super().__init__()
        config = copy.deepcopy(base_config)
        config.num_attention_heads = num_heads

        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.attention_bias = config.attention_bias
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim, bias=self.attention_bias)
        self.k_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim, bias=self.attention_bias)
        self.rotary_emb = RotaryEmbedding(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        position_ids: torch.Tensor,
        window_size: int,
    ) -> torch.Tensor:
        hidden_states = self.input_layernorm(hidden_states)
        batch_size, seq_len, _ = hidden_states.shape
        window_size = min(window_size, seq_len)

        query_states = self.q_proj(hidden_states[:, -window_size:])
        key_states = self.k_proj(hidden_states)

        query_states = query_states.view(batch_size, window_size, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        cos, sin = self.rotary_emb(hidden_states, position_ids)
        query_states = apply_rotary_pos_emb(query_states, cos[:, -window_size:], sin[:, -window_size:])
        key_states = apply_rotary_pos_emb(key_states, cos, sin)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3))
        attn_weights = attn_weights / math.sqrt(self.head_dim)

        if attention_mask is not None:
            attn_weights = attn_weights + build_key_mask(
                attention_mask=attention_mask,
                window_size=window_size,
                dtype=query_states.dtype,
                device=query_states.device,
            )

        return torch.softmax(attn_weights, dim=-1).to(query_states.dtype)
