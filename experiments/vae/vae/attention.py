"""Hybrid attention and padding positions shared by the LLaMA encoder and decoder."""
import torch


def additive_mask(allowed: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """bool [B,1,L,L] (True = attend) -> additive float mask (0 / finfo.min), the format `VAEEncoder` uses."""
    return torch.zeros(allowed.shape, dtype=dtype, device=allowed.device).masked_fill(~allowed, torch.finfo(dtype).min)


def build_mask(mode: str, key_mask: torch.Tensor, latent_start: int, num_latents: int, dtype: torch.dtype):
    """key_mask: [B, L] 1 = real token. Returns what to pass as `attention_mask` to the HF model."""
    if mode == "causal":
        return key_mask
    L = key_mask.shape[1]
    m = torch.tril(torch.ones(L, L, dtype=torch.bool, device=key_mask.device))
    m[latent_start:latent_start + num_latents, latent_start:latent_start + num_latents] = True
    m = m[None, None] & key_mask.bool()[:, None, None, :]
    return additive_mask(m, dtype)


def encoder_position_ids(attention_mask: torch.Tensor, num_latents: int) -> torch.Tensor:
    """Pad-skipping positions for [text (right-padded), queries]: text i -> i, query j -> len + j."""
    lengths = attention_mask.sum(1, keepdim=True)
    text_pos = (attention_mask.long().cumsum(1) - 1).clamp(min=0)
    q_pos = lengths + torch.arange(num_latents, device=attention_mask.device)[None]
    return torch.cat([text_pos, q_pos], dim=1)
