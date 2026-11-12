"""Verify the isolated VAE environment without downloading or training a model.

Uses small tensor operations and a tiny, randomly initialized Qwen3 VAE solely
to check installed APIs and gradient propagation. No optimizer updates occur.
"""

import importlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import sys
from datetime import datetime, timezone

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import torch
from transformers import Qwen3Config, Qwen3ForCausalLM
from accelerate import Accelerator, DataLoaderConfiguration

from vae.model import ReasoningVAE, VAEConfig


ROOT = Path(__file__).resolve().parents[1]


def main():
    if sys.prefix == sys.base_prefix:
        raise RuntimeError("Run this check with the workspace virtual environment")
    modules = ["vae.data", "vae.model", "vae.train", "vae.prepare_sentences",
               "vae.validate_sentences", "ladir.checkpoint", "sft.vae_weights", "sft.precompute_latents"]
    for module in modules:
        importlib.import_module(module)
    Accelerator(cpu=True, dataloader_config=DataLoaderConfiguration(use_seedable_sampler=True))
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        raise RuntimeError("Both expected CUDA GPUs must be visible for this check")
    devices = []
    for index in range(2):
        device = f"cuda:{index}"
        with torch.cuda.device(index):
            if not torch.cuda.is_bf16_supported():
                raise RuntimeError(f"BF16 unavailable on {device}")
            left = torch.ones((64, 64), device=device, dtype=torch.bfloat16, requires_grad=True)
            right = torch.ones_like(left, requires_grad=False)
            value = (left @ right).float().mean()
            assert value.item() == 64.0
            value.backward()
            assert left.grad is not None and torch.isfinite(left.grad).all()
            props = torch.cuda.get_device_properties(index)
            devices.append({"index": index, "name": props.name, "memory_bytes": props.total_memory,
                            "bf16_matmul_and_backward": "passed"})
            del left, right, value
            torch.cuda.empty_cache()

    # A tiny offline model tests the exact VAE path: hybrid attention masks,
    # gradient checkpointing, posterior sampling, frozen decoder and FP32 CE.
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    config = Qwen3Config(vocab_size=128, hidden_size=64, intermediate_size=128,
                        num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=2,
                        head_dim=16, max_position_embeddings=128, attention_dropout=0.0,
                        bos_token_id=1, eos_token_id=2, pad_token_id=0)
    config._attn_implementation = "sdpa"
    model = ReasoningVAE(VAEConfig(
        encoder_name="synthetic-offline-qwen3", decoder_name="synthetic-offline-qwen3",
        num_latents=6, latent_dim=16, train_encoder_lm=True, train_decoder_lm=False,
        train_encoder_queries=True, train_encoder_heads=True, train_decoder_adapter=True,
        encoder_mask_mode="hybrid", decoder_mask_mode="hybrid", gradient_checkpointing=True,
    ), Qwen3ForCausalLM(config), Qwen3ForCausalLM(config)).to("cuda:0", dtype=torch.bfloat16)
    model.train()
    ids = torch.randint(3, 128, (2, 12), device="cuda:0")
    mask = torch.ones_like(ids)
    mask[1, -3:] = 0
    ids[1, -3:] = 0
    result = model(ids, mask, kl_weight=1e-5)
    assert torch.isfinite(result["loss"])
    assert result["mu"].shape == (2, 6, 16)
    assert result["n_tokens"] == 21
    result["loss"].backward()
    groups = {
        "encoder": list(model.encoder_model.parameters()),
        "queries": [model.latents_query],
        "posterior_mean": list(model.fc_mu.parameters()),
        "posterior_variance": list(model.fc_var.parameters()),
        "decoder_interface": list(model.latent_to_decoder.parameters()) + [model.ae_speical_token],
    }
    for name, parameters in groups.items():
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in parameters), name
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in parameters), name
    assert all(not p.requires_grad and p.grad is None for p in model.decoder_model.parameters())
    probe = {"passed": True, "device": "cuda:0", "parameters": sum(p.numel() for p in model.parameters()),
             "loss": result["loss"].item(), "ce": result["ce_loss"].item(), "kl": result["kl_loss"].item(),
             "latent_shape": list(result["mu"].shape), "gradient_groups_passed": list(groups),
             "decoder_backbone_frozen": True, "gradient_checkpointing": True, "dtype": "bfloat16",
             "optimizer_steps": 0, "pretrained_weights_downloaded": False}
    report = {
        "checked_at_utc": datetime.now(timezone.utc).isoformat(), "passed": True,
        "python": platform.python_version(), "executable": sys.executable,
        "prefix": sys.prefix, "base_prefix": sys.base_prefix,
        "packages": {name: version(name) for name in ["torch", "transformers", "accelerate", "huggingface-hub",
                      "safetensors", "omegaconf", "datasets", "pyarrow", "laser-vae-training"]},
        "cuda_build": torch.version.cuda, "devices": devices, "imports_passed": modules,
        "synthetic_vae_probe": probe, "dataset_training_started": False,
        "scope": "Environment/API verification only; no pretrained weights, optimizer updates, or benchmark generation.",
        "optional_components_not_installed": ["DeepSpeed", "vLLM", "RL stack"],
    }
    (ROOT / "manifests/environment.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
