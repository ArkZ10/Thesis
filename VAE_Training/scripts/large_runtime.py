"""Shared explicit encoder/decoder placement for large VAE experiments."""
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM,AutoTokenizer
from vae.model import VAEConfig
from vae.data import VAECollator
from vae.split_training import SplitReasoningVAE

ROOT=Path(__file__).resolve().parents[1]


def build_model(config):
    vc=VAEConfig(**config['model'])
    tokenizer=AutoTokenizer.from_pretrained(vc.encoder_name,local_files_only=True)
    if tokenizer.pad_token_id is None:tokenizer.pad_token=tokenizer.eos_token
    encoder=AutoModelForCausalLM.from_pretrained(vc.encoder_name,dtype=torch.bfloat16,
        attn_implementation='sdpa',device_map={'':'cuda:1'},local_files_only=True)
    decoder=AutoModelForCausalLM.from_pretrained(vc.decoder_name,dtype=torch.bfloat16,
        attn_implementation='sdpa',device_map={'':'cuda:0'},local_files_only=True)
    model=SplitReasoningVAE(vc,encoder,decoder)
    model.fc_mu.to('cuda:1');model.fc_var.to('cuda:1')
    model.latents_query.data=model.latents_query.data.to('cuda:1')
    model.latent_to_decoder.to('cuda:0')
    model.ae_speical_token.data=model.ae_speical_token.data.to('cuda:0')
    assert all(not p.requires_grad for p in decoder.parameters())
    return model,tokenizer


def batch_for(rows,tokenizer,device='cuda:1'):
    batch=VAECollator(tokenizer.pad_token_id)(rows)
    return {k:v.to(device) for k,v in batch.items()}
