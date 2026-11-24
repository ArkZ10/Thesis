"""Trainable latent-prefix encoder separated from the frozen decoder."""
import torch
from torch import nn
from torch.nn import functional as F

from .attention import build_mask, encoder_position_ids


class FP32Projection(nn.Linear):
    def forward(self, x):
        # Keep module calls so ZeRO's leaf-module parameter hooks still run.
        with torch.autocast(device_type=x.device.type, enabled=False):
            return F.linear(x.float(), self.weight.float(), self.bias.float())


class LlamaLatentEncoder(nn.Module):
    def __init__(self, encoder, num_latents=6, latent_dim=512, *, decoder_hidden_size=None,
                 input_substitution_probability=0.3, extra_latent_noise_std=3.0,
                 sample_posterior=True, posterior_noise_scale=1.0):
        super().__init__()
        self.encoder = encoder
        hidden = encoder.config.hidden_size
        self.num_latents = num_latents
        self.input_substitution_probability = input_substitution_probability
        self.extra_latent_noise_std = extra_latent_noise_std
        self.sample_posterior = sample_posterior
        self.posterior_noise_scale = posterior_noise_scale
        decoder_hidden_size = decoder_hidden_size or hidden
        self.queries = nn.Parameter(torch.randn(1, num_latents, hidden))
        self.fc_mu = FP32Projection(hidden, latent_dim)
        self.fc_var = FP32Projection(hidden, latent_dim)
        self.to_decoder = FP32Projection(latent_dim, decoder_hidden_size)
        self.ae = nn.Parameter(torch.randn(1, 1, decoder_hidden_size))

    def forward(self, ids, mask, eos_id, bos_id):
        source = ids
        if self.training:
            eligible = mask.bool() & (ids != eos_id) & (ids != bos_id)
            replace = (torch.rand(ids.shape, device=ids.device) < self.input_substitution_probability) & eligible
            random_ids = torch.randint(self.encoder.config.vocab_size, ids.shape, device=ids.device)
            source = torch.where(replace, random_ids, ids)
        embeds = self.encoder.get_input_embeddings()(source)
        queries = self.queries.expand(ids.shape[0], -1, -1).to(embeds.dtype)
        embeds = torch.cat((embeds, queries), dim=1)
        keys = torch.cat((mask, mask.new_ones(ids.shape[0], self.num_latents)), dim=1)
        attention = build_mask("hybrid", keys, ids.shape[1], self.num_latents, embeds.dtype)
        states = self.encoder(inputs_embeds=embeds, attention_mask=attention,
                              position_ids=encoder_position_ids(mask, self.num_latents),
                              use_cache=False, return_dict=True).last_hidden_state[:, -self.num_latents:]
        mu = self.fc_mu(states)
        logvar = self.fc_var(states).clamp(-30, 20)
        z = mu
        if self.training:
            if self.sample_posterior:
                z = z + torch.randn_like(mu) * torch.exp(0.5 * logvar) * self.posterior_noise_scale
            if self.extra_latent_noise_std:
                z = z + torch.randn_like(z) * self.extra_latent_noise_std
        prefix = torch.cat((self.to_decoder(z), self.ae.expand(ids.shape[0], -1, -1).float()), dim=1)
        kl = -0.5 * (1 + logvar - mu.square() - logvar.exp()).mean()
        return prefix, kl, mu.detach().mean(), logvar.detach().mean()


def frozen_decoder_loss(decoder, prefix, ids, mask, num_latents=6):
    device = decoder.get_input_embeddings().weight.device
    ids, mask = ids.to(device), mask.to(device)
    text = decoder.get_input_embeddings()(ids)
    embeds = torch.cat((prefix.to(device=device, dtype=text.dtype), text), dim=1)
    keys = torch.cat((mask.new_ones(ids.shape[0], num_latents + 1), mask), dim=1)
    attention = build_mask("hybrid", keys, 0, num_latents, embeds.dtype)
    hidden = decoder.model(inputs_embeds=embeds, attention_mask=attention,
                           use_cache=False, return_dict=True).last_hidden_state
    logits = decoder.lm_head(hidden[:, num_latents:num_latents + ids.shape[1]])
    labels = ids.masked_fill(mask == 0, -100)
    return F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), labels.reshape(-1))
