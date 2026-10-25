"""Untrained architecture prototype, not a reproduction of the LaDiR checkpoint.

Blocks are generated sequentially by conditional continuous flow matching.
Each block contains several continuous vectors, not vocabulary tokens or named
reasoning steps. The text answer remains LLaDA masked-token diffusion.
No published Llama/LaDiR tensors are transferred into this architecture.
"""
from dataclasses import dataclass
import math
import threading

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class BranchState:
    prompt_ids: torch.Tensor
    prefix: torch.Tensor  # [batch=1, completed_blocks, slots, latent_dim]
    assistant_ids: torch.Tensor

    def validate(self, slots, dim):
        if self.prompt_ids.ndim != 2 or self.prompt_ids.shape[0] != 1 or not self.prompt_ids.numel():
            raise ValueError('Prototype requires one nonempty prompt, without padding.')
        if self.assistant_ids.ndim != 2 or self.assistant_ids.shape[0] != 1:
            raise ValueError('Assistant boundary must be an explicit token sequence.')
        if self.prefix.ndim != 4 or self.prefix.shape[0] != 1 or self.prefix.shape[2:] != (slots, dim):
            raise ValueError('Invalid completed-block prefix shape.')
        if self.prompt_ids.dtype != torch.long or self.assistant_ids.dtype != torch.long:
            raise ValueError('Prompt and assistant boundary must contain integer token IDs.')
        if not torch.isfinite(self.prefix).all():
            raise ValueError('Nonfinite completed latent prefix.')

    def save(self, path):
        from safetensors.torch import save_file
        save_file({k: v.detach().cpu().contiguous() for k, v in vars(self).items()}, str(path))

    @classmethod
    def load(cls, path, device='cpu'):
        from safetensors.torch import load_file
        tensors = load_file(str(path), device=str(device))
        if set(tensors) != {'prompt_ids', 'prefix', 'assistant_ids'}:
            raise ValueError('Unexpected saved branch-state fields.')
        return cls(**tensors)


def state_from_messages(tokenizer, messages, slots=3, dim=128, device='cpu'):
    """Split the actual LLaDA template before its final assistant header.

    This tokenizer appends the assistant header even when add_generation_prompt
    is false. Comparing the true/false renderings therefore gives no boundary.
    Split token IDs directly, preserving the complete native template exactly.
    """
    if len(messages) != 1 or messages[0].get('role') != 'user':
        raise ValueError('This pilot requires a fresh, single-user conversation.')
    ids = tokenizer.apply_chat_template(messages, tokenize=True,
        add_generation_prompt=True, return_tensors='pt').to(device=device, dtype=torch.long)
    header = tokenizer.convert_tokens_to_ids('<|start_header_id|>')
    if '<|start_header_id|>' not in tokenizer.get_vocab():
        raise ValueError('Expected pinned LLaDA header token.')
    starts = (ids[0] == header).nonzero().flatten()
    if not starts.numel():
        raise ValueError('Missing assistant header.')
    boundary = int(starts[-1])
    suffix = tokenizer.decode(ids[0, boundary:], skip_special_tokens=False)
    if not suffix.startswith('<|start_header_id|>assistant<|end_header_id|>'):
        raise ValueError('Template does not end with the expected assistant boundary.')
    return BranchState(ids[:, :boundary].clone(),
        torch.empty((1, 0, slots, dim), device=device), ids[:, boundary:].clone())


def block_bias(prompt_length, blocks, slots, tail_length, device):
    """Prompt cannot read blocks; block k reads prompt and blocks <= k.

    The answer region is bidirectional internally, as required by LLaDA. Earlier
    regions cannot read its tokens, so teacher-forced answers cannot leak back
    into the latent computation. Float additive masks avoid bool-mask ambiguity
    in the published LLaDA implementation.
    """
    groups = torch.cat((torch.zeros(prompt_length, device=device, dtype=torch.long),
        torch.arange(1, blocks + 1, device=device).repeat_interleave(slots),
        torch.full((tail_length,), blocks + 1, device=device, dtype=torch.long)))
    allowed = groups[:, None] >= groups[None, :]
    return torch.zeros(allowed.shape, device=device).masked_fill(~allowed, -torch.inf)[None, None]


class LLaDABlocks(nn.Module):
    def __init__(self, backbone, latent_dim=128, slots=3, max_blocks=4):
        super().__init__()
        if backbone.config.weight_tying:
            raise ValueError('Sparse output capture currently requires untied LLaDA output weights.')
        self.backbone = backbone
        self.dim, self.slots, self.max_blocks = latent_dim, slots, max_blocks
        hidden = backbone.config.d_model
        for p in backbone.parameters():
            p.requires_grad_(False)
        backbone.eval()
        self.queries = nn.Parameter(torch.randn(slots, hidden) / math.sqrt(hidden))
        self.mean = nn.Linear(hidden, latent_dim)
        self.logvar = nn.Linear(hidden, latent_dim)
        self.from_latent = nn.Linear(latent_dim, hidden)
        self.to_velocity = nn.Linear(hidden, latent_dim)
        self.time = nn.Sequential(nn.Linear(32, hidden), nn.SiLU(), nn.Linear(hidden, hidden))
        self.block_index = nn.Embedding(max_blocks, hidden)
        self._forward_lock = threading.RLock()

    def train(self, mode=True):
        super().train(mode)
        self.backbone.eval()  # Frozen backbone dropout must remain disabled.
        return self

    def _run(self, embeddings, bias, positions, logits=False):
        """Capture selected final states without allocating every vocabulary logit.

        Hooks are temporary and serialized. This is a single-process prototype;
        do not access its backbone directly from another thread while running.
        The unchanged output head receives only selected answer positions, or a
        zero-length input when only continuous hidden states are required.
        """
        captured = []

        def select(module, args):
            hidden = args[0].index_select(1, positions)
            captured.append(hidden)
            return (hidden if logits else hidden[:, :0],)

        with self._forward_lock:
            hook = self.backbone.model.transformer.ff_out.register_forward_pre_hook(select)
            try:
                output = self.backbone(inputs_embeds=embeddings, attention_bias=bias,
                    use_cache=False, output_hidden_states=False, return_dict=True)
            finally:
                hook.remove()
        if len(captured) != 1:
            raise RuntimeError('Expected exactly one output projection.')
        return output.logits if logits else captured[0]

    def _embed(self, ids):
        return self.backbone.get_input_embeddings()(ids)

    def encode_trace(self, trace_ids):
        """Training-only codec input: a supplied trace chunk, never test references."""
        if trace_ids.ndim != 2 or trace_ids.shape[0] != 1 or not trace_ids.numel():
            raise ValueError('Expected one nonempty trace chunk.')
        positions = torch.arange(trace_ids.shape[1], device=trace_ids.device)
        # Bidirectional trace encoding, separate from next-block generation.
        with torch.no_grad():
            hidden = self._run(self._embed(trace_ids), None, positions)
        scores = torch.einsum('sh,bth->bst', self.queries.to(hidden.dtype), hidden) / math.sqrt(hidden.shape[-1])
        pooled = torch.einsum('bst,bth->bsh', scores.softmax(-1), hidden)
        mu = self.mean(pooled)
        logvar = self.logvar(pooled).clamp(-20, 10)
        return mu, logvar

    def _context(self, state):
        state.validate(self.slots, self.dim)
        if state.prefix.shape[1] > self.max_blocks:
            raise ValueError('Too many completed blocks.')
        prompt = self._embed(state.prompt_ids)
        prefix = self.from_latent(state.prefix).flatten(1, 2).to(prompt.dtype)
        return torch.cat((prompt, prefix), dim=1)

    def velocity(self, state, noisy, t):
        if noisy.shape != (1, self.slots, self.dim) or not torch.isfinite(noisy).all():
            raise ValueError('Invalid current latent block.')
        k = state.prefix.shape[1]
        if k >= self.max_blocks:
            raise ValueError('No next-block capacity remains.')
        if t.numel() != 1 or not bool(((t >= 0) & (t <= 1)).all()):
            raise ValueError('Flow time must be a scalar in [0,1].')
        context = self._context(state)
        frequency = torch.exp(torch.arange(16, device=noisy.device) * (-math.log(10000) / 15))
        phase = t.float().reshape(1, 1) * frequency[None] * 1000
        te = self.time(torch.cat((phase.sin(), phase.cos()), -1)).unsqueeze(1)
        current = self.from_latent(noisy) + te + self.block_index.weight[k][None, None]
        embeddings = torch.cat((context, current.to(context.dtype)), 1)
        bias = block_bias(state.prompt_ids.shape[1], k + 1, self.slots, 0, noisy.device)
        positions = torch.arange(context.shape[1], embeddings.shape[1], device=noisy.device)
        return self.to_velocity(self._run(embeddings, bias, positions)).float()

    def flow_loss(self, state, clean, noise, t):
        # Linear flow: z_t=(1-t)z_clean+t*noise, v=noise-z_clean.
        noisy = (1 - t) * clean + t * noise
        return F.mse_loss(self.velocity(state, noisy, t), (noise - clean).float())

    def answer_logits(self, state, answer_ids, positions=None):
        """Same-position masked diffusion logits; no autoregressive target shift."""
        context = self._context(state)
        tail = torch.cat((self._embed(state.assistant_ids), self._embed(answer_ids)), 1)
        embeddings = torch.cat((context, tail), 1)
        offset = context.shape[1] + state.assistant_ids.shape[1]
        if positions is None:
            positions = torch.arange(answer_ids.shape[1], device=answer_ids.device)
        if positions.numel() == 0 or bool(((positions < 0) | (positions >= answer_ids.shape[1])).any()):
            raise ValueError('Answer projection positions are out of range or empty.')
        bias = block_bias(state.prompt_ids.shape[1], state.prefix.shape[1], self.slots,
                          tail.shape[1], answer_ids.device)
        return self._run(embeddings, bias, positions + offset, logits=True)

    def masked_answer_loss(self, state, target_ids, masked_positions, mask_id, mask_probability):
        if not 0 < mask_probability <= 1 or masked_positions.dtype != torch.bool:
            raise ValueError('Invalid corruption probability or mask.')
        if target_ids.shape != masked_positions.shape or target_ids.shape[0] != 1 or not masked_positions.any():
            raise ValueError('Expected at least one masked answer target.')
        corrupted = target_ids.masked_fill(masked_positions, mask_id)
        positions = masked_positions[0].nonzero().flatten()
        logits = self.answer_logits(state, corrupted, positions)
        # Mean over all target positions, weighting sampled positions by 1/p.
        loss = F.cross_entropy(logits.float().flatten(0, 1), target_ids[masked_positions], reduction='sum')
        return loss / (mask_probability * target_ids.numel())

    @torch.no_grad()
    def sample_next(self, state, seed, steps=16):
        """Sample exactly one next block; completed prefix is copied unchanged."""
        state.validate(self.slots, self.dim)
        if steps < 1:
            raise ValueError('At least one flow step is required.')
        rng = torch.Generator(device=state.prefix.device).manual_seed(seed)
        z = torch.randn((1, self.slots, self.dim), device=state.prefix.device, generator=rng)
        for i in range(steps):
            t = torch.tensor(1 - i / steps, device=z.device)
            z = z - self.velocity(state, z, t) / steps
            if not torch.isfinite(z).all():
                raise RuntimeError('Nonfinite sampled latent block.')
        return BranchState(state.prompt_ids.detach().clone(),
            torch.cat((state.prefix.detach().clone(), z[:, None]), dim=1),
            state.assistant_ids.detach().clone())
