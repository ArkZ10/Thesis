"""Inference-only continuous workspace; the backbone and projection are frozen."""
import time
import torch
from llada_elephant.inference import Generator


class WorkspaceGenerator(Generator):
    def __init__(self, config, device, arm, directory):
        super().__init__(config, device)
        self.arm, self.directory = arm, directory
        embedding = self.model.get_input_embeddings()
        self.initial = embedding.weight[self.mask_id].detach().clone().repeat(4, 1)
        width = self.initial.shape[-1]
        self.projection = torch.nn.Linear(width, width, bias=False, device=device, dtype=self.initial.dtype)
        with torch.no_grad():
            self.projection.weight.copy_(torch.eye(width, device=device, dtype=self.initial.dtype))
        self.projection.requires_grad_(False)
        self.model.requires_grad_(False)
        torch.save({'initial_workspace': self.initial.cpu(), 'projection': 'identity',
                    'projection_shape': [width, width]}, directory / 'initialization.pt')

    @torch.inference_mode()
    def workspace(self, prompt, attention, rounds):
        embedding = self.model.get_input_embeddings()
        p = embedding(prompt)
        z = self.initial.unsqueeze(0).expand(prompt.shape[0], -1, -1).clone()
        masks = torch.cat([attention, torch.ones_like(attention[:, :1]).expand(-1, 4)], 1)
        diagnostics = [{'round': 0, 'mean_norm': z.float().norm(dim=-1).mean().item()}]
        for index in range(rounds):
            # Capture only the final normalized hidden states, avoiding retention of every layer.
            captured = []
            handle = self.model.model.transformer.ln_f.register_forward_hook(
                lambda module, args, output: captured.append(output[:, -4:].clone()))
            try:
                self.model(inputs_embeds=torch.cat([p, z], 1), attention_mask=masks, use_cache=False)
            finally:
                handle.remove()
            next_z = self.projection(captured[0])
            if not torch.isfinite(next_z).all():
                raise RuntimeError('Nonfinite workspace')
            diagnostics.append({'round': index + 1, 'mean_norm': next_z.float().norm(dim=-1).mean().item(),
                                'delta_norm': (next_z.float() - z.float()).norm().item()})
            z = next_z
        return p, z, diagnostics

    @torch.inference_mode()
    def generate(self, encoded):
        if self.arm == 'control':
            return super().generate(encoded)
        c = self.config
        length, block = c['generation_length'], c['block_length']
        blocks, steps = length // block, c['diffusion_steps'] // (length // block)
        if length % block or c['diffusion_steps'] % blocks or not 1 <= steps <= block:
            raise ValueError('Invalid schedule')
        padded = self.tokenizer.pad({'input_ids': encoded}, padding=True, return_tensors='pt')
        prompt, mask = (padded[k].to(self.device) for k in ['input_ids', 'attention_mask'])
        started = time.perf_counter()
        p, z, diagnostics = self.workspace(prompt, mask, 2 if self.arm == 'recurrent' else 0)
        torch.cuda.synchronize(self.device)
        workspace_seconds = time.perf_counter() - started
        prefix = torch.cat([p, z], 1)
        frozen = prefix.clone()
        answer = torch.full((len(encoded), length), self.mask_id, device=self.device, dtype=torch.long)
        attention = torch.cat([mask, torch.ones((len(encoded), 4 + length), device=self.device, dtype=mask.dtype)], 1)
        embedding = self.model.get_input_embeddings()
        for block_index in range(blocks):
            for step in range(steps):
                logits = self.model(inputs_embeds=torch.cat([prefix, embedding(answer)], 1),
                                    attention_mask=attention, use_cache=False).logits.float()
                if logits.shape[:2] != answer.shape or not torch.isfinite(logits).all():
                    raise RuntimeError('Invalid answer logits or indexing')
                predictions = logits.argmax(-1)
                confidence = logits.softmax(-1).gather(-1, predictions.unsqueeze(-1)).squeeze(-1)
                eligible = answer == self.mask_id
                eligible[:, (block_index + 1) * block:] = False
                confidence.masked_fill_(~eligible, -torch.inf)
                count = block // steps + (step < block % steps)
                positions = confidence.topk(count, dim=-1).indices
                answer.scatter_(1, positions, predictions.gather(1, positions))
        if (answer == self.mask_id).any() or not torch.equal(prefix, frozen):
            raise RuntimeError('Unresolved answer masks or modified prefix')
        import workflow as common
        with (self.directory / 'workspace_diagnostics.jsonl').open('a') as stream:
            common.append(stream, {'encoded_input_sha256': common.identity_hash(encoded), 'arm': self.arm,
                'rounds': diagnostics, 'workspace_seconds': workspace_seconds,
                'extra_backbone_passes': 2 if self.arm == 'recurrent' else 0,
                'answer_start': prefix.shape[1], 'answer_tokens': length,
                'prefix_unchanged': True, 'peak_allocated_gib': torch.cuda.max_memory_allocated(self.device)/1024**3})
        return answer.cpu().tolist()

    @torch.inference_mode()
    def verify(self, ids):
        prompt = torch.tensor([ids], device=self.device)
        mask = torch.ones_like(prompt)
        embedding = self.model.get_input_embeddings()
        token_logits = self.model(prompt, attention_mask=mask, use_cache=False).logits
        embed_logits = self.model(inputs_embeds=embedding(prompt), attention_mask=mask, use_cache=False).logits
        difference = (token_logits.float() - embed_logits.float()).abs().max().item()
        torch.testing.assert_close(token_logits, embed_logits, rtol=1e-3, atol=1e-3)
        del token_logits, embed_logits
        p, z, diag = self.workspace(prompt, mask, 2)
        if not all(x['delta_norm'] > 0 for x in diag[1:]):
            raise RuntimeError('Workspace did not change')
        # Verify hook captures exactly the public final-hidden-state interface.
        out = self.model(inputs_embeds=torch.cat([p, self.initial[None]], 1),
                         attention_mask=torch.ones((1, len(ids)+4), device=self.device),
                         output_hidden_states=True, use_cache=False)
        _, one, _ = self.workspace(prompt, mask, 1)
        torch.testing.assert_close(one, self.projection(out.hidden_states[-1][:, -4:]), rtol=0, atol=0)
        del out
        answer = embedding(torch.full((1, self.config['generation_length']), self.mask_id, device=self.device))
        attention = torch.ones((1, len(ids)+4+answer.shape[1]), device=self.device)
        original = self.model(inputs_embeds=torch.cat([p, z, answer], 1), attention_mask=attention).logits.float()
        changed = self.model(inputs_embeds=torch.cat([p, torch.zeros_like(z), answer], 1), attention_mask=attention).logits.float()
        effect = (original - changed).abs().max().item()
        if not effect > 0 or not torch.isfinite(changed).all():
            raise RuntimeError('Answer logits do not depend on workspace')
        del original, changed
        previous = self.arm
        self.arm = 'control'
        control = self.generate([ids])
        direct = super().generate([ids])
        if control != direct:
            raise RuntimeError('Disabled path differs from original sampler')
        self.arm = 'recurrent'
        recurrent = self.generate([ids])
        self.arm = 'fixed'
        fixed = self.generate([ids])
        self.arm = previous
        return {'status': 'passed', 'embedding_logits_max_abs_difference': difference,
                'workspace_logit_max_abs_effect': effect, 'rounds': diag,
                'hidden_capture_equals_public_interface': True, 'disabled_path_exact_tokens_equal': True,
                'control_tokens': control[0], 'fixed_tokens': fixed[0], 'recurrent_tokens': recurrent[0]}
