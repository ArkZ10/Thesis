"""Local, unquantized masked-diffusion inference for the LLaDA checkpoint."""
import gc
import time

import torch
from loguru import logger
from transformers import AutoModel, AutoTokenizer


def validate_schedule(length, steps, block):
    if length < 1 or steps < 1 or block < 1 or length % block:
        raise ValueError('Require positive lengths and an output window divisible by block_length')
    blocks = length // block
    if steps % blocks or not 1 <= steps // blocks <= block:
        raise ValueError('diffusion_steps must divide across blocks, with 1..block_length steps per block')


@torch.inference_mode()
def unmask(model, prompt, length, steps, block, mask_id, end_ids, progress=None):
    """Unmask full-context blocks; an end token in a finalized block stops work."""
    validate_schedule(length, steps, block)
    tokens = torch.cat([
        prompt,
        torch.full((1, length), mask_id, dtype=torch.long, device=prompt.device),
    ], dim=1)
    output = tokens[:, prompt.shape[1]:]
    steps_per_block = steps // (length // block)
    completed_steps = 0
    for block_index in range(length // block):
        for step in range(steps_per_block):
            logits = model(tokens).logits[:, -length:].float()
            if not torch.isfinite(logits).all().item():
                raise RuntimeError('LLaDA produced nonfinite logits')
            predictions = logits.argmax(dim=-1)
            confidence = logits.softmax(dim=-1).gather(-1, predictions.unsqueeze(-1)).squeeze(-1)
            eligible = output == mask_id
            eligible[:, (block_index + 1) * block:] = False
            confidence.masked_fill_(~eligible, -torch.inf)
            count = block // steps_per_block + (step < block % steps_per_block)
            positions = confidence.topk(count, dim=-1).indices
            output.scatter_(1, positions, predictions.gather(1, positions))
            completed_steps += 1
        prefix = output[0, :(block_index + 1) * block].tolist()
        if mask_id in prefix or not torch.equal(tokens[:, :prompt.shape[1]], prompt):
            raise RuntimeError('LLaDA left unresolved masks or changed the input prompt')
        if progress:
            progress(len(prefix), completed_steps)
        first_end = next((i for i, token in enumerate(prefix) if token in end_ids), None)
        if first_end is not None:
            return prefix, first_end, completed_steps
    return output[0].tolist(), None, completed_steps


class LLaDAEngine:
    def __init__(self, model, max_tokens=1024, diffusion_steps=1024, block_length=32,
                 model_path=None, device='cuda:0', dtype='bfloat16', seed=0,
                 context_limit=4096, threads=4, temperature=0,
                 local_files_only=True):
        validate_schedule(max_tokens, diffusion_steps, block_length)
        if temperature != 0:
            raise ValueError('This LLaDA backend currently supports temperature=0 only')
        if device.startswith('cuda') and not torch.cuda.is_available():
            raise RuntimeError('CUDA unavailable. Run in a GPU-enabled terminal or execution environment.')
        torch.set_num_threads(threads)
        torch.manual_seed(seed)
        self.device = device
        self.length, self.steps, self.block = max_tokens, diffusion_steps, block_length
        self.seed = seed
        self.model_path = model_path or model
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, trust_remote_code=True, local_files_only=local_files_only,
        )
        self.model = AutoModel.from_pretrained(
            self.model_path, trust_remote_code=True, local_files_only=local_files_only,
            torch_dtype=getattr(torch, dtype), low_cpu_mem_usage=True, device_map=device,
        ).eval()
        self.context_limit = min(context_limit, self.model.config.max_sequence_length)
        self.mask_id = self.model.config.mask_token_id
        self.end_ids = {self.model.config.eos_token_id, self.tokenizer.eos_token_id}
        self.end_ids.add(self.tokenizer.get_vocab().get('<|eot_id|>'))
        self.end_ids.discard(None)
        # The output projection acts independently on each position. Check the
        # optimization on real weights before retaining only generated positions.
        with torch.inference_mode():
            probe = torch.tensor([[17, 781, self.mask_id, self.mask_id]], device=device)
            full = self.model(probe).logits[:, -2:].clone()
            hook = self.model.get_output_embeddings().register_forward_pre_hook(
                lambda module, values: (values[0][:, -2:],)
            )
            try:
                selected = self.model(probe).logits
            finally:
                hook.remove()
            if not torch.equal(full, selected):
                raise RuntimeError('Selected-position output projection differs from full projection')
        self.hook = self.model.get_output_embeddings().register_forward_pre_hook(
            lambda module, values: (values[0][:, -self.length:],)
        )
        self.dtype = dtype

    def run_queries(self, queries):
        for index, (messages, image_path) in enumerate(queries):
            if image_path is not None:
                raise ValueError('LLaDA is a text-only backend')
            ids = self.tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
            if self.mask_id in ids:
                raise ValueError('Prompt contains a LLaDA mask token')
            if len(ids) + self.length > self.context_limit:
                raise ValueError(f'Context overflow: {len(ids)} + {self.length} > {self.context_limit}')
            prompt = torch.tensor([ids], dtype=torch.long, device=self.device)
            started = time.perf_counter()
            def progress(tokens, steps):
                logger.info(f'LLaDA query {index}: {tokens}/{self.length} positions, '
                            f'{steps}/{self.steps} diffusion steps')
            tokens, first_end, completed_steps = unmask(
                self.model, prompt, self.length, self.steps, self.block,
                self.mask_id, self.end_ids, progress,
            )
            answer_tokens = tokens if first_end is None else tokens[:first_end]
            answer = self.tokenizer.decode(answer_tokens, skip_special_tokens=True).strip()
            generation = {
                'status': 'ok' if answer else 'empty',
                'model_path': self.model_path, 'dtype': self.dtype, 'seed': self.seed,
                'max_tokens': self.length, 'diffusion_steps': self.steps,
                'completed_steps': completed_steps, 'block_length': self.block,
                'temperature': 0, 'context_limit': self.context_limit,
                'seconds': round(time.perf_counter() - started, 3),
                'output_token_ids': tokens, 'first_end_position': first_end,
                'answer_tokens': len(answer_tokens), 'answer_words': len(answer.split()),
                'end_token_seen': first_end is not None,
                'output_window_exhausted': len(tokens) == self.length,
                'projection_check': 'exact_logits_equal',
            }
            yield index, answer, {
                'cost': 0, 'input_tokens': len(ids), 'output_tokens': len(answer_tokens),
                'generation': generation,
            }

    def close(self):
        self.hook.remove()
        del self.model
        gc.collect()
        if self.device.startswith('cuda'):
            torch.cuda.empty_cache()
