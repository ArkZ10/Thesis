"""Offline, resumable generation. Each row is an independent user conversation."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import torch
import transformers
from transformers import AutoModel, AutoTokenizer

from .data import ROOT, UPSTREAM, digest, read_csv, write_csv

sys.path.insert(0, str(UPSTREAM))
from moral_sycophancy_scorer import parse_label


def load_journal(path, repair=False):
    """Recover complete records after interruption; drop only a torn final write."""
    if not path.exists():
        return []
    data = path.read_bytes()
    if data and not data.endswith(b'\n'):
        end = data.rfind(b'\n') + 1
        if repair:
            path.write_bytes(data[:end])
        data = data[:end]
    return [json.loads(line) for line in data.splitlines()]


def append_record(stream, record):
    stream.write(json.dumps(record, ensure_ascii=False) + '\n')
    stream.flush()
    os.fsync(stream.fileno())


class Generator:
    def __init__(self, config, device):
        self.config = config
        self.device = device
        torch.set_num_threads(4)
        torch.manual_seed(config['seed'])
        if device.startswith('cuda') and not torch.cuda.is_available():
            raise RuntimeError('CUDA requested but unavailable')
        self.tokenizer = AutoTokenizer.from_pretrained(config['model_path'], trust_remote_code=True, local_files_only=True)
        self.tokenizer.padding_side = 'left'
        self.model = AutoModel.from_pretrained(
            config['model_path'], trust_remote_code=True, local_files_only=True,
            torch_dtype=getattr(torch, config['dtype']), low_cpu_mem_usage=True,
            device_map=device,
        ).eval()
        self.mask_id = self.model.config.mask_token_id
        self.eos_ids = {self.model.config.eos_token_id, self.tokenizer.convert_tokens_to_ids('<|eot_id|>')}
        self.head = self.model.get_output_embeddings()
        self.hook = None
        self.projection_check = self.check_projection()
        self.hook = self.head.register_forward_pre_hook(
            lambda module, args: (args[0][:, -self.config['generation_length']:],)
        )

    @torch.inference_mode()
    def check_projection(self):
        # Output projection is positionwise. Verify selected logits against the
        # full projection on real weights before enabling the memory optimization.
        x = torch.tensor([[17, 781, self.mask_id, self.mask_id]], device=self.device)
        full = self.model(x).logits[:, -2:].clone()
        hook = self.head.register_forward_pre_hook(lambda module, args: (args[0][:, -2:],))
        try:
            selected = self.model(x).logits
        finally:
            hook.remove()
        if not torch.equal(full, selected):
            raise RuntimeError('Selected-position projection differs from full projection')
        return 'exact_logits_equal'

    def encode(self, row):
        text = self.config['prompt_template'].format(prompt=row['prompt'], instruction=self.config['instruction'])
        ids = self.tokenizer.apply_chat_template(
            [{'role': 'user', 'content': text}], tokenize=True,
            add_generation_prompt=self.config['add_generation_prompt'],
        )
        if self.mask_id in ids:
            raise ValueError('Prompt contains the model mask token')
        if len(ids) + self.config['generation_length'] > self.config['context_limit']:
            raise ValueError(f'Context limit exceeded: {len(ids)} input tokens')
        return ids

    @torch.inference_mode()
    def generate(self, encoded):
        config = self.config
        length, block = config['generation_length'], config['block_length']
        blocks = length // block
        steps_per_block = config['diffusion_steps'] // blocks
        if length % block or config['diffusion_steps'] % blocks or not 1 <= steps_per_block <= block:
            raise ValueError('Invalid diffusion schedule')
        padded = self.tokenizer.pad({'input_ids': encoded}, padding=True, return_tensors='pt')
        prompt = padded['input_ids'].to(self.device)
        attention = padded['attention_mask'].to(self.device)
        x = torch.cat([prompt, torch.full((len(encoded), length), self.mask_id, dtype=torch.long, device=self.device)], dim=1)
        attention = torch.cat([attention, torch.ones((len(encoded), length), dtype=attention.dtype, device=self.device)], dim=1)
        output = x[:, -length:]
        for block_index in range(blocks):
            for step in range(steps_per_block):
                logits = self.model(x, attention_mask=attention).logits.float()
                if not torch.isfinite(logits).all().item():
                    raise RuntimeError('Nonfinite output logits')
                predictions = logits.argmax(dim=-1)
                confidence = logits.softmax(dim=-1).gather(-1, predictions.unsqueeze(-1)).squeeze(-1)
                eligible = output == self.mask_id
                eligible[:, (block_index + 1) * block:] = False
                confidence.masked_fill_(~eligible, -torch.inf)
                count = block // steps_per_block + (step < block % steps_per_block)
                positions = confidence.topk(count, dim=-1).indices
                output.scatter_(1, positions, predictions.gather(1, positions))
        if (output == self.mask_id).any().item() or not torch.equal(x[:, :prompt.shape[1]], prompt):
            raise RuntimeError('Unresolved output masks or changed prompt')
        return output.cpu().tolist()


def input_rows(task, split, datasets):
    if not datasets:
        datasets = ['AITA-NTA-OG', 'AITA-NTA-FLIP'] if task == 'moral' else ['AITA-YTA', 'SS', 'OEQ']
    result = []
    sources = {}
    for name in datasets:
        path = ROOT / f'data/{split}/{name}.csv'
        if not path.exists():
            raise FileNotFoundError(f'Missing dataset: {path}')
        sources[str(path.relative_to(ROOT))] = digest(path.read_bytes())
        result.extend([{**r, 'dataset': name, 'key': f'{name}:{r["id"]}'} for r in read_csv(path)])
    return result, sources


def run(task, split, device='cuda:0', worker_index=0, workers=1, datasets=None, batch_size=None, output=None):
    config = json.loads((ROOT / f'configs/{task}.json').read_text())
    if batch_size:
        config['batch_size'] = batch_size
    rows, sources = input_rows(task, split, datasets)
    rows = [r for i, r in enumerate(rows) if i % workers == worker_index]
    directory = Path(output) if output else ROOT / f'results/{task}/{split}/worker-{worker_index}'
    directory.mkdir(parents=True, exist_ok=True)
    identity = {'config': config, 'input_sha256': sources, 'worker_index': worker_index, 'workers': workers}
    fingerprint = digest(json.dumps(identity, sort_keys=True).encode())
    metadata_path = directory / 'metadata.json'
    if metadata_path.exists() and json.loads(metadata_path.read_text())['fingerprint'] != fingerprint:
        raise ValueError('Refusing to resume a run with different input data or generation settings')
    journal = directory / 'responses.jsonl'
    previous = load_journal(journal, repair=True)
    done = {r['key'] for r in previous}
    if len(done) != len(previous):
        raise ValueError('Duplicate journal records')
    pending = [r for r in rows if r['key'] not in done]
    print(f'{task}/{split} worker {worker_index}: {len(previous)} done; {len(pending)} pending', flush=True)
    if not pending:
        write_csv(directory / 'responses.csv', previous)
        if metadata_path.exists():
            metadata = json.loads(metadata_path.read_text())
            metadata['completed_rows'] = len(previous)
            metadata.setdefault('completed_at_utc', datetime.now(timezone.utc).isoformat())
            metadata_path.write_text(json.dumps(metadata, indent=2) + '\n')
        return
    engine = Generator(config, device)
    metadata = {
        **identity, 'fingerprint': fingerprint,
        'started_at_utc': datetime.now(timezone.utc).isoformat(),
        'device': device, 'gpu': torch.cuda.get_device_name(device) if device.startswith('cuda') else None,
        'torch_version': torch.__version__, 'transformers_version': transformers.__version__,
        'python_executable': sys.executable, 'chat_template': engine.tokenizer.chat_template,
        'projection_check': engine.projection_check,
        'inference_source_sha256': digest(Path(__file__).read_bytes()),
        'confidence_dtype': 'float32', 'expected_rows': len(rows),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + '\n')
    encoded = []
    completed = list(previous)
    with journal.open('a') as stream:
        for row in pending:
            try:
                ids = engine.encode(row)
                encoded.append((row, ids))
            except ValueError as e:
                result = {**row, 'response': '', 'label': None, 'error': str(e), 'status': 'error', 'input_tokens': None}
                append_record(stream, result)
                completed.append(result)
        # Similar lengths reduce padding while each sequence remains isolated.
        encoded.sort(key=lambda pair: len(pair[1]))
        def process(batch):
            started = time.perf_counter()
            try:
                token_lists = engine.generate([ids for _, ids in batch])
            except torch.OutOfMemoryError:
                torch.cuda.empty_cache()
                if len(batch) == 1:
                    raise
                middle = len(batch) // 2
                process(batch[:middle])
                process(batch[middle:])
                return
            elapsed = time.perf_counter() - started
            for (row, ids), tokens in zip(batch, token_lists):
                raw = engine.tokenizer.decode(tokens, skip_special_tokens=True).strip()
                result = {
                    **row, 'response': raw, 'label': parse_label(raw) if task == 'moral' else None,
                    'error': '' if raw else 'Empty decoded response', 'status': 'ok' if raw else 'error',
                    'input_tokens': len(ids), 'output_token_ids': tokens,
                    'contains_end_token': bool(set(tokens) & engine.eos_ids),
                    'batch_seconds': round(elapsed, 4), 'batch_size': len(batch),
                }
                append_record(stream, result)
                completed.append(result)
            progress = {'completed': len(completed), 'expected': len(rows), 'last_batch_seconds': round(elapsed, 3)}
            (directory / 'progress.json').write_text(json.dumps(progress) + '\n')
            print(json.dumps(progress), flush=True)
        for offset in range(0, len(encoded), config['batch_size']):
            process(encoded[offset:offset + config['batch_size']])
    write_csv(directory / 'responses.csv', completed)
    metadata['completed_at_utc'] = datetime.now(timezone.utc).isoformat()
    metadata['completed_rows'] = len(completed)
    if device.startswith('cuda'):
        metadata['peak_cuda_allocated_gib'] = round(torch.cuda.max_memory_allocated(device) / 1024**3, 3)
    metadata_path.write_text(json.dumps(metadata, indent=2) + '\n')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--task', choices=['moral', 'advice'], required=True)
    p.add_argument('--split', choices=['sample', 'full'], required=True)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--worker-index', type=int, default=0)
    p.add_argument('--workers', type=int, default=1)
    p.add_argument('--datasets', nargs='+')
    p.add_argument('--batch-size', type=int)
    p.add_argument('--output')
    run(**vars(p.parse_args()))
