"""Run the cached, unquantized LLaDA checkpoint and measure text before EOS/EOT."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time

import torch
import transformers
from transformers import AutoModel, AutoTokenizer

MODEL = Path('/root/.cache/huggingface/hub/models--GSAI-ML--LLaDA-8B-Instruct/snapshots/08b83a6feb34df1a6011b80c3c00c7563e963b07')
PROMPT = '''Write a detailed mathematical explanation of approximately 600 words for a student. Give a complete proof that there are infinitely many primes, followed by a complete proof that the square root of 2 is irrational. For each proof, explain the assumptions, each logical step, why the contradiction works, and one common mistake to avoid. Use paragraphs and equations. Finish with a short comparison of the two proofs. Do not just give an outline.'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--length', type=int, default=1024)
    parser.add_argument('--steps', type=int, default=128)
    parser.add_argument('--block', type=int, default=32)
    parser.add_argument('--threads', type=int, default=12)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--benchmark-index', type=int)
    args = parser.parse_args()
    if args.length % args.block:
        parser.error('length must be divisible by block')
    blocks = args.length // args.block
    if args.steps % blocks or not 1 <= args.steps // blocks <= args.block:
        parser.error('invalid diffusion schedule')
    torch.set_num_threads(args.threads)
    torch.manual_seed(0)
    prompt_text = PROMPT
    problem_id = None
    if args.benchmark_index is not None:
        import yaml
        root = Path(__file__).resolve().parents[1] / 'broken-math'
        data = json.loads((root / 'data/raw/sycophancy_recent/sample.json').read_text())
        row = data[args.benchmark_index]
        config = yaml.safe_load((root / 'configs/solvers/sycophancy_recent.yaml').read_text())
        template = config['prompt_gold'] if row.get('gold_answer') else config['prompt']
        prompt_text = template.format(**row)
        problem_id = row['problem_id']
    dtype = torch.float32 if args.device == 'cpu' else torch.bfloat16
    report = {
        'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'model_path': str(MODEL), 'device': args.device, 'dtype': str(dtype),
        'quantized': False, 'torch_version': torch.__version__,
        'transformers_version': transformers.__version__,
        'length': args.length, 'steps': args.steps, 'block_length': args.block,
        'temperature': 0, 'seed': 0, 'threads': args.threads,
        'prompt': prompt_text, 'problem_id': problem_id, 'status': 'loading',
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        temporary = args.output.with_suffix('.tmp')
        temporary.write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n')
        os.replace(temporary, args.output)

    save()
    started = time.perf_counter()
    try:
        tokenizer = AutoTokenizer.from_pretrained(str(MODEL), trust_remote_code=True, local_files_only=True)
        ids = tokenizer.apply_chat_template([{'role': 'user', 'content': prompt_text}], tokenize=True, add_generation_prompt=True)
        model = AutoModel.from_pretrained(
            str(MODEL), trust_remote_code=True, local_files_only=True,
            torch_dtype=dtype, low_cpu_mem_usage=True, device_map=args.device,
        ).eval()
        report['load_seconds'] = round(time.perf_counter() - started, 3)
        report['input_tokens'] = len(ids)
        limit = model.config.max_sequence_length
        if len(ids) + args.length > limit:
            raise ValueError(f'Context overflow: {len(ids)} + {args.length} > {limit}')
        mask_id = model.config.mask_token_id
        if mask_id in ids:
            raise ValueError('Prompt contains a mask token')
        prompt = torch.tensor([ids], dtype=torch.long, device=args.device)
        x = torch.cat([prompt, torch.full((1, args.length), mask_id, dtype=torch.long, device=args.device)], dim=1)
        output = x[:, len(ids):]
        hook = model.get_output_embeddings().register_forward_pre_hook(lambda module, values: (values[0][:, -args.length:],))
        end_ids = {model.config.eos_token_id, tokenizer.eos_token_id, tokenizer.get_vocab()['<|eot_id|>']}
        report['status'] = 'generating'
        save()
        print(json.dumps({'event': 'loaded', 'seconds': report['load_seconds'], 'input_tokens': len(ids)}), flush=True)
        generation_started = time.perf_counter()
        steps_per_block = args.steps // blocks
        ended = False
        with torch.inference_mode():
            for b in range(blocks):
                for step in range(steps_per_block):
                    tick = time.perf_counter()
                    logits = model(x).logits.float()
                    if not torch.isfinite(logits).all().item():
                        raise RuntimeError('Nonfinite logits')
                    prediction = logits.argmax(dim=-1)
                    confidence = logits.softmax(dim=-1).gather(-1, prediction.unsqueeze(-1)).squeeze(-1)
                    eligible = output == mask_id
                    eligible[:, (b + 1) * args.block:] = False
                    confidence.masked_fill_(~eligible, -torch.inf)
                    count = args.block // steps_per_block + (step < args.block % steps_per_block)
                    positions = confidence.topk(count, dim=-1).indices
                    output.scatter_(1, positions, prediction.gather(1, positions))
                    elapsed = time.perf_counter() - generation_started
                    report.update(completed_steps=b * steps_per_block + step + 1,
                                  generation_seconds=round(elapsed, 3),
                                  partial_output_token_ids=output[0].tolist())
                    save()
                    print(json.dumps({'event': 'step', 'step': report['completed_steps'], 'total': args.steps,
                                      'step_seconds': round(time.perf_counter() - tick, 3), 'elapsed_seconds': round(elapsed, 3)}), flush=True)
                prefix = output[0, :(b + 1) * args.block].tolist()
                end_position = next((i for i, token in enumerate(prefix) if token in end_ids), None)
                visible = prefix if end_position is None else prefix[:end_position]
                report['completed_prefix_tokens'] = len(prefix)
                report['partial_response'] = tokenizer.decode(visible, skip_special_tokens=True).strip()
                save()
                print(json.dumps({'event': 'block', 'completed_prefix_tokens': len(prefix),
                                  'end_position': end_position, 'partial_response': report['partial_response']}, ensure_ascii=False), flush=True)
                if end_position is not None:
                    ended = True
                    break
        hook.remove()
        if not torch.equal(x[:, :len(ids)], prompt) or (not ended and (output == mask_id).any().item()):
            raise RuntimeError('Prompt changed or unresolved masks remain')
        tokens = output[0, :(b + 1) * args.block].tolist()
        first_end = next((i for i, token in enumerate(tokens) if token in end_ids), None)
        answer_tokens = tokens if first_end is None else tokens[:first_end]
        answer = tokenizer.decode(answer_tokens, skip_special_tokens=True).strip()
        report.update(status='completed', response=answer, output_token_ids=tokens,
                      answer_tokens_before_end=len(answer_tokens), answer_words=len(answer.split()),
                      first_end_position=first_end, possibly_truncated=first_end is None,
                      stopped_after_complete_block_at_end_token=ended,
                      generation_seconds=round(time.perf_counter() - generation_started, 3))
        args.output.with_suffix('.txt').write_text(answer + '\n')
        save()
        print(json.dumps({k: v for k, v in report.items() if k not in ('output_token_ids', 'partial_output_token_ids', 'partial_response', 'prompt')}, ensure_ascii=False), flush=True)
    except BaseException as error:
        report.update(status='failed', error=f'{type(error).__name__}: {error}', total_seconds=round(time.perf_counter() - started, 3))
        save()
        raise


if __name__ == '__main__':
    main()
