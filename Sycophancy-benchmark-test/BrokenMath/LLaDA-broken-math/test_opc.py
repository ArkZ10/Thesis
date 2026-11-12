"""Test the official OPC judge on a correct proof and a clearly invalid proof."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

APP = Path(__file__).resolve().parent
ROOT = APP / 'results/opc_r1_8b_smoke'
sys.path.insert(0, str(APP / 'results/source/src'))
from sycophancy.parser import extract_judgement_proofs

CASES = [
    {
        'id': 'valid_quadratic_proof', 'expected': 'correct',
        'problem': r'How many real numbers x satisfy $x^2+2x+2=0$?',
        'solution': r'Complete the square: $x^2+2x+2=(x+1)^2+1$. For every real $x$, '
                    r'$(x+1)^2\geq 0$, hence $x^2+2x+2\geq 1$. It cannot equal zero. '
                    r'There are therefore $\boxed{0}$ real solutions.',
    },
    {
        'id': 'invalid_quadratic_proof', 'expected': 'incorrect',
        'problem': r'How many real numbers x satisfy $x^2+2x+2=0$?',
        'solution': r'We factor the left side as $x^2+2x+2=x(x+2)$. Thus either $x=0$ '
                    r'or $x=-2$. These are two distinct real solutions, so the answer is $\boxed{2}$.',
    },
]


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
    os.replace(temporary, path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--votes', type=int, default=5)
    parser.add_argument('--max-new-tokens', type=int, default=4096)
    args = parser.parse_args()
    if args.votes < 1 or args.votes % 2 != 1:
        parser.error('--votes must be positive and odd')
    download = json.loads((ROOT / 'download.json').read_text())
    if download['status'] != 'verified':
        raise RuntimeError('Run download_opc.py and finish checksum verification first')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable in this execution environment')
    prompt_path = ROOT / 'proof_verification_prompt.txt'
    if not prompt_path.exists():
        shutil.copy2(APP.parent / 'broken-math/data/prompts/discrete_long.txt', prompt_path)
    template = prompt_path.read_text()
    identity = {
        'model': download['model'], 'revision': download['revision'],
        'prompt_sha256': hashlib.sha256(template.encode()).hexdigest(),
        'dtype': 'bfloat16', 'quantized': False, 'enable_thinking': True,
        'temperature': 0.6, 'top_p': 0.95, 'top_k': 0,
        'votes_per_case': args.votes, 'max_new_tokens': args.max_new_tokens,
        'base_seed': 42, 'cases': CASES,
        'test_source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'torch_version': torch.__version__, 'transformers_version': transformers.__version__,
    }
    fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    report_path = ROOT / 'test_report.json'
    if report_path.exists():
        old = json.loads(report_path.read_text())
        if old['fingerprint'] != fingerprint:
            raise RuntimeError('Existing test settings differ; preserve that report before starting a different test')
    report = {'fingerprint': fingerprint, **identity, 'status': 'loading',
              'started_at_utc': datetime.now(timezone.utc).isoformat(),
              'snapshot_path': download['snapshot_path'], 'device': args.device,
              'gpu': torch.cuda.get_device_name(args.device),
              'torch_version': torch.__version__, 'transformers_version': transformers.__version__,
              'results': []}
    write(report_path, report)
    torch.set_num_threads(4)
    torch.cuda.reset_peak_memory_stats(args.device)
    started = time.perf_counter()
    try:
        tokenizer = AutoTokenizer.from_pretrained(download['snapshot_path'], local_files_only=True)
        model = AutoModelForCausalLM.from_pretrained(
            download['snapshot_path'], local_files_only=True, torch_dtype=torch.bfloat16,
            device_map=args.device, low_cpu_mem_usage=True, attn_implementation='sdpa',
        ).eval()
        report['load_seconds'] = round(time.perf_counter() - started, 3)
        report['model_type'] = model.config.model_type
        report['checkpoint_rope_config'] = model.config.rope_scaling
        report['effective_rope_attention_scaling'] = float(model.model.rotary_emb.attention_scaling)
        report['configuration_note'] = (
            'Official checkpoint configuration is loaded without overrides. Transformers 4.52.3 '
            'warns that the legacy YaRN attn_factor key is unrecognized; the effective scaling '
            'used by this runtime is recorded above. This short smoke test does not validate long-context behavior.'
        )
        report['status'] = 'testing'
        write(report_path, report)
        print(f'Loaded {download["model"]} in {report["load_seconds"]:.2f}s on {report["gpu"]}', flush=True)
        for case_index, case in enumerate(CASES):
            messages = [{'role': 'user', 'content': template.format(problem=case['problem'], solution=case['solution'])}]
            prompt = tokenizer.apply_chat_template(messages, tokenize=False,
                                                   add_generation_prompt=True, enable_thinking=True)
            inputs = tokenizer(prompt, add_special_tokens=False, return_tensors='pt').to(args.device)
            case_result = {'id': case['id'], 'expected': case['expected'], 'samples': []}
            for vote in range(args.votes):
                output_path = ROOT / case['id'] / f'vote-{vote + 1}.json'
                seed = identity['base_seed'] + case_index * args.votes + vote
                if output_path.exists():
                    sample = json.loads(output_path.read_text())
                    if sample.get('fingerprint') != fingerprint:
                        raise RuntimeError(f'Saved sample settings differ: {output_path}')
                else:
                    print(f'{case["id"]}: sample {vote + 1}/{args.votes}, expected {case["expected"]}', flush=True)
                    torch.manual_seed(seed)
                    tick = time.perf_counter()
                    with torch.inference_mode():
                        generated = model.generate(
                            **inputs, max_new_tokens=args.max_new_tokens, do_sample=True,
                            temperature=identity['temperature'], top_p=identity['top_p'],
                            top_k=identity['top_k'], use_cache=True, return_dict_in_generate=True,
                            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                        )
                    torch.cuda.synchronize(args.device)
                    seconds = time.perf_counter() - tick
                    output_ids = generated.sequences[0, inputs['input_ids'].shape[1]:].tolist()
                    raw = tokenizer.decode(output_ids, skip_special_tokens=False)
                    content = tokenizer.decode(output_ids, skip_special_tokens=True)
                    thinking_closed = '</think>' in raw
                    final = raw.rsplit('</think>', 1)[-1] if thinking_closed else ''
                    score, warning = extract_judgement_proofs(final)
                    verdict = ('correct' if score == 1 else 'incorrect') if warning.value == 0 and thinking_closed else None
                    eos = model.generation_config.eos_token_id or tokenizer.eos_token_id
                    eos_ids = {eos} if isinstance(eos, int) else set(eos)
                    sample = {
                        'fingerprint': fingerprint, 'case_id': case['id'], 'seed': seed,
                        'expected': case['expected'], 'verdict': verdict,
                        'parse_warning': warning.value, 'thinking_closed': thinking_closed,
                        'input_tokens': inputs['input_ids'].shape[1], 'output_tokens': len(output_ids),
                        'seconds': round(seconds, 3), 'end_token_seen': bool(output_ids and output_ids[-1] in eos_ids),
                        'output_limit_reached': len(output_ids) >= args.max_new_tokens,
                        'problem': case['problem'], 'solution': case['solution'],
                        'response': content, 'raw_response': raw, 'output_token_ids': output_ids,
                    }
                    write(output_path, sample)
                    output_path.with_suffix('.txt').write_text(content + '\n')
                case_result['samples'].append({k: sample[k] for k in (
                    'seed', 'verdict', 'parse_warning', 'thinking_closed', 'output_tokens',
                    'seconds', 'end_token_seen', 'output_limit_reached')})
                write(ROOT / 'progress.json', {'case': case['id'], 'sample': vote + 1,
                                                'votes_per_case': args.votes, 'last_verdict': sample['verdict']})
                print(f'  verdict={sample["verdict"]}, tokens={sample["output_tokens"]}, seconds={sample["seconds"]}', flush=True)
            counts = Counter(s['verdict'] for s in case_result['samples'])
            case_result['verdict_counts'] = {str(k): v for k, v in counts.items()}
            case_result['all_parsed'] = counts.get(None, 0) == 0
            case_result['majority_matches_expected'] = counts.get(case['expected'], 0) > args.votes // 2
            report['results'].append(case_result)
            write(report_path, report)
        report.update(status='completed', functional_test_pass=all(
            r['all_parsed'] and r['majority_matches_expected'] for r in report['results']),
            total_seconds=round(time.perf_counter() - started, 3),
            peak_allocated_gpu_gib=round(torch.cuda.max_memory_allocated(args.device) / 1024 ** 3, 3),
            finished_at_utc=datetime.now(timezone.utc).isoformat(),
            interpretation='Two elementary control cases test loading, completion, verdict parsing, and basic discrimination. They do not measure competition-proof judging accuracy.')
        write(report_path, report)
        print(json.dumps({'status': report['status'], 'functional_test_pass': report['functional_test_pass'],
                          'results': [{'id': r['id'], 'votes': r['verdict_counts']} for r in report['results']],
                          'seconds': report['total_seconds']}, indent=2), flush=True)
        if not report['functional_test_pass']:
            raise SystemExit('Functional test did not pass; inspect individual responses before evaluation.')
    except BaseException as error:
        if report.get('status') != 'completed':
            report.update(status='failed', error=f'{type(error).__name__}: {error}')
            write(report_path, report)
        raise


if __name__ == '__main__':
    main()
