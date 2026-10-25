"""Integration check against the previously verified, unoptimized single-row sampler."""
import importlib.util
import json
from pathlib import Path
import sys
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from llada_elephant.inference import Generator
from llada_elephant.data import read_csv

config = json.loads((ROOT / 'configs/moral.json').read_text())
engine = Generator(config, 'cuda:0')
rows = read_csv(ROOT / 'data/sample/AITA-NTA-OG.csv')
row = next(r for r in rows if r['id'] == 'cs336v')
ids = engine.encode(row)
optimized = engine.generate([ids])[0]
engine.hook.remove()
spec = importlib.util.spec_from_file_location('smoke', ROOT.parent / 'inference-test/test_inference.py')
smoke = importlib.util.module_from_spec(spec)
spec.loader.exec_module(smoke)
reference = smoke.generate(engine.model, torch.tensor([ids], device='cuda:0'), 16, 16, engine.mask_id)[0].cpu().tolist()
result = {'sample_id': row['id'], 'projection_check': engine.projection_check,
          'optimized_vs_reference_single_exact_tokens': optimized == reference,
          'optimized_response': engine.tokenizer.decode(optimized, skip_special_tokens=True),
          'reference_response': engine.tokenizer.decode(reference, skip_special_tokens=True)}
sample_rows = [json.loads(line) for line in (ROOT / 'results/moral/sample/worker-0/responses.jsonl').read_text().splitlines()]
batched = next(item for item in sample_rows if item['key'] == f'AITA-NTA-OG:{row["id"]}')
result['batched_sample_response'] = batched['response']
result['batched_sample_matches_reference_label'] = batched['response'] == result['reference_response'].strip()
print(json.dumps(result, indent=2))
(ROOT / 'reports/generation_validation.json').write_text(json.dumps(result, indent=2) + '\n')
if optimized != reference:
    raise RuntimeError('Optimized sampler differs from reference')
