"""Verify part 1 only: packages, frozen files, tokenizer and CUDA availability."""
import hashlib, importlib, importlib.metadata as md, json, os, pathlib, subprocess, sys, traceback
from datetime import datetime, timezone
ROOT = pathlib.Path(__file__).resolve().parents[1]
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['WANDB_MODE'] = 'disabled'
manifest = json.loads((ROOT / 'setup/manifest.json').read_text())
checks = {}

def check(name, fn):
    try:
        checks[name] = {'ok': True, 'detail': fn()}
    except Exception:
        checks[name] = {'ok': False, 'error': traceback.format_exc()}

def sources():
    source = pathlib.Path(manifest['source']['snapshot'])
    for relative, expected in manifest['source']['file_sha256'].items():
        assert hashlib.sha256((source / relative).read_bytes()).hexdigest() == expected, relative
    assert subprocess.check_output(['git', '-C', manifest['source']['original'], 'rev-parse', 'HEAD'], text=True).strip() == manifest['source']['commit']
    assert not subprocess.check_output(['git', '-C', manifest['source']['original'], 'status', '--porcelain'], text=True).strip()
    return {'commit': manifest['source']['commit'], 'files_verified': len(manifest['source']['file_sha256'])}

def original():
    now = subprocess.check_output(['/opt/miniforge3/envs/diffuguard/bin/python', '-m', 'pip', 'freeze'], text=True).strip()
    assert now == (ROOT / 'setup/original-environment.freeze.txt').read_text().strip(), 'Original package listing changed'
    return 'Original diffuguard package listing unchanged'

def dependency_consistency():
    return subprocess.check_output([sys.executable, '-m', 'pip', 'check'], text=True).strip()

def independent_copy():
    import torch
    old = subprocess.check_output(['/opt/miniforge3/envs/diffuguard/bin/python', '-c', 'import torch; print(torch.__file__)'], text=True).strip()
    assert not os.path.samefile(torch.__file__, old), 'Torch files shared with original environment'
    return 'Torch package is an independent file copy'

def versions():
    expected = {'torch':'2.6.0', 'transformers':'4.57.0', 'accelerate':'1.11.0', 'deepspeed':'0.18.0', 'datasets':'4.2.0', 'sentencepiece':'0.2.0', 'lm_eval':'0.4.9.2'}
    found = {name: md.version(name) for name in expected}
    for name, value in expected.items():
        assert found[name] == value, (name, found[name], value)
    import torch
    assert torch.__version__ == '2.6.0+cu124', torch.__version__
    return found | {'torch_runtime':torch.__version__, 'python':sys.version, 'executable':sys.executable}

def imports():
    names = ['transformers', 'accelerate', 'deepspeed', 'datasets', 'torchmetrics', 'lm_eval', 'dllm.core.modules.recurrent', 'dllm.pipelines.recurrent_llada.models.modeling_recurrent_llada', 'dllm.core.trainers.recurrent_mdlm', 'dllm.core.samplers.recurrent_mdlm']
    return {name: str(importlib.import_module(name).__file__) for name in names}

def tokenizer():
    from transformers import AutoTokenizer
    path = manifest['model']['local_path']
    t = AutoTokenizer.from_pretrained(path, local_files_only=True)
    ids = t.apply_chat_template([{'role':'user','content':'Setup verification.'}], tokenize=True, add_generation_prompt=True)
    assert ids and t.convert_tokens_to_ids('<|mdm_mask|>') != t.unk_token_id
    return {'path':path, 'vocab_size':len(t), 'prompt_tokens':len(ids), 'mask_token_id':t.convert_tokens_to_ids('<|mdm_mask|>'), 'note':'Tokenization only; no model weights loaded; training boundary checks deferred to part 2'}

def cuda():
    import torch
    assert torch.cuda.is_available(), 'CUDA unavailable to this verification process'
    return {'cuda_runtime':torch.version.cuda, 'devices':[{'index':i, 'name':torch.cuda.get_device_name(i), 'total_bytes':torch.cuda.get_device_properties(i).total_memory, 'bf16_supported':torch.cuda.is_bf16_supported(including_emulation=False)} for i in range(torch.cuda.device_count())], 'note':'No model loaded or training memory fit verified'}

for name, fn in [('source_integrity',sources),('original_environment_preserved',original),('versions',versions),('dependency_consistency',dependency_consistency),('independent_copy',independent_copy),('imports',imports),('tokenizer',tokenizer),('cuda_visibility',cuda)]:
    check(name,fn)
result = {'timestamp_utc':datetime.now(timezone.utc).isoformat(), 'checks':checks, 'ok':all(v['ok'] for v in checks.values()), 'training_started':False,'model_inference_started':False}
(ROOT / 'setup/verification.json').write_text(json.dumps(result, indent=2)+'\n')
print(json.dumps(result,indent=2))
sys.exit(0 if result['ok'] else 1)
