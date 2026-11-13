"""Restore the complete real checkpoint and continue one discarded update."""
import json
from pathlib import Path
import time
import torch
from safetensors import safe_open

from scripts.large_runtime import ROOT,build_model,batch_for
from scripts.run_large_split_smoke import frozen_hash,write
from vae.split_training import StreamedCPUAdamW


def main():
    started=time.time()
    run=ROOT/'outputs/vae-large-split-smoke-20261112-02'
    checkpoint=run/'checkpoint_step_2'
    config=json.loads((run/'config.json').read_text())
    torch.manual_seed(42)
    print('Loading model pair for checkpoint-resume verification',flush=True)
    model,tokenizer=build_model(config)
    optimizer=StreamedCPUAdamW(model.named_parameters(),config['train']['lr'],
        tuple(config['train']['betas']),config['train']['weight_decay'])
    print('Restoring FP32 master weights and Adam moments from the full checkpoint',flush=True)
    state=torch.load(checkpoint/'optimizer.pt',map_location='cpu',weights_only=True,mmap=True)
    optimizer.load_state_dict(state)
    assert optimizer.steps==2
    assert all(torch.equal(master,saved) for master,saved in zip(optimizer.masters,state['master_weights']))
    assert len(optimizer.optimizer.state)==len(optimizer.masters)==406
    del state
    named=dict(model.named_parameters())
    index=json.loads((checkpoint/'trainable/index.json').read_text())['weight_map']
    verified=0
    for filename in sorted(set(index.values())):
        with safe_open(checkpoint/'trainable'/filename,framework='pt',device='cpu') as stream:
            for name in stream.keys():
                assert torch.equal(stream.get_tensor(name),named[name].detach().cpu()),name
                verified+=1
    assert verified==406
    rng=torch.load(checkpoint/'rng.pt',map_location='cpu',weights_only=True)
    torch.set_rng_state(rng['torch_rng']);torch.cuda.set_rng_state_all(rng['cuda_rng'])
    assert torch.equal(torch.get_rng_state(),rng['torch_rng'])
    assert all(torch.equal(a,b) for a,b in zip(torch.cuda.get_rng_state_all(),rng['cuda_rng']))
    before=frozen_hash(model)
    assert before==json.loads((run/'summary.json').read_text())['decoder_hash_after']
    fixtures=[json.loads(l) for l in (ROOT/'data/processed/vae-large-comparable-v1/smoke.jsonl').open()]
    rows=[fixtures[i%len(fixtures)] for i in range(16)]
    batch=batch_for(rows,tokenizer)
    first=optimizer.masters[0].detach().flatten()[:1024].clone()
    print('Checkpoint restored; checking a real continuation update (discarded afterward)',flush=True)
    model.train()
    with torch.autocast('cuda',dtype=torch.bfloat16):result=model(**batch,kl_weight=config['train']['kl_weight'])
    assert torch.isfinite(result['loss'])
    loss=float(result['loss']);result['loss'].backward()
    norm=optimizer.step()
    assert optimizer.steps==3
    assert not torch.equal(first,optimizer.masters[0].detach().flatten()[:1024])
    after=frozen_hash(model);assert after==before
    report=dict(passed=True,source=str(checkpoint),source_step=2,continued_step=3,
        model_tensors_verified=verified,fp32_masters_and_adam_entries_restored=406,
        rng_restored=True,continuation_loss=loss,continuation_gradient_norm=norm,
        decoder_unchanged=True,probe_updates_discarded=True,
        pilot_initialization='Fresh pretrained checkpoints in a separate process; no smoke/resume-test updates reused',
        seconds=time.time()-started)
    write(ROOT/'manifests/large_resume_verification.json',report)
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()
