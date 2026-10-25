"""Finish verification of the already completed 20-step pilot. Never trains."""
import argparse
import gc
import json
from pathlib import Path
import subprocess
import sys
import traceback
from datetime import datetime, timezone
import phase3_pilot as pilot


def main(run):
    torch,sft=pilot.dependencies()
    from dllm.utils.models import get_model
    from safetensors.torch import load_file
    cfg=json.loads((run/'configuration.json').read_text())
    state=json.loads((run/'checkpoints/checkpoint-20/trainer_state.json').read_text())
    assert state['global_step']==20
    before=json.loads((run/'backbone_before_sha256.json').read_text())
    after=json.loads((run/'backbone_after_sha256.json').read_text())
    assert before==after
    gradients=[json.loads(s) for s in (run/'gradients.jsonl').read_text().splitlines()]
    assert len(gradients)==20
    assert all(any(r['modules'][m]['nonzero_gradient_tensors']>0 for r in gradients) for m in pilot.MODULES)
    assert not any('Nonfinite' in (run/'train.log').read_text() for _ in [0])
    # Recreate the initial memory modules with the original seed and arguments.
    # This supplements the original in-process change assertions, which passed
    # before the erroneous buffer-vs-parameter key comparison.
    args=sft.ModelArguments(model_name_or_path=cfg['backbone_path'],backbone_name_or_path=cfg['backbone_path'],freeze_backbone=True,dtype='bfloat16',num_memory_slots=64,state_hidden_size=1024,time_cond_dim=1024,mixer_dim=768,injector_dim=768,mixer_slot_attn=True,unroll_steps=4,state_norm_weight=1e-4,slot_diversity_weight=0,debug_stats=True)
    sft.prepare_recurrent_model_args(args)
    model=get_model(args,is_recurrent=True,training=True)
    assert pilot.hash_parameters(model.model)==before
    initial=pilot.module_snapshot(model)
    saved=load_file(str(run/'checkpoints/checkpoint-20/recurrent_modules.safetensors'))
    assert set(initial)<=set(saved)
    changes={}
    for m in pilot.MODULES:
        keys=[n for n in initial if n.startswith(m+'.')]
        changed=[n for n in keys if not torch.equal(initial[n],saved[n])]
        assert changed,f'No changes in {m}'
        changes[m]={'parameter_tensors':len(keys),'changed_parameter_tensors':len(changed),'max_absolute_change':max(float((saved[n].float()-initial[n].float()).abs().max()) for n in keys)}
    del model,initial;gc.collect();torch.cuda.empty_cache()
    pilot.write(run/'training_verification.json',{'status':'passed','optimizer_steps':20,'backbone_unchanged':True,'backbone_parameters_hashed':len(before),'modules':changes,'each_module_received_nonzero_gradients':True,'finite_loss_and_gradients':True,'changes_check':'Original in-process change assertions passed; supplemented by fresh seed-42 initialization comparison.','checkpoint_extra_buffer':['time_conditioner.time_emb_table'],'memory_preflight':json.loads((run/'memory_preflight.json').read_text())['memory'],'last_training_log':state['log_history'][-1],'no_additional_training_performed':True})
    # A separate process exercises the official lightweight-checkpoint loader.
    with (run/'reload.log').open('w') as log:
        subprocess.run([sys.executable,str(pilot.ROOT/'scripts/phase3_pilot.py'),'reload','--directory',str(run)],stdout=log,stderr=subprocess.STDOUT,check=True,cwd=pilot.ROOT)
    pilot.write(run/'verification_correction.json',{'issue':'Original verifier compared parameter-only keys to state_dict keys, which also contain a fixed time embedding buffer.','missing_parameters':[],'fix':'Compare full module state_dict, including buffers.','original_script_sha256':pilot.sha(run/'phase3_pilot.executed.py'),'fixed_script_sha256':pilot.sha(pilot.ROOT/'scripts/phase3_pilot.py'),'completion_initial_preserved':'completion.initial.json','training_repeated':False})
    pilot.write(run/'completion.json',{'status':'passed','phase':3,'finished_at_utc':datetime.now(timezone.utc).isoformat(),'optimizer_steps':20,'training_repeated':False,'verification_only_correction':True,'next_phase_requires_user_confirmation':True})
    print('PHASE 3 PASSED. No extra training steps. Stop before phase 4.')

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--directory',type=Path,required=True);a=p.parse_args()
    try:main(a.directory)
    except Exception:
        pilot.write(a.directory/'verification_error.json',{'error':traceback.format_exc()})
        raise
