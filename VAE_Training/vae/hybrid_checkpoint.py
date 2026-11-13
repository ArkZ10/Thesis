"""Supplement ZeRO-3 checkpoints with hybrid GPU Adam states.

DeepSpeed 0.19.9's stage3._rigid_state_dict serializes the primary CPU
optimizer but omits backup_optimizer's GPU moment tensors. Store them per
subgroup without gathering model weights or copying all moments at once.
Call restore_hybrid_state after the ordinary engine.load_checkpoint.
"""
import json
from pathlib import Path

import torch


def save_hybrid_state(zero_optimizer, directory):
    directory=Path(directory)
    directory.mkdir(parents=True,exist_ok=False)
    backup=getattr(zero_optimizer,'backup_optimizer',None)
    entries=[]
    if backup is not None:
        for i,param in enumerate(zero_optimizer.fp32_partitioned_groups_flat):
            state=backup.state.get(param)
            if not state:
                continue
            name=f'subgroup_{i}.pt'
            torch.save(state,directory/name)
            entries.append(dict(subgroup=i,file=name,keys=sorted(state),numel=param.numel()))
    report=dict(format='deepspeed_hybrid_adam_v1',subgroups=entries)
    (directory/'index.json').write_text(json.dumps(report,indent=2)+'\n')
    return report


def restore_hybrid_state(zero_optimizer,directory):
    directory=Path(directory)
    report=json.loads((directory/'index.json').read_text())
    assert report['format']=='deepspeed_hybrid_adam_v1'
    backup=getattr(zero_optimizer,'backup_optimizer',None)
    if report['subgroups'] and backup is None:
        raise ValueError('Saved hybrid GPU optimizer requires the same offload/subgroup configuration')
    for item in report['subgroups']:
        param=zero_optimizer.fp32_partitioned_groups_flat[item['subgroup']]
        assert param.numel()==item['numel']
        saved=torch.load(directory/item['file'],map_location='cpu',weights_only=True,mmap=True)
        # torch AdamW uses a CPU scalar step unless capturable/fused is enabled.
        backup.state[param]={key:(value.to(param.device) if torch.is_tensor(value) and key!='step' else value)
                             for key,value in saved.items()}
    return report
