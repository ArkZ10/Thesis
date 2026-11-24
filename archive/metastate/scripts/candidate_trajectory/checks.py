"""CPU/synthetic checks, explicitly separate from real GPU verification."""
import json
import tempfile
from pathlib import Path
from .events import Events,read_events,replay,atomic_json


def check():
    import torch
    from .engine import clone,distance,save_snapshot,load_snapshot,token_hash
    with tempfile.TemporaryDirectory(prefix='trajectory-synthetic-') as tmp:
        directory=Path(tmp);events=Events(directory);original={'tokens':torch.tensor([[1,2,3]]),'memory':torch.zeros(1,4,2),'conditioning':torch.ones(1,2),'rng':{'torch_cpu':torch.get_rng_state()}}
        copied=clone(original);copied['memory'].add_(1);assert original['memory'].sum()==0
        save_snapshot(directory/'synthetic.pt',original);loaded=load_snapshot(directory/'synthetic.pt','cpu');assert torch.equal(loaded['memory'],original['memory']);assert loaded['memory'].data_ptr()!=original['memory'].data_ptr()
        assert distance(torch.zeros(2),torch.zeros(2))['zero_norm_case']=='both_zero'
        assert distance(torch.zeros(2),torch.ones(2))['cosine'] is None
        ids=[10,99,99];events.emit('trajectory_initial',synthetic=True,trajectory_id='synthetic-c0-r0',token_ids=ids,tokens_sha256=token_hash(ids),status='generating')
        assert len(read_events(events.path))==1 # flushed before completion
        events.emit('generation_progress',synthetic=True,trajectory_id='synthetic-c0-r0',changes=[{'position':1,'old_id':99,'new_id':20,'kind':'revealed'}],tokens_sha256=token_hash([10,20,99]))
        events.emit('trajectory_failed',synthetic=True,trajectory_id='synthetic-c1-r0',status='failed',error='deliberate synthetic error')
        events.emit('branch_diversity',synthetic=True,duplicate_token_groups=[[0,1]])
        assert replay(read_events(events.path))['synthetic-c0-r0']==[10,20,99]
        bad=read_events(events.path);bad[1]['changes'][0]['old_id']=17
        try:replay(bad)
        except ValueError:pass
        else:raise AssertionError('Replay did not reject corrupt delta')
        with events.path.open('ab') as f:f.write(b'{"torn":')
        Events(directory);assert len(read_events(events.path))==4;assert list(directory.glob('events.partial.*'))
        try:
            from .dashboard import HTML
            assert 'textContent' in HTML and 'innerHTML' not in HTML # rendered data never interpreted as HTML
        except AssertionError:raise
    return {'kind':'SYNTHETIC_CPU_CHECKS','passed':True,'checks':['deep clone isolation','snapshot roundtrip','zero norm handling','event flush before completion','token delta replay','corrupt delta rejected','partial final event recovery','failed generation and duplicate event retained','dashboard data rendered as text'],'browser_rendering':'NOT_CHECKED: no browser automation installed; HTTP endpoint separately checked.'}

if __name__=='__main__':print(json.dumps(check(),indent=2))
