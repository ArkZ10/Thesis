"""Step the existing no-cache recurrent sampler with explicit snapshots and RNG.

Boundary semantics: a forward reads T[v], previous memory and conditioning;
Mixer/Updater produce new memory FROM T[v]; sampling then produces T[v+1].
Thus the returned memory has NOT processed the newly sampled tokens.
"""
import copy
from contextlib import contextmanager
import hashlib
import json
import random
import time
from pathlib import Path
import numpy as np
import torch


def clone(value, device=None):
    if isinstance(value,torch.Tensor):return value.detach().to(device or value.device).clone()
    if isinstance(value,dict):return {k:clone(v,device) for k,v in value.items()}
    if isinstance(value,list):return [clone(v,device) for v in value]
    if isinstance(value,tuple):return tuple(clone(v,device) for v in value)
    return copy.deepcopy(value)


def seeded_rng(seed, device):
    cpu=torch.Generator(device='cpu').manual_seed(seed)
    gpu=torch.Generator(device=device).manual_seed(seed)
    numpy_state=np.random.RandomState(seed).get_state()
    return {'torch_cpu':cpu.get_state(),'torch_cuda':gpu.get_state().cpu(),'python':random.Random(seed).getstate(),'numpy':(numpy_state[0],numpy_state[1].tolist(),*numpy_state[2:]),'seed':seed}


@contextmanager
def use_rng(state, device):
    py=random.getstate();np_state=np.random.get_state()
    with torch.random.fork_rng(devices=[device]):
        torch.set_rng_state(state['torch_cpu'].cpu());torch.cuda.set_rng_state(state['torch_cuda'].cpu(),device)
        random.setstate(state['python']);n=state['numpy'];np.random.set_state((n[0],np.array(n[1],dtype=np.uint32),*n[2:]))
        try:yield
        finally:
            state['torch_cpu']=torch.get_rng_state().clone();state['torch_cuda']=torch.cuda.get_rng_state(device).cpu().clone();state['python']=random.getstate()
            n=np.random.get_state();state['numpy']=(n[0],n[1].tolist(),*n[2:]);random.setstate(py);np.random.set_state(np_state)


def token_hash(ids):return hashlib.sha256(json.dumps(ids,separators=(',',':')).encode()).hexdigest()


def distance(a,b):
    a=a.detach().cpu().double().reshape(-1);b=b.detach().cpu().double().reshape(-1)
    an=float(a.norm());bn=float(b.norm());delta=float((a-b).norm());den=.5*(an+bn)
    return {'frobenius':delta,'normalized_frobenius':delta/den if den else 0.,'normalization':'||A-B||F / (0.5*(||A||F+||B||F)); both-zero = 0','cosine':float(torch.dot(a,b)/(an*bn)) if an and bn else None,'zero_norm_case':'both_zero' if not an and not bn else 'one_zero' if not an or not bn else None}


def save_snapshot(path,snapshot):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);temp=path.with_suffix('.tmp')
    torch.save(clone(snapshot,'cpu'),temp);temp.replace(path)


def load_snapshot(path,device):
    value=torch.load(path,map_location='cpu',weights_only=True)
    # RNG byte tensors must stay on CPU; sequence/state tensors move to GPU.
    for k in ('tokens','prompt_mask','generation_mask','memory','conditioning','quotas'):
        if value.get(k) is not None:value[k]=value[k].to(device).clone()
    return value


class Engine:
    def __init__(self,config,checkpoint,events):
        from types import SimpleNamespace
        from dllm.utils.models import get_model,get_tokenizer
        from safetensors.torch import load_file
        self.config=config;self.events=events;self.device=torch.device('cuda:0');self.calls=0;self.max_calls=config['max_backbone_calls'];self.observer_seconds=0.
        checkpoint=Path(checkpoint)
        metadata=json.loads((checkpoint/'recurrent_checkpoint_meta.json').read_text())
        assert metadata['model_type']=='recurrent_llada' and metadata['is_lightweight_checkpoint']
        assert metadata['backbone_name_or_path']==config['backbone_path']
        args=SimpleNamespace(model_name_or_path=str(checkpoint),dtype='bfloat16',attn_implementation=None)
        self.model=get_model(args,is_recurrent=True,training=False).eval();self.tokenizer=get_tokenizer(args);self.tokenizer.pad_token=self.tokenizer.eos_token
        stored=load_file(str(checkpoint/'recurrent_modules.safetensors'))
        actual={f'{m}.{k}':v for m in ('time_conditioner','mixer','updater','injector') for k,v in getattr(self.model,m).state_dict().items()}
        assert set(stored)==set(actual),'Missing or unexpected recurrent checkpoint keys'
        for k,p in actual.items():
            assert stored[k].shape==p.shape and stored[k].dtype==p.dtype and torch.equal(stored[k],p.detach().cpu()),k
            assert torch.isfinite(p).all(),k
        assert not self.model.training and all(not p.requires_grad for p in self.model.parameters())
        assert not any(m.training for m in self.model.modules() if isinstance(m,torch.nn.Dropout))
        self.mask=self.tokenizer.mask_token_id;self.stops={self.tokenizer.eos_token_id,self.tokenizer.convert_tokens_to_ids('<|eot_id|>')}
        self.events.emit('prerequisites_verified',recurrent_state_entries=len(stored),weights_exact=True,dropout_disabled=True,caching=False,backbone_path=config['backbone_path'],checkpoint=str(checkpoint))
    def _forward(self,s,bootstrap=False):
        if self.calls>=self.max_calls:raise RuntimeError('Tiny-run backbone-call cap reached')
        v=s['token_version'];t=torch.tensor([1.] if bootstrap else [float((s['tokens']==self.mask).logical_and(s['generation_mask']).sum())/self.config['generation_length']],device=self.device)
        if not bootstrap:t=t.clamp(1e-3,1.-1e-3) # Same RecurrentSamplerMixin schedule below replaces this value.
        if not bootstrap:
            from dllm.core.samplers.utils import RecurrentSamplerMixin
            # Existing sampler maps actual mask ratio through the default scheduler.
            t=self.schedule._compute_timestep_from_mask(s['tokens'],self.mask,s['generation_mask'])
        # Opt-in selective writes; existing states default to the native all-slot path.
        from .selective_memory import forward_with_memory_action
        out=forward_with_memory_action(self.model,input_ids=s['tokens'],attention_mask=torch.ones_like(s['tokens']),state=s['memory'],past_cond=s['conditioning'],t=t,use_cache=False,return_dict=True,action=s.get('memory_update_action','all'))
        self.calls+=1;s['trajectory_calls']+=1
        assert torch.isfinite(out.new_state).all() and torch.isfinite(out.logits).all()
        previous=s['memory'];s['memory']=out.new_state.detach();s['conditioning']=out.past_cond.detach();s['memory_source_token_version']=v
        diag={'memory_norm':float(out.new_state.float().norm()),'memory_change_frobenius':float((out.new_state.float()-previous.float()).norm()) if previous is not None else None,'memory_change_normalized':float((out.new_state.float()-previous.float()).norm()/previous.float().norm().clamp_min(1e-12)) if previous is not None else None,'memory_source_token_version':v}
        dn=getattr(out,'delta_norm',None);en=getattr(out,'embedding_norm',None)
        diag['injection_ratio']=float(dn.float().mean()/en.float().mean()) if dn is not None and en is not None and float(en.float().mean()) else None
        return out.logits,diag
    def initialize(self,messages,seed,trajectory_id):
        from dllm.core.samplers.recurrent_mdlm import RecurrentMDLMSampler
        self.schedule=RecurrentMDLMSampler(model=self.model,tokenizer=self.tokenizer)
        ids=self.tokenizer.apply_chat_template(messages,tokenize=True,add_generation_prompt=True);n=self.config['generation_length']
        assert self.mask not in ids and len(ids)+n<=4096
        tokens=torch.tensor([ids+[self.mask]*n],dtype=torch.long,device=self.device)
        s={'tokens':tokens,'prompt_length':len(ids),'generation_length':n,'prompt_mask':torch.arange(tokens.size(1),device=self.device)[None,:]<len(ids),'generation_mask':tokens==self.mask,'memory':None,'conditioning':None,'block':0,'iteration_in_block':0,'completed_steps':0,'quotas':None,'token_version':0,'memory_source_token_version':None,'trajectory_calls':0,'rng':seeded_rng(seed,self.device),'configuration':copy.deepcopy(self.config),'trajectory_id':trajectory_id,'remaining_decoding_iterations':self.config['steps']}
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16),use_rng(s['rng'],self.device):
            _,diag=self._forward(s,bootstrap=True)
        s['bootstrap_diagnostics']=diag
        return s
    def rendered(self,s):
        return self.tokenizer.decode(s['tokens'][0,s['prompt_length']:].tolist(),skip_special_tokens=False).replace(self.tokenizer.mask_token,'[MASK]')
    def start_event(self,s,context):
        ids=s['tokens'][0].tolist();self.events.emit('trajectory_initial',**context,trajectory_id=s['trajectory_id'],token_ids=ids,tokens_sha256=token_hash(ids),prompt_length=s['prompt_length'],mask_id=self.mask,generated_text=self.rendered(s),completed_steps=s['completed_steps'],memory_source_token_version=s['memory_source_token_version'],status='generating',actual_backbone_calls=self.calls)
        return ids
    def advance(self,s,steps=None,context=None,monitor=True):
        from dllm.core.samplers.utils import get_num_transfer_tokens_uniform,get_transfer_index
        if not hasattr(self,'schedule'):
            from dllm.core.samplers.recurrent_mdlm import RecurrentMDLMSampler
            self.schedule=RecurrentMDLMSampler(model=self.model,tokenizer=self.tokenizer)
        context=context or {};last=self.start_event(s,context) if monitor else None
        budget=self.config['steps'];per=budget//(self.config['generation_length']//self.config['block_size']);target=min(budget,s['completed_steps']+(steps if steps is not None else budget))
        start=time.perf_counter();s['last_diagnostics']=s.get('bootstrap_diagnostics',{})
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16),use_rng(s['rng'],self.device):
            while s['completed_steps']<target and s['block']<self.config['generation_length']//self.config['block_size']:
                b=s['block'];left=s['prompt_length']+b*self.config['block_size'];right=left+self.config['block_size']
                if s['quotas'] is None:s['quotas']=get_num_transfer_tokens_uniform(s['tokens'][:,left:right]==self.mask,per)
                allowed=s['tokens']==self.mask;allowed[:,right:]=False
                if not allowed.any() or s['iteration_in_block']>=per:
                    s['block']+=1;s['iteration_in_block']=0;s['quotas']=None;continue
                logits,diag=self._forward(s)
                proposed,transfer=get_transfer_index(logits=logits,temperature=self.config['temperature'],remasking='low_confidence',mask_index=allowed,x=s['tokens'],num_transfer_tokens=s['quotas'][:,s['iteration_in_block']],threshold=None,factor=None,suppress_token_ids=None)
                s['tokens']=torch.where(transfer,proposed,s['tokens']);s['completed_steps']+=1;s['token_version']+=1;s['iteration_in_block']+=1;s['remaining_decoding_iterations']=budget-s['completed_steps'];s['last_diagnostics']=diag
                if s['iteration_in_block']==per:s['block']+=1;s['iteration_in_block']=0;s['quotas']=None
                if monitor and (s['completed_steps']%self.config['event_every']==0 or s['completed_steps']==target):
                    observed=time.perf_counter();ids=s['tokens'][0].tolist();changes=[{'position':i,'old_id':old,'new_id':new,'kind':'revealed' if old==self.mask else 'revised','new_text':self.tokenizer.decode([new],skip_special_tokens=False)} for i,(old,new) in enumerate(zip(last,ids)) if old!=new];last=ids
                    full={'token_ids':ids} if s['completed_steps']%self.config['token_snapshot_every']==0 else {}
                    self.events.emit('generation_progress',**context,trajectory_id=s['trajectory_id'],changes=changes,tokens_sha256=token_hash(ids),generated_text=self.rendered(s),block=s['block'],iteration_in_block=s['iteration_in_block'],completed_steps=s['completed_steps'],remaining_budget=s['remaining_decoding_iterations'],remaining_masks=int((s['tokens']==self.mask).logical_and(s['generation_mask']).sum()),actual_backbone_calls=self.calls,trajectory_calls=s['trajectory_calls'],memory_source_token_version=s['memory_source_token_version'],token_version=s['token_version'],elapsed_seconds=time.perf_counter()-start,gpu_allocated_bytes=torch.cuda.memory_allocated(),gpu_peak_bytes=torch.cuda.max_memory_allocated(),diagnostics=diag if s['completed_steps']%self.config['diagnostic_every']==0 or s['completed_steps']==target else None,**full)
                    self.observer_seconds+=time.perf_counter()-observed
        return s
    def finish(self,s,context,reference=None):
        tokens=s['tokens'][0,s['prompt_length']:].tolist();end=next((i for i,v in enumerate(tokens) if v in self.stops),len(tokens));raw=self.tokenizer.decode(tokens[:end],skip_special_tokens=True).strip()
        import re
        markers=list(re.finditer(r'(?m)^Answer:\s*',raw));answer=raw[markers[0].end():].strip() if len(markers)==1 else raw if not markers else None
        result={**context,'trajectory_id':s['trajectory_id'],'raw_response':raw,'extracted_final_answer':answer,'answer_extraction':'single_answer_marker' if len(markers)==1 else 'entire_final_only_response' if not markers else 'ambiguous_answer_markers','empty':not bool(raw),'invalid':not bool(answer),'residual_mask_tokens':tokens.count(self.mask),'budget_exhausted_without_stop':end==len(tokens),'generated_tokens':sum(t!=self.mask for t in tokens),'generation_canvas_tokens':len(tokens),'visible_tokens_before_stop':end,'repeated_trigram_fraction':0.,'correctness':'NOT_GRADED','false_belief_endorsement':'NOT_GRADED','trajectory_calls_including_shared_history':s['trajectory_calls'],'memory_source_token_version':s['memory_source_token_version'],'token_version':s['token_version'],'tokens_sha256':token_hash(s['tokens'][0].tolist()),'memory_intervention':s.get('intervention'),'temperature':self.config['temperature']}
        words=raw.split();grams=[tuple(words[i:i+3]) for i in range(max(0,len(words)-2))]
        result['repeated_trigram_fraction']=1-len(set(grams))/len(grams) if grams else 0.
        self.events.emit('trajectory_completed',**result,status='completed',actual_backbone_calls=self.calls,grading_status='not_run_no_paid_calls_authorized')
        return result
