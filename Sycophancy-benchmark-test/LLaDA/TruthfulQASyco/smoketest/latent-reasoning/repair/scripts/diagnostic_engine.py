"""Controlled workspace interventions; identical answer schedule and stop rules."""
import time
import torch
import audit
from workspace import WorkspaceGenerator
from llada_elephant.inference import Generator

VARIANTS={
 'control':{'placement':'none','rounds':0,'norm_match':False},
 'fixed_after':{'placement':'after','rounds':0,'norm_match':False},
 'fixed_before':{'placement':'before','rounds':0,'norm_match':False},
 'recurrent_after':{'placement':'after','rounds':2,'norm_match':False},
 'norm_after':{'placement':'after','rounds':2,'norm_match':True},
 'recurrent_before':{'placement':'before','rounds':2,'norm_match':False},
 'norm_before':{'placement':'before','rounds':2,'norm_match':True}}


def norm_match(h,target,epsilon=1e-8):
    # Heuristic only: preserve direction and match each initial vector's L2 norm.
    f=h.float()
    return (target.float()*f/f.norm(dim=-1,keepdim=True).clamp_min(epsilon)).to(h.dtype)


class Engine(WorkspaceGenerator):
    @torch.inference_mode()
    def prepare_prefix(self,ids,variant):
        v=VARIANTS[variant]
        tokens=torch.tensor([ids],device=self.device)
        p=self.model.get_input_embeddings()(tokens)
        original=p.clone()
        if v['placement']=='none':
            return p,[],{'prompt_ranges':[[0,len(ids)]],'workspace_range':None,'assistant_marker_range':[audit.marker_start(ids),len(ids)],'answer_range':[len(ids),len(ids)+128]},None
        start=audit.marker_start(ids) if v['placement']=='before' else len(ids)
        z=self.initial[None].clone()
        initial=z.clone()
        target=z.float().norm(dim=-1,keepdim=True)
        norms=[{'round':0,'norms':target.flatten().tolist()}]
        def combine(z):
            return torch.cat([p[:,:start],z,p[:,start:]],1)
        for index in range(v['rounds']):
            captured=[]
            hook=self.model.model.transformer.ln_f.register_forward_hook(
                lambda module,args,output: captured.append(output[:,start:start+4].clone()))
            try:
                prefix=combine(z)
                self.model(inputs_embeds=prefix,attention_mask=torch.ones(prefix.shape[:2],device=self.device,dtype=torch.long),use_cache=False)
            finally:
                hook.remove()
            h=self.projection(captured[0])
            next_z=norm_match(h,target) if v['norm_match'] else h
            if not torch.isfinite(next_z).all():
                raise RuntimeError('Nonfinite workspace')
            norms.append({'round':index+1,'raw_hidden_norms':h.float().norm(dim=-1).flatten().tolist(),
                'norms':next_z.float().norm(dim=-1).flatten().tolist(),'delta_norm':(next_z.float()-z.float()).norm().item()})
            z=next_z
        assert torch.equal(p,original)
        prefix=combine(z)
        assert prefix.dtype==p.dtype==torch.bfloat16
        assert torch.equal(prefix[:,:start],p[:,:start]) and torch.equal(prefix[:,start+4:],p[:,start:])
        layout={'prompt_ranges':[[0,start],[start+4,len(ids)+4]],'workspace_range':[start,start+4],
                'assistant_marker_range':[audit.marker_start(ids)+(4 if v['placement']=='before' else 0),len(ids)+(4 if v['placement']=='before' else 0)],
                'answer_range':[len(ids)+4,len(ids)+132]}
        return prefix,norms,layout,initial

    @torch.inference_mode()
    def run_row(self,row,variant):
        torch.manual_seed(self.config['seed']);torch.cuda.manual_seed_all(self.config['seed'])
        ids=audit.smoke.encode_messages(self.tokenizer,row,self.mask_id,self.config)
        torch.cuda.synchronize(self.device);started=time.perf_counter()
        prefix,norms,layout,_=self.prepare_prefix(ids,variant)
        frozen=prefix.clone()
        length=self.config['generation_length'];block=self.config['block_length']
        blocks=length//block;steps=self.config['diffusion_steps']//blocks
        assert length==128 and block==32 and steps==32
        answer=torch.full((1,length),self.mask_id,device=self.device,dtype=torch.long)
        attention=torch.ones((1,prefix.shape[1]+length),device=self.device,dtype=torch.long)
        initial_eos=None
        for b in range(blocks):
            for step in range(steps):
                embeddings=torch.cat([prefix,self.model.get_input_embeddings()(answer)],1)
                logits=self.model(inputs_embeds=embeddings,attention_mask=attention,use_cache=False).logits.float()
                if logits.shape[:2] != answer.shape or not torch.isfinite(logits).all():
                    raise RuntimeError('Nonfinite or incorrectly sliced answer logits')
                predictions=logits.argmax(-1)
                probs=logits.softmax(-1)
                if initial_eos is None:
                    initial_eos={str(token):probs[0,0,token].item() for token in sorted(self.eos_ids)}
                confidence=probs.gather(-1,predictions.unsqueeze(-1)).squeeze(-1)
                eligible=answer==self.mask_id
                eligible[:,(b+1)*block:]=False
                confidence.masked_fill_(~eligible,-torch.inf)
                positions=confidence.topk(1,dim=-1).indices
                assert eligible.gather(1,positions).all(), 'Attempt to overwrite unmasked/out-of-block answer'
                answer.scatter_(1,positions,predictions.gather(1,positions))
        assert torch.equal(prefix,frozen), 'Prefix mutation'
        assert not (answer==self.mask_id).any(), 'Remaining masks'
        torch.cuda.synchronize(self.device)
        tokens=answer[0].cpu().tolist()
        return {'id':row['id'],'question_id':row['question_id'],'condition':row['condition'],'variant':variant,
                'messages':row['messages'],'input_sha256':audit.common.identity_hash(row),
                'input_tokens':len(ids),**audit.describe(tokens),'workspace_norms':norms,
                'initial_first_position_end_probabilities':initial_eos,'initial_first_position_end_probability':sum(initial_eos.values()),
                'layout':layout,'prefix_unchanged':True,'all_values_finite':True,
                'workspace_injected_on_all_128_passes':variant!='control','elapsed_seconds':time.perf_counter()-started,
                'peak_allocated_gib':torch.cuda.max_memory_allocated(self.device)/1024**3,'saved_at_utc':audit.common.now()}

    @torch.inference_mode()
    def equivalence(self,row):
        ids=audit.smoke.encode_messages(self.tokenizer,row,self.mask_id,self.config)
        prompt=torch.tensor([ids],device=self.device)
        x=torch.cat([prompt,torch.full((1,128),self.mask_id,device=self.device,dtype=torch.long)],1)
        embeddings=self.model.get_input_embeddings()(x)
        mask=torch.ones_like(x)
        a=self.model(x,attention_mask=mask,use_cache=False).logits.float()
        b=self.model(inputs_embeds=embeddings,attention_mask=mask,use_cache=False).logits.float()
        diff=(a-b).abs().max().item()
        torch.testing.assert_close(a,b,rtol=1e-3,atol=1e-3)
        # Force left padding to verify masked positions do not affect the answer.
        padded=torch.cat([torch.full((1,3),self.tokenizer.pad_token_id,device=self.device,dtype=torch.long),x],1)
        padded_mask=torch.cat([torch.zeros((1,3),device=self.device,dtype=mask.dtype),mask],1)
        c=self.model(padded,attention_mask=padded_mask,use_cache=False).logits.float()
        padding_diff=(a-c).abs().max().item()
        # BF16 kernels may differ slightly with sequence length; record rather than assume bitwise equality.
        padding_mean=(a-c).abs().mean().item()
        padding_argmax_agreement=(a.argmax(-1)==c.argmax(-1)).float().mean().item()
        del a,b,c
        direct=Generator.generate(self,[ids])[0]
        result=self.run_row(row,'control')
        assert direct==result['output_token_ids'], 'Disabled custom path differs from original sampler'
        capture_checks=[]
        for placement in ['after','before']:
            start=len(ids) if placement=='after' else audit.marker_start(ids)
            p=self.model.get_input_embeddings()(prompt)
            z=self.initial[None]
            combined=torch.cat([p[:,:start],z,p[:,start:]],1)
            captured=[]
            handle=self.model.model.transformer.ln_f.register_forward_hook(lambda module,args,output:captured.append(output[:,start:start+4].clone()))
            try:
                out=self.model(inputs_embeds=combined,attention_mask=torch.ones(combined.shape[:2],device=self.device),output_hidden_states=True,use_cache=False)
            finally:
                handle.remove()
            torch.testing.assert_close(captured[0],out.hidden_states[-1][:,start:start+4],rtol=0,atol=0)
            capture_checks.append(placement)
            del out,captured
        return {'id':row['id'],'embedding_max_abs_difference':diff,'disabled_original_tokens_exact':True,
                'padding_max_abs_difference':padding_diff,'padding_mean_abs_difference':padding_mean,
                'padding_argmax_agreement':padding_argmax_agreement,'public_hidden_state_matches':capture_checks},result
