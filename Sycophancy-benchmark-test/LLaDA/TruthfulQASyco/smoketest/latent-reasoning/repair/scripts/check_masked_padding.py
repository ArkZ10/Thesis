"""Check ignored padding content and exact workspace boundaries at both placements."""
import torch
import audit
from diagnostic_engine import Engine
root=audit.OUT/'mask_regression';root.mkdir(exist_ok=True)
engine=Engine(audit.config,'cuda:1','recurrent',root)
rows=audit.common.prepared.read_jsonl(audit.OUT/'development_inputs.jsonl')
checks=[]
try:
    with torch.inference_mode():
        for row in rows:
            ids=audit.smoke.encode_messages(engine.tokenizer,row,engine.mask_id,engine.config)
            x=torch.tensor([[engine.tokenizer.pad_token_id]*3+ids+[engine.mask_id]*128],device=engine.device)
            attention=torch.ones_like(x);attention[:,:3]=0
            a=engine.model(x,attention_mask=attention,use_cache=False).logits.float()
            x[:,:3]=torch.tensor([17,781,23],device=engine.device)
            b=engine.model(x,attention_mask=attention,use_cache=False).logits.float()
            delta=(a-b).abs().max().item()
            torch.testing.assert_close(a,b,rtol=0,atol=0)
            for variant in ['fixed_after','fixed_before']:
                prefix,_,layout,_=engine.prepare_prefix(ids,variant)
                assert prefix.shape[1]==len(ids)+4
                start,end=layout['workspace_range']
                assert end-start==4 and torch.equal(prefix[:,start:end],engine.initial[None])
                marker_start,marker_end=layout['assistant_marker_range']
                marker=engine.model.get_input_embeddings()(torch.tensor([ids[audit.marker_start(ids):]],device=engine.device))
                assert torch.equal(prefix[:,marker_start:marker_end],marker)
                assert layout['answer_range']==[prefix.shape[1],prefix.shape[1]+128]
            checks.append({'id':row['id'],'masked_padding_content_max_logit_difference':delta,'workspace_and_marker_boundaries_exact':True})
    audit.common.save(root/'checks.json',{'status':'passed','checks':checks})
finally:
    engines=[engine];del engine;audit.old.release_engines(engines)
print('Masked-padding and placement checks passed')
