"""VAE-only codec diagnostic. No diffusion, reasoning benchmark or training.

Four known short inputs -> posterior mean -> greedy decoder. Additional zero-code
and plain-text decoder controls help localize failures. Uses both documented stop
possibilities, retaining tokens and the stop ID. Original files stay untouched.
"""
from contextlib import nullcontext
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path

import torch

from checkpoint_runtime import ROOT,skeleton,compare_layout
from latent_smoke import write,append

CASES=[
    {'id':'plain','encoder_text':'The cat sat on the mat.','expected_text':'The cat sat on the mat.'},
    {'id':'arithmetic','encoder_text':'3 * 4 = 12.','expected_text':'3 * 4 = 12.'},
    {'id':'countdown-trace','encoder_text':'5 + 3 = 8, 8 - 2 = 6, 6 * 4 = 24.',
     'expected_text':'5 + 3 = 8, 8 - 2 = 6, 6 * 4 = 24.'},
    {'id':'full-format','encoder_text':'Using 2, 3, 4, and 5 exactly once, make 24.\n5 + 3 = 8, 8 - 2 = 6, 6 * 4 = 24.',
     'expected_text':'5 + 3 = 8, 8 - 2 = 6, 6 * 4 = 24.',
     'target_policy':'Question plus trace encoded; trace-only decoder target, following released preprocessing.'},
]


def load_vae(progress):
    from accelerate.utils import set_module_tensor_to_device
    from safetensors import safe_open
    assets=json.loads((ROOT/'result/setup/checkpoint.json').read_text())
    assert assets['checksum_verified'] and Path(assets['file']).stat().st_size==assets['bytes']
    model,t=skeleton()
    header=json.loads((ROOT/'result/audit/checkpoint_header.json').read_text())
    layout=compare_layout(model,header)
    ae=model.autoencoder;del model
    actual=ae.state_dict()
    expected={'autoencoder.'+k for k in actual}
    with safe_open(assets['file'],framework='pt',device='cpu') as f:
        assert {k for k in f.keys() if k.startswith('autoencoder.')}==expected
        for i,key in enumerate(sorted(expected)):
            value=f.get_tensor(key)
            assert torch.isfinite(value).all(),key
            local=key.removeprefix('autoencoder.')
            assert value.shape==actual[local].shape
            set_module_tensor_to_device(ae,local,'cuda:0',value=value,dtype=torch.bfloat16)
            assert torch.equal(ae.get_parameter(local).detach().cpu(),value.to(torch.bfloat16)),key
            del value
            if i%32==0 or i+1==len(expected):progress(i+1,len(expected))
    assert not any(p.is_meta for p in ae.parameters())
    for module in ae.modules():
        for key,buffer in list(module._buffers.items()):
            if buffer is not None:
                assert not buffer.is_meta
                module._buffers[key]=buffer.to('cuda:0')
        if hasattr(module,'original_inv_freq'):module.original_inv_freq=module.inv_freq
    ae.device=torch.device('cuda:0')
    ae.append_sequence=torch.arange(ae.vocab_size,ae.vocab_size+ae.mem_size,device=ae.device).unsqueeze(0)
    # All selected inputs are <1536 tokens, hence a single segment under this
    # released default. This does not recover the checkpoint's training policy.
    ae.mean_compression_rate=512
    ae.eval()
    for p in ae.parameters():p.requires_grad=False
    return ae,t,{'status':'passed','full_checkpoint_layout':layout,'loaded_vae_tensors':len(expected),
        'all_vae_weights_equal_checkpoint_after_bf16_conversion':True,
        'reasoner_weights_loaded':False,'device':'cuda:0'}


def encode(ae,t,text):
    ids=t(text,return_tensors='pt')['input_ids'].to(ae.device)
    assert ids.shape[1]<1536,'This diagnostic uses exactly one latent segment'
    full=torch.cat((ids,ae.append_sequence),dim=1)
    flags=full>=ae.vocab_size
    # Clone also permits memory replacement if a training checkpoint hook marks
    # the embedding output as a leaf requiring gradients.
    embeddings=ae.icae.get_base_model().model.embed_tokens(full).clone()
    embeddings[flags]=ae.decompress_layer(ae.memory_token_embed(full[flags]-ae.vocab_size))
    # Encoder adapters enabled, decoder adapters disabled, as in vendor methods.
    assert not any(m.disable_adapters for m in ae.icae.modules() if hasattr(m,'disable_adapters') and isinstance(m.disable_adapters,bool))
    output=ae.icae(inputs_embeds=embeddings,output_hidden_states=True,use_cache=False)
    hidden=output.hidden_states[-1][flags].reshape(1,3,4096)
    mu=ae.mean(hidden);logvar=ae.log_var(hidden)
    assert torch.isfinite(mu).all() and torch.isfinite(logvar).all()
    return mu,logvar,ids


def greedy(ae,t,embeddings,max_tokens):
    current=embeddings;cache=None;tokens=[];stop=None
    with ae.icae.disable_adapter():
        for _ in range(max_tokens):
            out=ae.icae(inputs_embeds=current,past_key_values=cache,use_cache=True)
            logits=out.logits[:,-1,:128256]
            assert torch.isfinite(logits).all()
            chosen=logits.argmax(-1);token=int(chosen.item());cache=out.past_key_values
            tokens.append(token)
            if token in (2,t.eos_token_id):stop=token;break
            current=ae.icae.get_base_model().model.embed_tokens(chosen).unsqueeze(1)
    content=tokens[:-1] if stop is not None else tokens
    text=t.decode(content,skip_special_tokens=True).strip()
    grams=[tuple(content[i:i+3]) for i in range(max(0,len(content)-2))]
    return {'decoded_text':text,'raw_decoded_text':t.decode(content,skip_special_tokens=False),
        'token_ids_including_stop':tokens,'stop_id':stop,'budget_exhausted':stop is None,
        'repeated_token_trigram_fraction':1-len(set(grams))/len(grams) if grams else 0.}


def decode(ae,t,z,max_tokens=96):
    z=z.to(ae.device,dtype=torch.bfloat16)
    marker=ae.decompress_layer(ae.memory_token_embed(torch.tensor([[3]],device=ae.device)))
    return greedy(ae,t,torch.cat((ae.decompress_layer(z),marker),dim=1),max_tokens)


def main():
    directory=ROOT/'result/vae-reconstruction'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    directory.mkdir(parents=True,exist_ok=False)
    write(ROOT/'result/vae-reconstruction/latest.json',{'directory':str(directory)})
    status=lambda **kw:write(directory/'status.json',kw)
    try:
        torch.set_num_threads(4);torch.manual_seed(42)
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        write(directory/'configuration.json',{'cases':CASES,'purpose':'Known-input codec diagnostic, not reference-free question solving.',
            'posterior':'Mean only; no VAE sampling noise','latent_shape':[1,3,128],
            'encoder_adapters':'enabled','decoder_adapters':'disabled','decoder':'greedy',
            'stop_ids':[2,128001],'max_reconstruction_tokens':96,
            'reasoner_or_diffusion_used':False,'training':False,
            'compression_assumption':'512 tokens per memory position, single segment for these short inputs; not verified original checkpoint policy.',
            'code_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (
                Path(__file__),ROOT/'scripts/checkpoint_runtime.py')},
            'limitations':['Task and input formatting of published checkpoint are incompletely documented.',
                'Failed reconstruction under this configuration does not establish failure of all possible configurations.']})
        ae,t,loaded=load_vae(lambda n,total:status(stage='loading_vae',loaded=n,total=total))
        write(directory/'loading_verification.json',loaded)
        versions={n:p._version for n,p in ae.named_parameters()}
        results=[];latents={};parity=False
        from safetensors.torch import save_file
        with torch.inference_mode():
            for i,row in enumerate(CASES):
                status(stage='reconstructing',case=row['id'],completed=len(results),total=4)
                mu,logvar,ids=encode(ae,t,row['encoder_text'])
                if i==0:
                    native_mu,native_logvar=ae._compress(ids,return_sample='parameters')
                    assert torch.equal(mu,native_mu) and torch.equal(logvar,native_logvar)
                    parity=True
                decoded=decode(ae,t,mu)
                normalized=lambda x:' '.join(x.split())
                result={**row,**decoded,'input_token_count':int(ids.numel()),
                    'whitespace_normalized_exact_match':normalized(decoded['decoded_text'])==normalized(row['expected_text']),
                    'latent_norm':float(mu.float().norm()),'logvar_min':float(logvar.min()),'logvar_max':float(logvar.max())}
                append(directory/'reconstructions.jsonl',result);results.append(result)
                latents[row['id']+'_mean']=mu.detach().cpu().contiguous()
                latents[row['id']+'_logvar']=logvar.detach().cpu().contiguous()
            status(stage='decoder_controls',completed=4,total=4)
            zero=decode(ae,t,torch.zeros((1,3,128),device=ae.device,dtype=torch.bfloat16))
            write(directory/'zero_latent_control.json',zero)
            plain=[]
            for prompt in ['The capital of France is','2 + 2 =']:
                ids=t(prompt,return_tensors='pt')['input_ids'].to(ae.device)
                emb=ae.icae.get_base_model().model.embed_tokens(ids)
                plain.append({'prompt':prompt,**greedy(ae,t,emb,24)})
            write(directory/'plain_decoder_controls.json',plain)
        save_file(latents,str(directory/'encoded_latents.safetensors'))
        assert versions=={n:p._version for n,p in ae.named_parameters()}
        report={'execution_status':'passed','cases':4,
            'exact_reconstructions':sum(r['whitespace_normalized_exact_match'] for r in results),
            'budget_exhausted':sum(r['budget_exhausted'] for r in results),
            'vendor_encoder_parity':parity,'weights_unchanged':True,'training_performed':False,
            'diffusion_used':False,'known_text_given_to_encoder':True,
            'cuda_peak_allocated_gib':torch.cuda.max_memory_allocated()/2**30,
            'reconstructions':results,'plain_decoder_controls':plain,
            'zero_latent_same_as_case_ids':[r['id'] for r in results if r['decoded_text']==zero['decoded_text']]}
        write(directory/'summary.json',report)
        status(stage='complete',completed=4,total=4)
        print(json.dumps({'directory':str(directory),'exact_reconstructions':report['exact_reconstructions'],
            'reconstructions':[{k:r[k] for k in ('id','expected_text','decoded_text','stop_id')} for r in results],
            'plain_decoder_controls':plain},indent=2),flush=True)
    except Exception:
        import traceback
        status(stage='failed',traceback=traceback.format_exc());raise


if __name__=='__main__':main()
