"""Three reference-free latent samples, then greedy VAE decoding. No training.

Fixed 3x128 latent region, 50 Euler steps, seed 11, noise std 2.5 from released
denoise(). No ground-truth encoding/length inference, scaling, CFG or repulsion.
This is a single-region inspection, not paper-style causal block continuation.
"""
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path

import torch

from checkpoint_runtime import ROOT,load_checkpoint
from flow_scheduler import FlowMatchEulerDiscreteScheduler,check

PROMPTS=[
    {'id':'apples','question':'There are 3 bags with 4 apples in each bag. How many apples are there?'},
    {'id':'countdown-1','question':'Using the numbers 2, 3, 4, and 5 exactly once each, make 24. You may use addition, subtraction, multiplication, division, and parentheses.'},
    {'id':'countdown-2','question':'Using the numbers 1, 3, 4, and 6 exactly once each, make 24. You may use addition, subtraction, multiplication, division, and parentheses.'},
]


def write(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.tmp')
    tmp.write_text(json.dumps(obj,indent=2,sort_keys=True,ensure_ascii=False)+'\n');tmp.replace(path)


def append(path,obj):
    import os
    with path.open('a') as f:
        f.write(json.dumps(obj,ensure_ascii=False)+'\n');f.flush();os.fsync(f.fileno())


def decode_latent(model,latent,max_tokens=128):
    """VAE inspection decoder with AE prompt and checkpoint-compatible stopping.

    Reconstruction diagnostics observed terminator 2, as used by the released
    VAE training code. Also accept the tokenizer's native EOS. Historical smoke
    records used tokenizer EOS only; those files remain unchanged.
    """
    ae=model.autoencoder
    z=latent.to('cuda:1',dtype=torch.bfloat16)
    # AE marker has its own memory_token_embed row, as in vendor tokens_to_embeddings.
    marker=ae.decompress_layer(ae.memory_token_embed(torch.tensor([[3]],device='cuda:1')))
    current=torch.cat((ae.decompress_layer(z),marker),dim=1)
    ids=[];cache=None;stopped=False;stop_id=None
    eos=model.tokenizer.eos_token_id
    with ae.icae.disable_adapter():
        for _ in range(max_tokens):
            output=ae.icae(inputs_embeds=current,past_key_values=cache,use_cache=True)
            scores=output.logits[:,-1,:128256]
            if not torch.isfinite(scores).all():raise ValueError('Nonfinite decoder logits')
            chosen=scores.argmax(-1)
            token=int(chosen.item());cache=output.past_key_values
            if token in (2,eos):
                stopped=True;stop_id=token;break
            ids.append(token)
            current=ae.icae.get_base_model().model.embed_tokens(chosen).unsqueeze(1)
    text=model.tokenizer.decode(ids,skip_special_tokens=False)
    words=text.split();trigrams=[tuple(words[i:i+3]) for i in range(max(0,len(words)-2))]
    return {'decoded_latent_text':text,'decoded_token_ids':ids,'stopped_at_tokenizer_eos':stop_id==eos,
        'stop_id':stop_id,'accepted_stop_ids':[2,eos],
        'eos_token_id':eos,'decoder_budget_exhausted':not stopped,'empty':not bool(text.strip()),
        'repeated_trigram_fraction':1-len(set(trigrams))/len(trigrams) if trigrams else 0.}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check-only',action='store_true')
    args=parser.parse_args()
    if args.check_only:print(json.dumps(check(),indent=2));return
    directory=ROOT/'result/latent-smoke'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    directory.mkdir(parents=True,exist_ok=False)
    write(ROOT/'result/latent-smoke/latest.json',{'directory':str(directory)})
    status=lambda **kw:write(directory/'status.json',kw)
    try:
        torch.set_num_threads(4);torch.manual_seed(42)
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        write(directory/'scheduler_verification.json',check())
        write(directory/'configuration.json',{'prompts':PROMPTS,'latent_shape':[1,3,128],
            'seed':11,'steps':50,'initial_noise_std':2.5,'noise_scale_basis':'Released model.py denoise() uses 2.5; this artifact lacks inference metadata.',
            'latent_scaling':'None; rescale calls are commented out in released inference.',
            'guidance_scale':1,'diversity_guidance':False,'denoising_integrator':'Euler, FP32 state and BF16 model',
            'decoder':'Frozen VAE base, adapters disabled, greedy, checkpoint terminator 2 or tokenizer EOS, 128 token cap.',
            'references_used_by_generation':False,'training':False,
            'source_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (
                Path(__file__),ROOT/'scripts/flow_scheduler.py',ROOT/'scripts/checkpoint_runtime.py')},
            'limitations':['This fixed latent region is not a verified individual reasoning step.',
                'Decoded latent text is not a separately trained final answer.',
                'Original checkpoint inference conventions are incompletely documented.',
                'Not a reproduction of paper performance or a test of saved-prefix branching.']})
        assets=json.loads((ROOT/'result/setup/checkpoint.json').read_text())
        assert assets['checksum_verified']
        def progress(n,total):status(stage='loading',loaded=n,total=total)
        model,t,load=load_checkpoint(assets['file'],progress)
        write(directory/'loading_verification.json',load)
        versions={n:p._version for n,p in model.named_parameters()}
        results=[]
        from safetensors.torch import save_file
        with torch.inference_mode():
            for row in PROMPTS:
                # The only text supplied to the reasoner is this question + newline.
                ids=t(row['question']+'\n',return_tensors='pt')['input_ids'].to('cuda:0')
                rng=torch.Generator(device='cuda:0').manual_seed(11)
                x=torch.randn((1,3,128),device='cuda:0',generator=rng,dtype=torch.float32)*2.5
                initial=x.clone()
                scheduler=FlowMatchEulerDiscreteScheduler();scheduler.set_timesteps(50,device='cuda:0')
                for i,step in enumerate(scheduler.timesteps):
                    projected=model.ae_to_latent(x.to(torch.bfloat16))
                    velocity=model.inference_step(ids,projected,step.reshape(1),infer_cfg=1)
                    x=scheduler.step(velocity,step,x).prev_sample
                    append(directory/'denoising.jsonl',{'id':row['id'],'step':i+1,'timestep':float(step),
                        'latent_norm':float(x.norm()),'velocity_norm':float(velocity.float().norm())})
                    if (i+1)%10==0:status(stage='denoising',prompt=row['id'],step=i+1,total_steps=50,completed=len(results))
                save_file({'initial':initial.cpu().contiguous(),'denoised':x.cpu().contiguous()},str(directory/f'{row["id"]}-latents.safetensors'))
                status(stage='decoding_latent',prompt=row['id'],completed=len(results))
                result={**row,'seed':11,'finite_latents':True,**decode_latent(model,x)}
                append(directory/'responses.jsonl',result);results.append(result)
        assert versions=={n:p._version for n,p in model.named_parameters()}
        write(directory/'summary.json',{'completed':len(results),'scheduler_checks':'passed',
            'all_latents_finite':all(r['finite_latents'] for r in results),
            'empty_decodes':sum(r['empty'] for r in results),
            'decoder_budget_exhausted':sum(r['decoder_budget_exhausted'] for r in results),
            'weights_unchanged':True,'training_performed':False,'ground_truth_encoded':False,
            'reasoner_denoising_calls':150,'responses':results,
            'quality_grade':'Not automatically graded; inspect decoded_latent_text.'})
        status(stage='complete',completed=len(results),total=len(PROMPTS))
        print(json.dumps({'directory':str(directory),'responses':results},indent=2),flush=True)
    except Exception:
        import traceback
        status(stage='failed',traceback=traceback.format_exc());raise


if __name__=='__main__':main()
