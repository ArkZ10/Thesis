"""Setup-stage checks only; --layout-only or --load-check. No generation.

Load check needs two free GPUs. Run after scripts/prepare_assets.py completes.
"""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys
from datetime import datetime,timezone

from checkpoint_runtime import ROOT,skeleton,compare_layout,load_checkpoint
OUT=ROOT/'result/setup'


def write(name,value):
    OUT.mkdir(parents=True,exist_ok=True)
    path=OUT/name;tmp=path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value,indent=2,sort_keys=True)+'\n');tmp.replace(path)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def environment():
    return {'python':sys.version,'executable':sys.executable,
        'packages':{n:importlib.metadata.version(n) for n in (
            'torch','transformers','peft','safetensors','accelerate','huggingface_hub','tokenizers')},
        'setup_policy':'Separate conda clone; unrelated editable dllm removed. Pinned repository transformers/peft/safetensors versions.',
        'torch_deviation':'Retain working torch 2.6.0+cu124; vendor lists 2.8.0. Compatibility checked here only for loading and forwards.',
        'code_sha256':{str(p.relative_to(ROOT)):sha(p) for p in (
            ROOT/'scripts/checkpoint_runtime.py',Path(__file__),ROOT/'scripts/prepare_assets.py')}}


def layout_check():
    import torch
    model,t=skeleton()
    h=json.loads((ROOT/'result/audit/checkpoint_header.json').read_text())
    report=compare_layout(model,h)
    assert report['initial_parameters_all_meta']
    # Confirm mask representation accepted by HF eager attention, including
    # bidirectional current-block latents and causal question positions.
    ids=torch.tensor([[t.bos_token_id,42,t.bot_token_id,t.time_token_id,t.tht_token_id,t.tht_token_id]])
    m=model.build_bidirectional_tht_mask(ids)
    assert m.shape==(1,1,6,6)
    assert m[0,0,0,1]<-1e20 and m[0,0,1,0]==0
    assert m[0,0,4,5]==0 and m[0,0,5,4]==0
    assert torch.isfinite(m).all()
    report.update(status='passed',checkpoint_not_loaded=True,
        no_pretrained_weights_downloaded_during_construction=True,
        bos_id=t.bos_token_id,eos_id=t.eos_token_id,
        single_block_additive_mask_check='passed',
        caveat='Multi-block causal prefix generation remains unimplemented.')
    write('layout_verification.json',report)
    return report


def gpu_check():
    import torch
    torch.set_num_threads(4)
    torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    assets=json.loads((OUT/'checkpoint.json').read_text())
    assert assets['checksum_verified']
    assert json.loads((OUT/'assets_status.json').read_text())['stage']=='complete'
    assert Path(assets['file']).stat().st_size==assets['bytes']
    before_source=json.loads((ROOT/'result/audit/source_manifest.json').read_text())
    for rel,digest in before_source['files_sha256'].items():
        assert sha(ROOT/'vendor/LaDiR'/rel)==digest,'Vendor changed: '+rel
    def progress(n,total):
        write('load_status.json',{'stage':'loading_tensors','loaded':n,'total':total})
    model,t,loaded=load_checkpoint(assets['file'],progress)
    write('load_status.json',{'stage':'checking_forwards','loaded':741,'total':741})
    versions={n:p._version for n,p in model.named_parameters()}
    assert not any(p.requires_grad for p in model.parameters())
    assert not any(m.training for m in model.modules())
    text='Checkpoint verification input.'
    ids=t(text,return_tensors='pt')['input_ids'].to('cuda:0')
    with torch.inference_mode():
        # Direct backbone outputs, not sampled tokens or a generated answer.
        a=model.text_llama(input_ids=ids,use_cache=False).logits
        b=model.text_llama(input_ids=ids,use_cache=False).logits
        assert torch.isfinite(a).all() and torch.equal(a,b)
        # A single velocity prediction, twice: not a denoising trajectory.
        rng=torch.Generator(device='cuda:0').manual_seed(42)
        latent=torch.randn((1,3,128),device='cuda:0',dtype=torch.bfloat16,generator=rng)
        projected=model.ae_to_latent(latent)
        timesteps=torch.tensor([500.],device='cuda:0')
        v=model.inference_step(ids,projected,timesteps,infer_cfg=1)
        v2=model.inference_step(ids,projected,timesteps,infer_cfg=1)
        assert v.shape==(1,3,128) and torch.isfinite(v).all() and torch.equal(v,v2)
        # VAE encoder with three explicit memory positions. No source answer and
        # no assumption about checkpoint-specific compression-length policy.
        ae=model.autoencoder
        vae_ids=torch.cat((ids.to('cuda:1'),ae.append_sequence),dim=1)
        embeddings=ae.icae.get_base_model().model.embed_tokens(vae_ids)
        flags=vae_ids>=ae.vocab_size
        embeddings[flags]=ae.decompress_layer(ae.memory_token_embed(vae_ids[flags]-ae.vocab_size))
        output=ae.icae(inputs_embeds=embeddings,output_hidden_states=True,use_cache=False)
        hidden=output.hidden_states[-1][flags].reshape(1,3,4096)
        mu=ae.mean(hidden);logvar=ae.log_var(hidden)
        assert torch.isfinite(mu).all() and torch.isfinite(logvar).all() and torch.isfinite(output.logits).all()
        # Check decoder/base path without choosing or generating any tokens.
        with ae.icae.disable_adapter():
            decoder=ae.icae(input_ids=ids.to('cuda:1'),use_cache=False).logits
        assert torch.isfinite(decoder).all()
    assert versions=={n:p._version for n,p in model.named_parameters()}
    report={'status':'passed','checked_at_utc':datetime.now(timezone.utc).isoformat(),
        **loaded,'checkpoint_sha256':assets['sha256'],'reasoner_repeat_bitwise_equal':True,
        'single_velocity_prediction_repeat_bitwise_equal':True,'velocity_shape':list(v.shape),
        'vae_encoder_mean_logvar_finite':True,'vae_decoder_forward_finite':True,
        'parameter_versions_unchanged_by_forwards':True,'vendor_source_unchanged':True,
        'gpu_peak_allocated_gib':{str(i):torch.cuda.max_memory_allocated(i)/2**30 for i in range(2)},
        'environment':environment(),'training_performed':False,'text_generation_performed':False,
        'denoising_trajectory_performed':False,
        'remaining':['Validate or repair the released diffusion scheduler.',
            'Determine checkpoint compression/task conventions and verify actual decoded behavior.',
            'Implement and validate completed-latent-prefix block continuation; not present in this loader.']}
    write('load_verification.json',report)
    write('load_status.json',{'stage':'complete','loaded':741,'total':741})
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    g=p.add_mutually_exclusive_group(required=True)
    g.add_argument('--layout-only',action='store_true');g.add_argument('--load-check',action='store_true')
    args=p.parse_args()
    write('environment.json',environment())
    try:
        report=layout_check() if args.layout_only else gpu_check()
        print(json.dumps(report,indent=2),flush=True)
    except Exception:
        import traceback
        write('load_status.json',{'stage':'failed','traceback':traceback.format_exc()});raise


if __name__=='__main__':main()
