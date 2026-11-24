"""Download only the published VAE encoder and verify math encoding.

Uses pinned HTTP byte ranges from the author's safetensors checkpoint. No
reasoning backbone is downloaded. --download extracts original BF16 tensors;
--test loads the official encoder architecture and checks encoding behavior.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import struct
import sys
import time
from urllib.request import Request, urlopen

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
REPO='mk322/LaDiR-Code-8B'
REVISION='aab72a886105d0e53b0c1c5c3a02d55a9228a7c0'
DIRECTORY=ROOT/'models/LaDiR-Code-VAE'


def write(path,value):
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,indent=2)+'\n');os.replace(temporary,path)


def ranged_response(first,last):
    # Distinct query prevents a cached header-range response serving a body range.
    url=f'https://huggingface.co/{REPO}/resolve/{REVISION}/model.safetensors?download=true&vae_range={first}-{last}'
    response=urlopen(Request(url,headers={'Range':f'bytes={first}-{last}','Accept-Encoding':'identity'}),timeout=60)
    expected=f'bytes {first}-{last}/'
    if response.status!=206 or not response.headers.get('Content-Range','').startswith(expected):
        status=response.status;content_range=response.headers.get('Content-Range')
        response.close();raise RuntimeError(f'Server did not honor exact byte range: status={status}, range={content_range}')
    return response


def download():
    from huggingface_hub import HfApi
    from safetensors import safe_open
    DIRECTORY.mkdir(parents=True,exist_ok=True)
    target=DIRECTORY/'encoder.safetensors'
    if target.exists():
        print('Encoder already downloaded:',target);return
    api=HfApi()
    info=api.model_info(REPO,revision=REVISION,files_metadata=True)
    metadata=api.parse_safetensors_file_metadata(REPO,'model.safetensors',revision=REVISION)
    selected={name:t for name,t in metadata.tensors.items() if name.startswith('vae.')}
    assert selected and all(t.dtype=='BF16' for t in selected.values())
    entries=sorted(selected.items(),key=lambda pair:pair[1].data_offsets[0])
    first=entries[0][1].data_offsets[0];last=entries[-1][1].data_offsets[1]
    cursor=first
    for name,tensor in entries:
        assert tensor.data_offsets[0]==cursor,'VAE tensor storage must be contiguous'
        cursor=tensor.data_offsets[1]
    data_bytes=last-first
    assert data_bytes==8849400832
    with ranged_response(0,7) as response: source_header=struct.unpack('<Q',response.read(8))[0]
    header={'__metadata__':{'format':'pt','source_repo':REPO,'source_revision':REVISION,
                          'component':'trained VAE encoder only','num_latents':'64','latent_dim':'2560',
                          'mask_mode':'causal','precision':'original BF16, no conversion'}}
    for name,tensor in entries:
        header[name[4:]]={'dtype':tensor.dtype,'shape':tensor.shape,
                         'data_offsets':[value-first for value in tensor.data_offsets]}
    encoded=json.dumps(header,separators=(',',':')).encode()
    encoded+=b' '*((-len(encoded))%8)
    prefix=struct.pack('<Q',len(encoded))+encoded
    partial=target.with_suffix('.safetensors.partial')
    if partial.exists():
        with partial.open('rb') as stream: assert stream.read(len(prefix))==prefix,'Partial header differs'
        downloaded=partial.stat().st_size-len(prefix)
        assert 0<=downloaded<=data_bytes
    else:
        partial.write_bytes(prefix);downloaded=0
    initial_downloaded=downloaded;start=time.monotonic();last_report=0
    def progress(stage):
        elapsed=time.monotonic()-start
        rate=(downloaded-initial_downloaded)/max(elapsed,1e-6)
        value={'stage':stage,'repository':REPO,'revision':REVISION,'downloaded_tensor_bytes':downloaded,
               'total_tensor_bytes':data_bytes,'progress_percent':100*downloaded/data_bytes,
               'mib_per_second':rate/2**20,'remaining_seconds':(data_bytes-downloaded)/rate if rate else None,
               'full_checkpoint_downloaded':False,'updated_at_utc':datetime.now(timezone.utc).isoformat()}
        write(DIRECTORY/'download-status.json',value);print(json.dumps(value),flush=True)
    progress('downloading')
    if downloaded<data_bytes:
        begin=8+source_header+first+downloaded;end=8+source_header+last-1
        with ranged_response(begin,end) as response,partial.open('ab') as stream:
            while downloaded<data_bytes:
                block=response.read(min(8*1024**2,data_bytes-downloaded))
                if not block:raise RuntimeError('Connection ended before the requested range was complete; rerun to resume')
                stream.write(block);downloaded+=len(block)
                if downloaded-last_report>=128*1024**2:
                    stream.flush();progress('downloading');last_report=downloaded
            stream.flush();os.fsync(stream.fileno())
    assert partial.stat().st_size==len(prefix)+data_bytes
    with safe_open(str(partial),framework='pt',device='cpu') as saved:
        assert set(saved.keys())=={name[4:] for name in selected}
        for name,tensor in selected.items(): assert saved.get_slice(name[4:]).get_shape()==tensor.shape
    checksum=hashlib.sha256()
    with partial.open('rb') as stream:
        for block in iter(lambda:stream.read(16*1024**2),b''):checksum.update(block)
    os.replace(partial,target)
    original=next(f for f in info.siblings if f.rfilename=='model.safetensors')
    record={'completed':True,'repository':REPO,'revision':REVISION,'source_file':'model.safetensors',
            'source_full_file_bytes':original.size,'source_full_file_lfs_sha256':original.lfs.sha256,
            'component':'vae.* encoder tensors; prefix stripped','tensor_count':len(selected),
            'tensor_bytes':data_bytes,'encoder_file':str(target),'encoder_file_sha256':checksum.hexdigest(),
            'num_latents':64,'latent_dim':2560,'backbone':'Qwen3-4B-Base','precision':'BF16 unchanged',
            'source_header_bytes':source_header,'source_tensor_data_offsets':[first,last],
            'transfer_verification':'Pinned revision, exact HTTP Content-Range and tensor shapes; extracted file SHA256 recorded. Full source file was not downloaded, so its whole-file LFS hash was not recomputed.',
            'original_vae_decoder_and_reconstruction_bridge_included':False,
            'math_reconstruction_validated':False,'llada_training_started':False}
    write(DIRECTORY/'manifest.json',record)
    write(ROOT/'manifests/pretrained_vae.json',record)
    progress('completed')


def test(device):
    import torch
    from safetensors.torch import load_file
    from transformers import AutoTokenizer
    from vae.released_encoder import VAEEncoder
    from vae.adaptive_batches import padded_batch
    target=DIRECTORY/'encoder.safetensors'
    record=json.loads((DIRECTORY/'manifest.json').read_text())
    torch.set_num_threads(4)
    torch.manual_seed(42)
    tokenizer=AutoTokenizer.from_pretrained(ROOT/'models/Qwen3-4B-Base',local_files_only=True)
    # Instantiate directly on the GPU so nonpersistent rotary buffers are real tensors.
    with torch.device(device):
        encoder=VAEEncoder(str(ROOT/'models/Qwen3-4B-Base'),num_latents=64,latent_dim=2560,mask_mode='causal')
    state=load_file(str(target),device='cpu')
    encoder.load_state_dict(state,strict=True)
    del state
    encoder=encoder.to(device=device,dtype=torch.bfloat16).eval()
    encoder.requires_grad_(False)
    # Compare actual held-out mathematical steps, including a changed number.
    source=ROOT/'data/processed/math-sentences-v1/sentences/gsm8k.dev.jsonl'
    rows=[]
    with source.open() as stream:
        for line in stream:
            row=json.loads(line)
            if len(tokenizer.encode(row['reasoning_block']))<=96:
                rows.append(row)
            if len(rows)==4:break
    texts=[row['reasoning_block'] for row in rows]
    token_rows=[{'input_ids':tokenizer.encode(text,add_special_tokens=False)+[tokenizer.eos_token_id]} for text in texts]
    ids,mask=padded_batch(token_rows,tokenizer.eos_token_id,device)
    with torch.no_grad():
        latents=encoder.encode(ids,mask)
        repeat=encoder.encode(ids,mask)
        single=[]
        for row in token_rows:
            one,valid=padded_batch([row],tokenizer.eos_token_id,device)
            single.append(encoder.encode(one,valid))
        singles=torch.cat(single)
    assert list(latents.shape)==[4,64,2560]
    assert torch.isfinite(latents).all()
    assert torch.equal(latents,repeat)
    error=(latents.float()-singles.float()).abs()
    cosine=torch.nn.functional.cosine_similarity(latents.float().flatten(1),singles.float().flatten(1),dim=1)
    bf16_report={'minimum_cosine_similarity':float(cosine.min()),
                 'maximum_absolute_error':float(error.max())}
    # BF16 kernel rounding can differ between padded batches and individual calls.
    # Use an FP32 reference with exactly the same published weights to separate
    # floating-point drift from a position/mask error. No weights are retrained.
    encoder=encoder.float()
    with torch.no_grad():
        latents=encoder.encode(ids,mask)
        repeat=encoder.encode(ids,mask)
        single=[]
        for row in token_rows:
            one,valid=padded_batch([row],tokenizer.eos_token_id,device)
            single.append(encoder.encode(one,valid))
        singles=torch.cat(single)
    assert torch.isfinite(latents).all()
    assert torch.equal(latents,repeat)
    error=(latents-singles).abs()
    cosine=torch.nn.functional.cosine_similarity(latents.flatten(1),singles.flatten(1),dim=1)
    print(json.dumps({'bf16':bf16_report,'fp32_padding_cosines':cosine.tolist(),
                      'fp32_max_absolute_error':float(error.max())}),flush=True)
    assert cosine.min()>.99999,'FP32 reference shows strong batch-padding dependence'
    assert not torch.equal(latents[0],latents[-1]),'Different inputs returned identical latents'
    output=ROOT/'outputs/pretrained-vae-check'
    output.mkdir(exist_ok=True)
    torch.save({'latents':latents.cpu(),'texts':texts,'source_problem_ids':[row['problem_id'] for row in rows],
                'source_repository':REPO,'source_revision':REVISION},output/'math-latents.pt')
    report={'completed':True,'strict_state_dict_load':True,'weights_trainable':False,
            'inference_precision':'FP32 reference; source weights remain original BF16',
            'bf16_batch_rounding':bf16_report,
            'latent_shape':list(latents.shape),'all_finite':True,'repeated_encoding_identical':True,
            'different_inputs_produce_different_latents':True,
            'single_vs_batch_max_absolute_error':float(error.max()),
            'single_vs_batch_min_cosine_similarity':float(cosine.min()),
            'encoder_gpu_peak_gib':torch.cuda.max_memory_allocated(torch.device(device))/2**30 if 'cuda' in device else None,
            'math_reconstruction_validated':False,'llada_compatibility_validated':False,
            'why_not_reconstruction':'Release contains VAE encoder; original VAE decoder interface is not among the released vae.* tensors.',
            'source_manifest':str(DIRECTORY/'manifest.json'),'examples':texts,'test_used':False}
    write(output/'summary.json',report)
    record['local_encoding_check']=str(output/'summary.json')
    write(ROOT/'manifests/pretrained_vae.json',record)
    print(json.dumps(report,indent=2))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--download',action='store_true')
    parser.add_argument('--test',action='store_true')
    parser.add_argument('--status',action='store_true')
    parser.add_argument('--quality',action='store_true',help='Run frozen math-information probes')
    parser.add_argument('--quality-status',action='store_true',help='Read frozen-probe progress')
    parser.add_argument('--quality-diagnostic',action='store_true',help='Apply saved readouts to familiar wording; no refitting')
    parser.add_argument('--device',default='cuda:0')
    args=parser.parse_args()
    if not any([args.download,args.test,args.status,args.quality,args.quality_status,args.quality_diagnostic]):parser.error('Choose an action')
    if args.quality_diagnostic:
        from vae.encoder_evaluation import familiar_wording_diagnostic
        path=ROOT/'manifests/pretrained_vae_quality.json'
        output=Path(json.loads(path.read_text())['run_directory'])
        familiar_wording_diagnostic(output)
    if args.quality_status:
        path=ROOT/'manifests/pretrained_vae_quality.json'
        print(path.read_text() if path.exists() else 'Quality evaluation not started.')
    if args.quality:
        from vae.encoder_evaluation import run_evaluation
        run_evaluation(ROOT,encoder_device=args.device)
    if args.status:
        path=DIRECTORY/'download-status.json';print(path.read_text() if path.exists() else 'Download not started.')
    if args.download:download()
    if args.test:test(args.device)


if __name__=='__main__':main()
