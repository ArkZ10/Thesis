"""Download pinned author weights and byte-identical base config/tokenizer files.

Usage: .env/bin/python scripts/prepare_assets.py
No model imports, execution, training, or generation. Downloads are resumable.
"""
import hashlib
import json
import os
from pathlib import Path
import shutil

os.environ.setdefault('HF_XET_CHUNK_CACHE_SIZE_BYTES','0')
os.environ.setdefault('HF_HUB_DOWNLOAD_TIMEOUT','120')
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'result/setup'
REPO='mk322/LaDiR-CD4'
REVISION='b8e7cc3a4f51d24083a55e45a9dfd20d9db76dbe'
WEIGHT_SHA='5db5b415d6da3754231a195c974d5273e78337a5856295836608ab04a0ededbd'
SIZE=51210383456
BASE='meta-llama/Llama-3.1-8B'
BASE_REV='d04e592bb4f6aa9cfee91e2e20afa771667e1d4b'
MIRROR='NousResearch/Meta-Llama-3.1-8B'
MIRROR_REV='1f47e50cdbe801ad8a5174156ec3a0655108fb9f'
FILES=['config.json','tokenizer.json','tokenizer_config.json','special_tokens_map.json']


def write(name,value):
    OUT.mkdir(parents=True,exist_ok=True)
    path=OUT/name;tmp=path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value,indent=2,sort_keys=True)+'\n');tmp.replace(path)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(16*1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def main():
    from huggingface_hub import HfApi,hf_hub_download
    write('assets_status.json',{'stage':'checking_base_files'})
    try:
        api=HfApi()
        original=api.model_info(BASE,revision=BASE_REV,files_metadata=True)
        mirror=api.model_info(MIRROR,revision=MIRROR_REV,files_metadata=True)
        a={s.rfilename:s for s in original.siblings}
        b={s.rfilename:s for s in mirror.siblings}
        matches={}
        for name in FILES:
            assert a[name].blob_id==b[name].blob_id, f'Mirror differs from original: {name}'
            path=Path(hf_hub_download(MIRROR,name,revision=MIRROR_REV,
                local_dir=str(ROOT/'assets/llama-3.1-8b'),token=False))
            data=path.read_bytes()
            git_hash=hashlib.sha1(f'blob {len(data)}\0'.encode()+data).hexdigest()
            assert git_hash==a[name].blob_id,name
            matches[name]={'git_blob':git_hash,'sha256':sha(path),'size':len(data)}
        write('base_files.json',{'source':BASE,'source_revision':BASE_REV,'download_mirror':MIRROR,
            'mirror_revision':MIRROR_REV,'exact_git_blob_matches':matches,'base_weights_downloaded':False})
        destination=ROOT/'assets/LaDiR-CD4/cd4.safetensors'
        if not destination.exists():
            free=shutil.disk_usage(ROOT).free
            incomplete=sum(p.stat().st_size for p in (ROOT/'assets/LaDiR-CD4').rglob('*.incomplete')) if (ROOT/'assets/LaDiR-CD4').exists() else 0
            assert free+incomplete>SIZE+8*1024**3,'Insufficient disk space for checkpoint plus headroom'
            write('assets_status.json',{'stage':'downloading_checkpoint','expected_bytes':SIZE})
            destination=Path(hf_hub_download(REPO,'cd4.safetensors',revision=REVISION,
                local_dir=str(ROOT/'assets/LaDiR-CD4'),token=False))
        assert destination.stat().st_size==SIZE,'Checkpoint size mismatch'
        write('assets_status.json',{'stage':'verifying_checkpoint_sha256','expected_bytes':SIZE})
        digest=sha(destination);assert digest==WEIGHT_SHA,'Checkpoint checksum mismatch'
        write('checkpoint.json',{'repository':REPO,'revision':REVISION,'file':str(destination),
            'bytes':SIZE,'sha256':digest,'checksum_verified':True})
        write('assets_status.json',{'stage':'complete','checkpoint_sha256_verified':True})
        print('ASSET_PREPARATION_PASSED',flush=True)
    except Exception:
        import traceback
        write('assets_status.json',{'stage':'failed','traceback':traceback.format_exc()});raise


if __name__=='__main__':main()
