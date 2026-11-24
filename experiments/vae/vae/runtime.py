"""Shared runtime helpers for the current LLaMA training workflow."""
import hashlib
import os
from pathlib import Path


def decoder_hash(decoder):
    import torch
    digest = hashlib.sha256()
    for p in decoder.parameters():
        raw = p.detach().view(torch.uint8).reshape(-1)
        for offset in range(0, raw.numel(), 64 * 1024**2):
            digest.update(raw[offset:offset + 64 * 1024**2].cpu().numpy().tobytes())
    return digest.hexdigest()


def offload_environment(root):
    """Use the venv's local libaio without changing the shell environment."""
    prefix = Path(root) / '.venv/ladir-offload/libaio/prefix/usr'
    env = os.environ.copy()
    for key, folder in [('CPATH', prefix / 'include'),
                        ('LIBRARY_PATH', prefix / 'lib/x86_64-linux-gnu'),
                        ('LD_LIBRARY_PATH', prefix / 'lib/x86_64-linux-gnu')]:
        env[key] = str(folder) + (os.pathsep + env[key] if env.get(key) else '')
    env['CFLAGS'] = f"-I{prefix / 'include'} " + env.get('CFLAGS', '')
    env['LDFLAGS'] = f"-L{prefix / 'lib/x86_64-linux-gnu'} " + env.get('LDFLAGS', '')
    env.setdefault('DS_SKIP_CUDA_CHECK', '1')
    env.setdefault('MAX_JOBS', '2')
    env.update(CUBLAS_WORKSPACE_CONFIG=':4096:8', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
               TOKENIZERS_PARALLELISM='false', OMP_NUM_THREADS='8', WANDB_MODE='disabled',
               PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
    return env
