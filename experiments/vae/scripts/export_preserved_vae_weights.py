"""Stream exact FP32 master weights to portable shards; never delete a checkpoint."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from safetensors import safe_open
from safetensors.torch import save_file


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(16 * 1024**2), b''): h.update(chunk)
    return h.hexdigest()


def tensor_hash(tensor):
    assert tensor.device.type == 'cpu' and tensor.dtype == torch.float32
    return hashlib.sha256(memoryview(tensor.detach().contiguous().numpy()).cast('B')).hexdigest()


def llama_parameters(checkpoint):
    marker = json.loads((checkpoint / 'COMPLETE.json').read_text())
    model_path = checkpoint / 'zero_pp_rank_0_mp_rank_00_model_states.pt'
    optimizer_path = checkpoint / 'bf16_zero_pp_rank_0_mp_rank_00_optim_states.pt'
    for path in [model_path, optimizer_path]:
        assert file_hash(path) == marker['files'][path.name]['sha256']
    model = torch.load(model_path, map_location='cpu', mmap=True, weights_only=False)
    optimizer = torch.load(optimizer_path, map_location='cpu', mmap=True, weights_only=False)['optimizer_state_dict']
    assert model['dp_world_size'] == 1 and optimizer['partition_count'] == 1
    assert not model['shared_params'] and not model['buffer_names']
    shapes, flats = model['param_shapes'], optimizer['fp32_flat_groups']
    assert len(shapes) == len(flats)
    for group, flat in zip(shapes, flats):
        count = sum(shape.numel() for shape in group.values())
        if flat.numel() == 0:
            raw = checkpoint / 'offloaded_tensors/rank00' / (str(flat.ds_id) + '.tensor.swp')
            entry = marker['files'][str(raw.relative_to(checkpoint))]
            assert raw.stat().st_size == entry['bytes'] and file_hash(raw) == entry['sha256']
            flat = torch.from_file(str(raw), shared=False, size=count, dtype=torch.float32)
        assert flat.dtype == torch.float32 and flat.numel() >= count
        cursor = 0
        for name, shape in group.items():
            size = shape.numel()
            yield name, flat.narrow(0, cursor, size).view(shape)
            cursor += size




def export(parameters, destination, *, source, kind, shard_bytes=1024**3, resume_incomplete=False):
    destination.mkdir(parents=True, exist_ok=resume_incomplete)
    if resume_incomplete and ((destination / 'export-manifest.json').exists() or
                              (destination / 'model.safetensors.index.json').exists()):
        raise RuntimeError('Refusing to overwrite a published export')
    existing = {f.name for f in destination.glob('*.safetensors')}
    pending, pending_size, weight_map, records, shards = {}, 0, {}, {}, {}
    total_bytes = 0
    def flush():
        nonlocal pending, pending_size
        if not pending: return
        name = f'model-{len(shards):04d}.safetensors'
        if (destination / name).exists():
            if not resume_incomplete: raise FileExistsError(name)
            with safe_open(str(destination / name), framework='pt') as saved:
                assert set(saved.keys()) == set(pending), 'Existing shard has different parameters'
                for key in pending:
                    assert tensor_hash(saved.get_tensor(key)) == records[key]['sha256'], key
        else:
            temporary = destination / (name + '.tmp')
            save_file(pending, str(temporary), metadata={'format': 'pt', 'precision': 'exact FP32 optimizer masters'})
            os.replace(temporary, destination / name)
        shards[name] = {'bytes': (destination / name).stat().st_size, 'sha256': file_hash(destination / name)}
        for key in pending: weight_map[key] = name
        pending, pending_size = {}, 0
    for name, tensor in parameters:
        if name in records: raise ValueError('Duplicate parameter ' + name)
        if tensor.dtype != torch.float32: raise ValueError('FP32 masters required')
        size = tensor.numel() * 4
        if pending and pending_size + size > shard_bytes: flush()
        records[name] = {'shape': list(tensor.shape), 'dtype': 'float32', 'sha256': tensor_hash(tensor)}
        pending[name] = tensor.detach().contiguous(); pending_size += size; total_bytes += size
        if pending_size >= shard_bytes: flush()
    flush()
    assert existing.issubset(shards), 'Unrecognized existing shards'
    index = {'metadata': {'total_size': total_bytes}, 'weight_map': weight_map}
    (destination / 'model.safetensors.index.json').write_text(json.dumps(index, indent=2) + '\n')
    # Check every exported element against the source fingerprint, not just a sample.
    verified = 0
    for shard in shards:
        with safe_open(str(destination / shard), framework='pt', device='cpu') as stream:
            for name in stream.keys():
                tensor = stream.get_tensor(name)
                assert list(tensor.shape) == records[name]['shape'] and tensor_hash(tensor) == records[name]['sha256']
                verified += 1
    assert verified == len(records)
    report = {'completed': True, 'kind': kind, 'source': str(source), 'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'precision': 'FP32; exact master weights preserved, no rounding or quantization',
        'total_tensor_bytes': total_bytes, 'parameters': records, 'shards': shards,
        'all_exported_tensors_hash_verified': True, 'tensor_count': verified,
        'contains_optimizer_moments_or_rng': False, 'exact_training_resume_from_export': False,
        'original_checkpoint_deleted': False}
    (destination / 'export-manifest.json').write_text(json.dumps(report, indent=2) + '\n')
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--kind', choices=['llama'], default='llama')
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--destination', type=Path, required=True)
    parser.add_argument('--resume-incomplete', action='store_true')
    args = parser.parse_args()
    if shutil.disk_usage(args.destination.parent).free < 48 * 1024**3:
        raise RuntimeError('Insufficient export space reserve')
    parameters = llama_parameters(args.source)
    result = export(parameters, args.destination, source=args.source, kind=args.kind,
                    resume_incomplete=args.resume_incomplete)
    (args.destination / 'README.md').write_text('''# Preserved VAE weights

Every FP32 master parameter was exported and hash-verified against the original checkpoint. Cast to BF16 when loading to recover the trained live-model precision. The encoder, queries, posterior heads and decoder-input projection are retained. The original frozen decoder is loaded separately from its pinned pretrained source.

This directory preserves model weights, not Adam moments, data position or RNG. It supports evaluation and new optimizer initialization, not exact continuation of the original optimizer trajectory. The source checkpoint has NOT been deleted.
''')
    print(json.dumps({k: result[k] for k in ['completed', 'tensor_count', 'total_tensor_bytes', 'all_exported_tensors_hash_verified', 'original_checkpoint_deleted']}, indent=2))


if __name__ == '__main__': main()
