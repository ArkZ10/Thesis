"""Single-rank DeepSpeed 0.19.9 checkpoints with bounded disk use.

Unchanged FP32 optimizer files are hard-linked into an immutable snapshot.
The optimizer detaches shared inodes before writing. Native CPU gradient
buffers are retained until their optimizer subgroup is consumed, avoiding
redundant gradient swap files. No optimizer precision is changed.
"""
import hashlib
import json
import os
import random
import re
import shutil
import types
import uuid
from importlib.metadata import version
from pathlib import Path

import numpy as np
import torch


def _check_engine(engine):
    if version("deepspeed") != "0.19.9" or torch.distributed.get_world_size() != 1:
        raise RuntimeError("Only audited single-rank DeepSpeed 0.19.9 is supported")
    if not engine.optimizer.swap_optimizer or engine.has_moe_layers:
        raise RuntimeError("Expected a non-MoE NVMe ZeRO-3 engine")
    if type(engine.optimizer.optimizer_swapper).__name__ != "PartitionedOptimizerSwapper":
        raise RuntimeError("Pipelined swapping is not supported")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(16 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def fsync_directory(path):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w") as stream:
        json.dump(value, stream, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    fsync_directory(path.parent)


def capture_rng(sampler):
    state = np.random.get_state()
    return {"python": random.getstate(), "numpy": (state[0], torch.tensor(state[1].astype(np.int64)),
            state[2], state[3], state[4]), "torch_cpu": torch.get_rng_state(),
            "torch_cuda": torch.cuda.get_rng_state_all(), "sampler": sampler.get_state()}


def restore_rng(state, sampler):
    random.setstate(state["python"])
    name, keys, position, has_gauss, cached = state["numpy"]
    np.random.set_state((name, keys.numpy().astype(np.uint32), position, has_gauss, cached))
    torch.set_rng_state(state["torch_cpu"])
    torch.cuda.set_rng_state_all(state["torch_cuda"])
    sampler.set_state(state["sampler"])


def rng_probe(sampler):
    saved = capture_rng(sampler)
    result = {"python": random.random(), "numpy": np.random.normal(size=3).tolist(),
              "cpu": torch.rand(3).tolist(),
              "cuda": [torch.rand(3, device=f"cuda:{i}").cpu().tolist()
                       for i in range(torch.cuda.device_count())],
              "sampler": torch.randint(100000, (3,), generator=sampler).tolist()}
    restore_rng(saved, sampler)
    return result


def install_storage_policy(engine, run_directory, verify_gradient_views=False):
    _check_engine(engine)
    zero = engine.optimizer
    swapper = zero.optimizer_swapper
    run = Path(run_directory).resolve()
    base = Path(swapper.swap_folder).resolve()
    if not base.is_relative_to(run):
        raise RuntimeError("Optimizer swap files are outside this run")
    policy = {"cow_detaches": 0, "cpu_gradient_views": 0, "parameter_scratch_removed_bytes": 0}
    cpu_maps = [value for key, value in vars(zero).items() if key.endswith("__param_id_to_grad_partition")]
    if len(cpu_maps) != 1:
        raise RuntimeError("Unexpected gradient partition layout")
    cpu_map = cpu_maps[0]
    original_grad = zero._offload_grad_partition_at_boundary

    def keep_cpu_gradient(current, param, grad, gradients, offsets):
        parameter_id = current.get_param_id(param)
        group, offset, _ = current.grad_position[parameter_id]
        if not current._swappable_optimizer_subgroup(group):
            return original_grad(param, grad, gradients, offsets)
        current.norm_for_param_grads[parameter_id] = current._constant_buffered_norm2(grad)
        # The CPU accumulation buffer was already updated by partition_grads.
        # Finish that nonblocking transfer before a later CPU optimizer read.
        torch.cuda.synchronize(grad.device)
        view = cpu_map[param.ds_id].narrow(0, 0, grad.numel())
        if view.device.type != "cpu" or view.dtype != grad.dtype:
            raise RuntimeError("Unexpected CPU gradient buffer")
        if verify_gradient_views and not torch.equal(view.to(grad.device), grad):
            raise RuntimeError("CPU gradient view differs from the native gradient")
        info = current.optimizer_swapper._get_param_swap_info(current.fp32_partitioned_groups_flat[group])
        info.unswapped_gradients[offset] = view
        policy["cpu_gradient_views"] += 1

    zero._offload_grad_partition_at_boundary = types.MethodType(keep_cpu_gradient, zero)
    original_write = swapper._swap_out_optimizer_state

    def copy_on_write(current, info):
        for name in info.get_swap_paths():
            path = Path(name)
            if path.resolve().parent != base:
                raise RuntimeError("Unexpected optimizer state path")
            if path.exists() and path.stat().st_nlink > 1:
                temporary = path.with_name(path.name + ".cow-" + uuid.uuid4().hex)
                fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
                try:
                    os.ftruncate(fd, path.stat().st_size)
                finally:
                    os.close(fd)
                os.replace(temporary, path)
                policy["cow_detaches"] += 1
        return original_write(info)

    swapper._swap_out_optimizer_state = types.MethodType(copy_on_write, swapper)
    # Only cached CPU partitions have redundant BF16 files. Uncached partitions
    # must remain on disk and are explicitly preserved.
    for parameter in engine.module.parameters():
        partition = parameter.ds_tensor
        if getattr(partition, "final_location", None) != "not-nvme":
            continue
        if partition.device.type != "cpu" or partition.numel() != partition.ds_numel:
            raise RuntimeError("CPU parameter cache is incomplete")
        parameter_swapper = getattr(parameter, "nvme_swapper", None)
        if parameter_swapper is None:
            if engine._config.zero_config.offload_param.device != "cpu":
                raise RuntimeError("Missing parameter swapper outside CPU parameter offload")
            continue
        path = Path(parameter_swapper.get_path(parameter, must_exist=False))
        if path.is_file():
            if not path.resolve().is_relative_to(run) or path.is_symlink():
                raise RuntimeError("Redundant parameter scratch is outside this run")
            policy["parameter_scratch_removed_bytes"] += path.stat().st_size
            path.unlink()
    engine._linked_checkpoint_policy = policy
    engine._linked_checkpoint_run_directory = run
    return policy


def optimizer_file_manifest(engine):
    base = Path(engine.optimizer.optimizer_swapper.swap_folder)
    paths = {Path(name) for info in engine.optimizer.optimizer_swapper.swap_params_info.values()
             for name in info.get_swap_paths()}
    return {p.name: {"bytes": p.stat().st_size, "sha256": sha256(p)} for p in sorted(paths)}


def checkpoint_manifest(directory):
    directory = Path(directory)
    if any(p.is_symlink() for p in directory.rglob("*")):
        raise RuntimeError("Checkpoint symlinks are not supported")
    return {str(p.relative_to(directory)): {"bytes": p.stat().st_size, "sha256": sha256(p)}
            for p in sorted(directory.rglob("*")) if p.is_file() and p.name != "COMPLETE.json"}


def validate_checkpoint(directory):
    directory = Path(directory)
    marker = json.loads((directory / "COMPLETE.json").read_text())
    if marker["files"] != checkpoint_manifest(directory):
        raise RuntimeError("Checkpoint hash/size validation failed")
    return marker


def require_private_restore_target(engine):
    """Native copy2 must never overwrite a file still linked to a snapshot."""
    _check_engine(engine)
    paths = {Path(name) for item in engine.optimizer.optimizer_swapper.swap_params_info.values()
             for name in item.get_swap_paths()}
    if any(path.exists() and path.stat().st_nlink > 1 for path in paths):
        raise RuntimeError("Restore requires a fresh engine with private optimizer files")


def save_linked_checkpoint(engine, checkpoint_root, tag, client_state, *, retire_previous=True,
                           fail_before_commit=False):
    _check_engine(engine)
    if not hasattr(engine, "_linked_checkpoint_policy"):
        raise RuntimeError("Copy-on-write policy is not installed")
    if not re.fullmatch(r"step_[0-9]+", tag):
        raise ValueError("Invalid checkpoint tag")
    manual_boundary = getattr(engine, '_is_gradient_accumulation_boundary', None)
    if (manual_boundary is not None and not engine.is_gradient_accumulation_boundary()) or (
            manual_boundary is None and engine.micro_steps % engine.gradient_accumulation_steps()):
        raise RuntimeError("Checkpoints require a completed optimizer boundary")
    root = Path(checkpoint_root).resolve()
    if not root.is_relative_to(engine._linked_checkpoint_run_directory):
        raise RuntimeError("Checkpoint store is outside this private run")
    root.mkdir(parents=True, exist_ok=True)
    final = root / tag
    if final.exists():
        raise FileExistsError(final)
    previous = (root / "latest").read_text().strip() if (root / "latest").exists() else None
    if previous and not re.fullmatch(r"step_[0-9]+", previous):
        raise RuntimeError("Invalid previous checkpoint pointer")
    stage = root / (".building-" + uuid.uuid4().hex)
    stage.mkdir()
    state_dir = stage / "state"
    zero = engine.optimizer
    source = Path(zero.optimizer_swapper.swap_folder)
    if root.stat().st_dev != source.stat().st_dev:
        raise RuntimeError("Hard-link snapshots require the same filesystem")
    from deepspeed.runtime.checkpoint_engine import CheckpointCommitInfo
    info = CheckpointCommitInfo(tag="state", save_dir=str(stage), save_latest=False)
    zero.checkpoint_event_prologue()
    try:
        engine.checkpoint_engine.create(info)
        engine._create_checkpoint_file(str(stage), "state", False)
        engine._save_checkpoint(str(stage), "state", client_state=client_state, exclude_frozen_parameters=True)
        engine._create_zero_checkpoint_files(str(stage), "state")
        engine._save_zero_checkpoint(str(stage), "state")
        from deepspeed.runtime.engine import dp_index_to_str
        destination = state_dir / "offloaded_tensors" / ("rank" + dp_index_to_str(0))
        paths = {Path(name) for item in zero.optimizer_swapper.swap_params_info.values()
                 for name in item.get_swap_paths()}
        destination.mkdir(parents=True)
        for path in sorted(paths):
            with path.open("rb") as stream:
                os.fsync(stream.fileno())
            os.link(path, destination / path.name)
        for script in stage.glob("*.py"):
            shutil.copy2(script, state_dir / script.name)
        for path in state_dir.rglob("*"):
            if path.is_file():
                with path.open("rb") as stream:
                    os.fsync(stream.fileno())
        files = checkpoint_manifest(state_dir)
        if fail_before_commit:
            raise RuntimeError("Injected checkpoint publication failure")
        atomic_json(state_dir / "COMPLETE.json", {"tag": tag, "global_steps": engine.global_steps,
                                                  "files": files, "hard_link_snapshot": True})
        for path in state_dir.rglob("*"):
            if path.is_dir():
                fsync_directory(path)
        fsync_directory(state_dir)
        engine.checkpoint_engine.commit(info)
        os.rename(state_dir, final)
        fsync_directory(root)
        temporary = root / ".latest.tmp"
        with temporary.open("w") as stream:
            stream.write(tag + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, root / "latest")
        fsync_directory(root)
        if retire_previous and previous and previous != tag:
            old = root / previous
            if not old.is_dir() or old.is_symlink() or not (old / "COMPLETE.json").is_file():
                raise RuntimeError("Refusing to retire an unrecognized checkpoint")
            shutil.rmtree(old)
            fsync_directory(root)
        return final
    finally:
        zero.checkpoint_event_epilogue()
        shutil.rmtree(stage, ignore_errors=True)
