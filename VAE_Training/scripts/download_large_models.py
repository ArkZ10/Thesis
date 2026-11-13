"""Download and checksum-verify pinned public checkpoints for the larger VAE."""
import hashlib
import json
import os
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "120")
from huggingface_hub import HfApi, hf_hub_download

ROOT = Path(__file__).resolve().parents[1]
MODELS = {
    "Qwen/Qwen3-4B-Base": "906bfd4b4dc7f14ee4320094d8b41684abff8539",
    "Qwen/Qwen3-8B-Base": "49e3418fbbbca6ecbdf9608b4d22e5a407081db4",
}


def download(repo, revision):
    info = HfApi().model_info(repo, revision=revision, files_metadata=True)
    assert info.sha == revision
    directory = ROOT / "models" / repo.split("/")[1]
    records = []
    for remote in info.siblings:
        name = remote.rfilename
        if not (name.endswith((".json", ".safetensors", ".txt", ".md")) or name == "LICENSE"):
            continue
        print("Downloading", repo, name, flush=True)
        path = Path(hf_hub_download(repo, name, revision=revision, local_dir=directory))
        size = path.stat().st_size
        assert size == remote.size
        sha = hashlib.sha256()
        blob = hashlib.sha1(f"blob {size}\0".encode())
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                sha.update(block)
                blob.update(block)
        expected = remote.lfs.sha256 if remote.lfs else remote.blob_id
        assert (sha.hexdigest() if remote.lfs else blob.hexdigest()) == expected, name
        records.append(dict(path=str(path.relative_to(ROOT)), bytes=size, sha256=sha.hexdigest(),
                            upstream_checksum=expected, checksum_verified=True))
        print("Verified", repo, name, size, flush=True)
    report = dict(repository=repo, revision=revision, directory=str(directory),
                  verified_at_utc=datetime.now(timezone.utc).isoformat(), files=records,
                  all_checksums_verified=True, purpose="Larger VAE encoder-training smoke test")
    (ROOT / "manifests" / (repo.split("/")[1].lower().replace("-", "_") + ".json")).write_text(
        json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    with ThreadPoolExecutor(max_workers=2) as pool:
        reports = list(pool.map(lambda item: download(*item), MODELS.items()))
    print("Both checkpoints verified", flush=True)
