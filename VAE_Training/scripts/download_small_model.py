"""Download the public, revision-pinned Qwen3 checkpoint for the VAE diagnostic."""
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "120")
from huggingface_hub import HfApi, hf_hub_download

ROOT = Path(__file__).resolve().parents[1]
REPO = "Qwen/Qwen3-0.6B-Base"
REVISION = "da87bfb608c14b7cf20ba1ce41287e8de496c0cd"
FILES = ["config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json",
         "merges.txt", "vocab.json", "model.safetensors", "README.md", "LICENSE"]


def main():
    info = HfApi().model_info(REPO, revision=REVISION, files_metadata=True)
    assert info.sha == REVISION
    upstream = {f.rfilename: f for f in info.siblings}
    directory = ROOT / "models/Qwen3-0.6B-Base"
    records = []
    for name in FILES:
        print("Downloading", name, flush=True)
        path = Path(hf_hub_download(REPO, name, revision=REVISION, local_dir=directory))
        remote = upstream[name]
        size = path.stat().st_size
        assert size == remote.size
        sha = hashlib.sha256()
        blob = hashlib.sha1(f"blob {size}\0".encode())
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
                sha.update(block)
                blob.update(block)
        expected = remote.lfs.sha256 if remote.lfs else remote.blob_id
        actual = sha.hexdigest() if remote.lfs else blob.hexdigest()
        assert actual == expected, name
        records.append({"path": str(path.relative_to(ROOT)), "bytes": size, "sha256": sha.hexdigest(),
                        "upstream_checksum": expected, "checksum_verified": True})
        print("Verified", name, size, flush=True)
    report = {"repository": REPO, "revision": REVISION, "directory": str(directory),
              "verified_at_utc": datetime.now(timezone.utc).isoformat(), "files": records,
              "all_checksums_verified": True, "purpose": "Small-model VAE diagnostic only"}
    (ROOT / "manifests/qwen3_0_6b_base.json").write_text(json.dumps(report, indent=2) + "\n")
    print("Checkpoint ready:", directory, flush=True)


if __name__ == "__main__":
    main()
