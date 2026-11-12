"""Download revision-pinned raw datasets and verify their upstream checksums.

This prepares data only: no model loading, preprocessing, training, or API judge.
"""

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "120")

from huggingface_hub import HfApi, hf_hub_download
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[1]
DATASETS = (
    {
        "name": "dart-math-hard",
        "repository": "hkust-nlp/dart-math-hard",
        "revision": "0c2603fba274c2e83ec9121bee3067e286ced194",
        "fields": ["query", "response"],
        "files": {
            "data/train-00000-of-00002.parquet": "train",
            "data/train-00001-of-00002.parquet": "train",
        },
        "purpose": "Primary mathematical reasoning corpus; parent-level splitting still required.",
    },
    {
        "name": "gsm8k",
        "repository": "openai/gsm8k",
        "revision": "740312add88f781978c0658806c59bc2815b9866",
        "fields": ["question", "answer"],
        "files": {
            "main/train-00000-of-00001.parquet": "train",
            "main/test-00000-of-00001.parquet": "test",
        },
        "purpose": "Main subset only. Train for diagnostics; test reserved for evaluation. Source questions overlap DART-Math.",
    },
)


def now():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def verify(path, upstream):
    size = path.stat().st_size
    if upstream.size is not None and size != upstream.size:
        raise RuntimeError(f"Size mismatch: {path}")
    sha256 = hashlib.sha256()
    git_sha1 = hashlib.sha1(f"blob {size}\0".encode())
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            sha256.update(block)
            git_sha1.update(block)
    if upstream.lfs:
        expected = upstream.lfs.sha256
        actual = sha256.hexdigest()
        algorithm = "sha256"
    else:
        expected = upstream.blob_id
        actual = git_sha1.hexdigest()
        algorithm = "git-blob-sha1"
    if not expected or actual != expected:
        raise RuntimeError(f"Upstream checksum mismatch: {path}")
    return {
        "bytes": size,
        "sha256": sha256.hexdigest(),
        "upstream_checksum_algorithm": algorithm,
        "upstream_checksum": expected,
        "checksum_verified": True,
    }


def main():
    api = HfApi()
    status_path = ROOT / "manifests/download_status.json"
    report = {"started_at_utc": now(), "stage": "downloading", "datasets": []}
    write_json(status_path, report)
    try:
        for spec in DATASETS:
            local = ROOT / "data/raw" / spec["name"]
            local.mkdir(parents=True, exist_ok=True)
            info = api.dataset_info(
                spec["repository"], revision=spec["revision"], files_metadata=True
            )
            if info.sha != spec["revision"]:
                raise RuntimeError(f"Unexpected dataset revision: {spec['name']}")
            siblings = {f.rfilename: f for f in info.siblings}
            record = {
                "name": spec["name"],
                "repository": spec["repository"],
                "revision": info.sha,
                "purpose": spec["purpose"],
                "stage": "downloading",
                "files": [],
                "rows_by_split": {},
            }
            report["datasets"].append(record)
            write_json(status_path, report)
            filenames = list(spec["files"])
            filenames.extend(
                name for name in siblings
                if name == "README.md" or Path(name).name.upper() in {"LICENSE", "LICENSE.TXT", "LICENSE.MD"}
            )
            for filename in filenames:
                print(f"Downloading {spec['repository']} @ {info.sha}: {filename}", flush=True)
                path = Path(hf_hub_download(
                    repo_id=spec["repository"],
                    repo_type="dataset",
                    revision=info.sha,
                    filename=filename,
                    local_dir=str(local),
                ))
                item = {"path": str(path.relative_to(ROOT)), **verify(path, siblings[filename])}
                if filename in spec["files"]:
                    split = spec["files"][filename]
                    parquet = pq.ParquetFile(path)
                    columns = parquet.schema_arrow.names
                    if not set(spec["fields"]).issubset(columns):
                        raise RuntimeError(f"Missing expected columns in {path}: {columns}")
                    rows = parquet.metadata.num_rows
                    item.update({"split": split, "rows": rows, "columns": columns})
                    record["rows_by_split"][split] = record["rows_by_split"].get(split, 0) + rows
                    # Read every row in bounded batches to detect corrupt parquet pages.
                    scanned = 0
                    empty = {field: 0 for field in spec["fields"]}
                    for batch in parquet.iter_batches(batch_size=4096, columns=spec["fields"]):
                        scanned += batch.num_rows
                        for field in spec["fields"]:
                            empty[field] += sum(not text or not text.strip() for text in batch.column(field).to_pylist())
                    if scanned != rows:
                        raise RuntimeError(f"Parquet row count mismatch: {path}")
                    item.update({"all_rows_readable": True, "empty_or_null_fields": empty})
                    print(f"Verified {filename}: {rows:,} rows", flush=True)
                record["files"].append(item)
                write_json(status_path, report)
            record.update({"stage": "completed", "completed_at_utc": now()})
            write_json(ROOT / "manifests" / f"{spec['name']}.json", record)
            write_json(status_path, report)
        report.update({"stage": "completed", "completed_at_utc": now()})
        write_json(status_path, report)
        print(json.dumps({d["name"]: d["rows_by_split"] for d in report["datasets"]}, indent=2), flush=True)
    except Exception as error:
        report.update({"stage": "failed", "error": str(error), "failed_at_utc": now()})
        write_json(status_path, report)
        raise


if __name__ == "__main__":
    main()
