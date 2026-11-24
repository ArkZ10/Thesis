import re

from datasets import Dataset, DatasetDict, concatenate_datasets, load_from_disk

from dllm.utils.utils import get_default_logger, parse_spec, resolve_with_base_env

logger = get_default_logger(__name__)


def load_sft_dataset(dataset_args: str) -> DatasetDict:
    """
    Load preprocessed SFT datasets from disk.

    ``dataset_args`` supports one or more disk paths separated by ``+`` or ``|``.
    Split limits use the existing bracket syntax, e.g.
    ``/path/to/tulu3[train:20000,test:20]``.
    """
    specs = [part.strip() for part in re.split(r"[|+]", dataset_args) if part.strip()]
    if not specs:
        raise ValueError("Empty dataset_args for load_sft_dataset.")

    parts = []
    for raw in specs:
        dataset_name_or_path, limits = parse_spec(raw)
        dataset_name_or_path = resolve_with_base_env(
            dataset_name_or_path, "BASE_DATASETS_DIR"
        )
        logger.info("Loading preprocessed SFT data from %s", dataset_name_or_path)
        ds = load_from_disk(dataset_name_or_path)
        parts.append(_truncate_datasetdict(_ensure_datasetdict(ds), limits))

    if len(parts) == 1:
        return parts[0]

    merged = parts[0]
    for part in parts[1:]:
        merged = _merge_datasetdicts(merged, part)
    return merged


def load_pt_dataset(*args, **kwargs):
    raise NotImplementedError(
        "PT dataset loading was removed from this code path; only preprocessed "
        "SFT datasets are supported."
    )


def _ensure_datasetdict(ds) -> DatasetDict:
    if isinstance(ds, DatasetDict):
        return ds
    if isinstance(ds, dict):
        return DatasetDict(ds)
    if isinstance(ds, Dataset):
        return DatasetDict({"train": ds})
    return DatasetDict({"train": ds})


def _truncate_split(split_data, n: int | None):
    if n is None:
        return split_data
    total = getattr(split_data, "num_rows", None)
    if total is None:
        total = len(split_data)
    return split_data.select(range(min(n, total)))


def _truncate_datasetdict(ds: DatasetDict, limits: dict) -> DatasetDict:
    ds = _ensure_datasetdict(ds)
    numeric_limits = {
        split: limit for split, limit in limits.items() if isinstance(limit, int)
    }
    if not numeric_limits:
        return ds

    split_names = list(ds.keys())
    if len(split_names) == 1:
        base = ds[split_names[0]]
        return DatasetDict(
            {
                split: _truncate_split(base, limit)
                for split, limit in numeric_limits.items()
            }
        )

    out = {}
    for split, data in ds.items():
        out[split] = _truncate_split(data, numeric_limits.get(split))
    return DatasetDict(out)


def _merge_datasetdicts(d1: DatasetDict, d2: DatasetDict) -> DatasetDict:
    d1 = _ensure_datasetdict(d1)
    d2 = _ensure_datasetdict(d2)
    out = {}
    for split in sorted(set(d1.keys()) | set(d2.keys())):
        left = d1.get(split)
        right = d2.get(split)
        if left is None:
            out[split] = right
        elif right is None:
            out[split] = left
        else:
            out[split] = concatenate_datasets([left, right])
    return DatasetDict(out)
