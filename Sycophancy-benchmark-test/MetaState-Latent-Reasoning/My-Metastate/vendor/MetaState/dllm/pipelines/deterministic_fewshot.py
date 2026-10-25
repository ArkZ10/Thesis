"""Deterministic lm-eval few-shot sampling for process-count invariant evals."""

from __future__ import annotations

import functools
import json
import logging
import os
import random
from collections.abc import Sequence
from typing import Any


logger = logging.getLogger(__name__)

_PATCHED = False
_DEFAULT_FEWSHOT_SEED = 1234
_DEFAULT_CANONICAL_WORLD_SIZE = 2


def _env_enabled(name: str, default: bool = True) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no", "off"}


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None or value.strip() == "":
        return default
    return int(value)


def _advance_rng(
    rng: random.Random,
    docs: Sequence[Any],
    *,
    n_samples: int,
    sample_calls_per_doc: int,
) -> None:
    for _ in range(sample_calls_per_doc):
        rng.sample(docs, n_samples)


def build_virtual_fewshot_rng(
    *,
    seed: int,
    doc_id: int,
    docs: Sequence[Any],
    n_samples: int,
    sample_calls_per_doc: int,
    canonical_world_size: int = _DEFAULT_CANONICAL_WORLD_SIZE,
) -> random.Random:
    """Return RNG state matching a fixed virtual distributed few-shot stream.

    With ``canonical_world_size=2``, global doc ids are mapped to the same
    rank-local streams that lm-eval would have used in a 2-process run:
    doc 0,2,4,... are virtual rank 0 and doc 1,3,5,... are virtual rank 1.
    """

    if canonical_world_size < 1:
        raise ValueError("canonical_world_size must be >= 1")
    if doc_id < 0:
        raise ValueError("doc_id must be >= 0")
    if n_samples < 0:
        raise ValueError("n_samples must be >= 0")
    if sample_calls_per_doc < 0:
        raise ValueError("sample_calls_per_doc must be >= 0")
    if n_samples > len(docs):
        raise ValueError(
            f"Cannot sample {n_samples} few-shot docs from a pool of {len(docs)}"
        )

    rng = random.Random(seed)
    virtual_local_index = doc_id // canonical_world_size
    for _ in range(virtual_local_index):
        _advance_rng(
            rng,
            docs,
            n_samples=n_samples,
            sample_calls_per_doc=sample_calls_per_doc,
        )
    return rng


def _stable_doc_key(doc: Any) -> str:
    try:
        return json.dumps(doc, sort_keys=True, ensure_ascii=False, default=str)
    except TypeError:
        return repr(doc)


def _doc_id_by_key(task: Any) -> dict[str, int]:
    cached = getattr(task, "_metastate_doc_id_by_key", None)
    if cached is not None:
        return cached

    mapping: dict[str, int] = {}
    for idx, eval_doc in enumerate(task.eval_docs):
        mapping.setdefault(_stable_doc_key(eval_doc), idx)
    setattr(task, "_metastate_doc_id_by_key", mapping)
    return mapping


def _resolve_doc_id(task: Any, doc: Any) -> int | None:
    return _doc_id_by_key(task).get(_stable_doc_key(doc))


def _is_default_context_sampler(sampler: Any) -> bool:
    cls = sampler.__class__
    return cls.__name__ == "ContextSampler" and cls.__module__ == "lm_eval.api.samplers"


def _n_samples_for_context(task: Any, num_fewshot: int) -> int:
    config = task.config
    n_samples = int(num_fewshot)
    if getattr(config, "fewshot_split", None) == getattr(config, "test_split", None):
        n_samples += 1
    return n_samples


def _sample_calls_per_context(kwargs: dict[str, Any]) -> int:
    apply_chat_template = bool(kwargs.get("apply_chat_template", False))
    fewshot_as_multiturn = bool(kwargs.get("fewshot_as_multiturn", False))
    if apply_chat_template and not fewshot_as_multiturn:
        return 2
    return 1


def _cached_virtual_rng(
    task: Any,
    sampler: Any,
    *,
    seed: int,
    doc_id: int,
    n_samples: int,
    sample_calls_per_doc: int,
    canonical_world_size: int,
) -> tuple[random.Random, tuple[Any, ...], int]:
    docs = sampler.docs
    virtual_rank = doc_id % canonical_world_size
    local_index = doc_id // canonical_world_size
    cache_key = (
        seed,
        canonical_world_size,
        virtual_rank,
        n_samples,
        sample_calls_per_doc,
        id(docs),
    )

    cache = getattr(task, "_metastate_fewshot_rng_cache", None)
    if cache is None:
        cache = {}
        setattr(task, "_metastate_fewshot_rng_cache", cache)

    cached_index, cached_state = cache.get(cache_key, (0, random.Random(seed).getstate()))
    if cached_index > local_index:
        cached_index = 0
        rng = random.Random(seed)
    else:
        rng = random.Random()
        rng.setstate(cached_state)

    for _ in range(cached_index, local_index):
        _advance_rng(
            rng,
            docs,
            n_samples=n_samples,
            sample_calls_per_doc=sample_calls_per_doc,
        )

    cache[cache_key] = (local_index, rng.getstate())
    return rng, cache_key, local_index + 1


def _patch_set_fewshot_seed(task_cls: type) -> None:
    original = task_cls.set_fewshot_seed
    if getattr(original, "_metastate_patched", False):
        return

    @functools.wraps(original)
    def wrapped(self, seed=None):
        self._metastate_fewshot_seed = seed
        return original(self, seed=seed)

    wrapped._metastate_patched = True
    wrapped._metastate_original = original
    task_cls.set_fewshot_seed = wrapped


def _patch_configurable_task_fewshot(configurable_task_cls: type) -> None:
    original = configurable_task_cls.fewshot_context
    if getattr(original, "_metastate_patched", False):
        return

    @functools.wraps(original)
    def wrapped(self, doc, num_fewshot, *args, **kwargs):
        if not _env_enabled("METASTATE_DETERMINISTIC_FEWSHOT", default=True):
            return original(self, doc, num_fewshot, *args, **kwargs)

        if int(num_fewshot) <= 0:
            return original(self, doc, num_fewshot, *args, **kwargs)

        sampler = getattr(self, "sampler", None)
        if sampler is None or not _is_default_context_sampler(sampler):
            return original(self, doc, num_fewshot, *args, **kwargs)

        seed = getattr(self, "_metastate_fewshot_seed", _DEFAULT_FEWSHOT_SEED)
        if seed is None:
            return original(self, doc, num_fewshot, *args, **kwargs)

        doc_id = _resolve_doc_id(self, doc)
        if doc_id is None:
            return original(self, doc, num_fewshot, *args, **kwargs)

        n_samples = _n_samples_for_context(self, int(num_fewshot))
        if n_samples <= 0:
            return original(self, doc, num_fewshot, *args, **kwargs)

        sample_calls_per_doc = _sample_calls_per_context(kwargs)
        canonical_world_size = _env_int(
            "METASTATE_FEWSHOT_CANONICAL_WORLD_SIZE",
            _DEFAULT_CANONICAL_WORLD_SIZE,
        )

        rng, cache_key, next_index = _cached_virtual_rng(
            self,
            sampler,
            seed=int(seed),
            doc_id=int(doc_id),
            n_samples=n_samples,
            sample_calls_per_doc=sample_calls_per_doc,
            canonical_world_size=canonical_world_size,
        )

        old_task_rng = getattr(self, "fewshot_rnd", None)
        old_sampler_rng = sampler.rnd
        self.fewshot_rnd = rng
        sampler.rnd = rng
        try:
            result = original(self, doc, num_fewshot, *args, **kwargs)
            self._metastate_fewshot_rng_cache[cache_key] = (next_index, rng.getstate())
            return result
        finally:
            self.fewshot_rnd = old_task_rng
            sampler.rnd = old_sampler_rng

    wrapped._metastate_patched = True
    wrapped._metastate_original = original
    configurable_task_cls.fewshot_context = wrapped


def install_deterministic_fewshot_patch() -> bool:
    """Install local lm-eval patch for deterministic ConfigurableTask few-shot."""

    global _PATCHED
    if _PATCHED:
        return True
    if not _env_enabled("METASTATE_DETERMINISTIC_FEWSHOT", default=True):
        return False

    from lm_eval.api.task import ConfigurableTask, Task

    _patch_set_fewshot_seed(Task)
    _patch_configurable_task_fewshot(ConfigurableTask)
    _PATCHED = True
    logger.info(
        "Installed deterministic few-shot patch with canonical_world_size=%s",
        _env_int(
            "METASTATE_FEWSHOT_CANONICAL_WORLD_SIZE",
            _DEFAULT_CANONICAL_WORLD_SIZE,
        ),
    )
    return True
