from importlib import import_module

__all__ = [
    "meters",
    "metrics",
    "BaseMetricsCallback",
    "OnEvaluateMetricsCallback",
    "NLLMetric",
    "PPLMetric",
]

_MODULE_EXPORTS = {
    "BaseMetricsCallback": "meters",
    "OnEvaluateMetricsCallback": "meters",
    "NLLMetric": "metrics",
    "PPLMetric": "metrics",
}


def __getattr__(name):
    if name in _MODULE_EXPORTS:
        module = import_module(f"{__name__}.{_MODULE_EXPORTS[name]}")
        value = getattr(module, name)
        globals()[name] = value
        return value
    if name in {"meters", "metrics"}:
        module = import_module(f"{__name__}.{name}")
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(list(globals()) + __all__)
