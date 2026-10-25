from importlib import import_module

__all__ = ["LLaDAConfig", "LLaDAModelLM"]

_MODULE_EXPORTS = {
    "LLaDAConfig": "models.configuration_llada",
    "LLaDAModelLM": "models.modeling_llada",
}


def __getattr__(name):
    if name in _MODULE_EXPORTS:
        module = import_module(f"{__name__}.{_MODULE_EXPORTS[name]}")
        value = getattr(module, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(list(globals()) + __all__)
