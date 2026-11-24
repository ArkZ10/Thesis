import ast
from functools import lru_cache
from importlib import import_module
from pathlib import Path

__all__ = [
    "base",
    "recurrent_mdlm",
    "utils",
    "BaseSampler",
    "SamplerConfig",
    "SamplerOutput",
    "RecurrentMDLMSampler",
    "RecurrentMDLMSamplerConfig",
]

_SUBMODULES = {"base", "recurrent_mdlm", "utils"}

_MODULE_EXPORTS = {
    "BaseSampler": "base",
    "SamplerConfig": "base",
    "SamplerOutput": "base",
    "RecurrentMDLMSampler": "recurrent_mdlm",
    "RecurrentMDLMSamplerConfig": "recurrent_mdlm",
}


@lru_cache(maxsize=1)
def _utils_exports():
    tree = ast.parse(Path(__file__).with_name("utils.py").read_text())
    exports = {}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and not target.id.startswith("_"):
                    exports[target.id] = "utils"
        elif isinstance(node, ast.AnnAssign):
            target = node.target
            if isinstance(target, ast.Name) and not target.id.startswith("_"):
                exports[target.id] = "utils"
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if not node.name.startswith("_"):
                exports[node.name] = "utils"
    return exports


__all__ = [*__all__, *sorted(_utils_exports())]


def __getattr__(name):
    module_name = _MODULE_EXPORTS.get(name) or _utils_exports().get(name)
    if module_name is not None:
        module = import_module(f"{__name__}.{module_name}")
        value = getattr(module, name)
        globals()[name] = value
        return value
    if name in _SUBMODULES:
        module = import_module(f"{__name__}.{name}")
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(list(globals()) + __all__)
