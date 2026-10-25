import ast
from functools import lru_cache
from importlib import import_module
from pathlib import Path

_SUBMODULES = ("configs", "data", "models", "utils")


def _iter_public_names(module_name):
    module_path = Path(__file__).with_name(f"{module_name}.py")
    tree = ast.parse(module_path.read_text())
    module_all = None
    public_names = set()

    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if not isinstance(target, ast.Name):
                    continue
                if target.id == "__all__":
                    try:
                        values = ast.literal_eval(node.value)
                    except (SyntaxError, ValueError):
                        values = None
                    if isinstance(values, (list, tuple)):
                        module_all = [value for value in values if isinstance(value, str)]
                elif not target.id.startswith("_"):
                    public_names.add(target.id)
        elif isinstance(node, ast.AnnAssign):
            target = node.target
            if isinstance(target, ast.Name) and not target.id.startswith("_"):
                public_names.add(target.id)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if not node.name.startswith("_"):
                public_names.add(node.name)

    return module_all or sorted(public_names)


@lru_cache(maxsize=1)
def _module_exports():
    exports = {}
    for module_name in _SUBMODULES:
        for export_name in _iter_public_names(module_name):
            exports.setdefault(export_name, module_name)
    return exports


__all__ = [*_SUBMODULES, *sorted(_module_exports())]


def __getattr__(name):
    if name in _SUBMODULES:
        module = import_module(f"{__name__}.{name}")
        globals()[name] = module
        return module

    module_name = _module_exports().get(name)
    if module_name is not None:
        module = import_module(f"{__name__}.{module_name}")
        value = getattr(module, name)
        globals()[name] = value
        return value

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(list(globals()) + __all__)
