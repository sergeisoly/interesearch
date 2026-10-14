"""Load a study's published recipe or a complete user-supplied replacement."""

from importlib.resources import files
import math
from pathlib import Path

import yaml


class RecipeLoader(yaml.SafeLoader):
    """YAML mappings reject repeated keys instead of silently replacing values."""


def _mapping(loader, node):
    result = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=True)
        if key in result:
            raise ValueError(f"duplicate recipe key: {key!r}")
        result[key] = loader.construct_object(value_node, deep=True)
    return result


RecipeLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _mapping)


def _read(path):
    value = yaml.load(path.read_text(), Loader=RecipeLoader)
    if not isinstance(value, dict):
        raise ValueError("a recipe must be a mapping")
    return value


def _checked(value, published, name):
    if isinstance(published, dict):
        if not isinstance(value, dict):
            raise ValueError(f"{name}: expected a mapping")
        missing, unknown = published.keys() - value.keys(), value.keys() - published.keys()
        if missing or unknown:
            raise ValueError(f"{name}: missing keys {sorted(map(str, missing))}; "
                             f"unknown keys {sorted(map(str, unknown))}")
        return {k: _checked(value[k], p, f"{name}.{k}") for k, p in published.items()}
    if isinstance(published, list):
        if not isinstance(value, list):
            raise ValueError(f"{name}: expected a list")
        return [_checked(v, published[0] if published else v, f"{name}[{i}]")
                for i, v in enumerate(value)]
    if published is None:
        valid = value is None or type(value) in (int, float)
    elif isinstance(published, float):
        valid = type(value) in (int, float)
        if valid:
            value = float(value)
    else:
        valid = type(value) is type(published)
    if not valid:
        raise ValueError(f"{name}: expected {type(published).__name__}, got {type(value).__name__}")
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"{name}: expected a finite number")
    return value


def load_recipe(package: str, path: str | Path | None = None) -> dict:
    """Read packaged ``recipe.yaml`` or a complete replacement with the same keys and value types.

    The packaged recipe is the fixed published profile. It also describes the input shape; study-specific
    constraints such as positive step counts belong at the study's entry point. Numeric optional defaults
    (for example LoRA alpha) accept null or a finite number. This function does not merge profiles or read CLI
    arguments. Call it at an entry point and pass the resolved values to computations explicitly.
    """
    published = _read(files(package).joinpath("recipe.yaml"))
    return _checked(published if path is None else _read(Path(path)), published, "recipe")
