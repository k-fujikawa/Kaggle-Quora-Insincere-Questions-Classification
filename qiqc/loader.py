import importlib
from pathlib import Path


def load_module(filename):
    assert isinstance(filename, Path)
    name = filename.stem
    spec = importlib.util.spec_from_file_location(name, filename)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod
