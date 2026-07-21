import importlib
import sys

_mod = importlib.import_module("computer2-PHASEOUT")
sys.modules["computer2-PHASEOUT"] = _mod

# Re-export everything for backward compatibility
for _attr in dir(_mod):
    if not _attr.startswith("_"):
        globals()[_attr] = getattr(_mod, _attr)
