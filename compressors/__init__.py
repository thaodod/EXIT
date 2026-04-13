"""Document compression implementations."""

from .base import BaseCompressor, SearchResult
from .factory import SUPPORTED_METHODS, get_compressor

_BASELINE_EXPORTS = {
    "AttnCompCompressor": ".baselines.attn_comp.compressor",
    "CompActCompressor": ".baselines.compact.compressor",
    "EXITCompressor": ".baselines.exit.compressor",
    "RefinerCompressor": ".baselines.refiner.compressor",
    "RecompAbstractiveCompressor": ".baselines.recomp_abst.compressor",
    "RecompExtractiveCompressor": ".baselines.recomp_extr.compressor",
    "LongLLMLinguaCompressor": ".baselines.longllmlingua.compressor",
    "ProvenceCompressor": ".baselines.provence.compressor",
}


def __getattr__(name):
    if name in _BASELINE_EXPORTS:
        from importlib import import_module

        module = import_module(_BASELINE_EXPORTS[name], __name__)
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    'BaseCompressor',
    'SearchResult',
    'AttnCompCompressor',
    'CompActCompressor',
    'EXITCompressor',
    'RefinerCompressor',
    'RecompAbstractiveCompressor',
    'RecompExtractiveCompressor',
    'LongLLMLinguaCompressor',
    'ProvenceCompressor',
    'SUPPORTED_METHODS',
    'get_compressor',
]
