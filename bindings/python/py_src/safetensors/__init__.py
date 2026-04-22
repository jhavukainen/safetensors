# Re-export this
from ._safetensors_rust import (  # noqa: F401
    SafetensorError,
    TensorSpec,
    __version__,
    deserialize,
    safe_open as _rust_safe_open,
    _safe_open_handle,
    serialize,
    serialize_file,
)


def safe_open(filename, framework, device="cpu", *, parallel=False):
    """Opens a safetensors file lazily and returns tensors as asked.

    For ``framework='pt'`` on MPS (always) or CPU (when ``parallel=True``
    or ``SAFETENSORS_PARALLEL_CPU=1``), eagerly bulk-loads the file via
    parallel ``preadv(2)`` straight into destination tensors
    (see ``safetensors._parallel_load._ParallelSafeOpen``) on first
    tensor access. All other framework/device combinations delegate to
    the underlying Rust loader unchanged."""
    if framework == "pt":
        try:
            from ._parallel_load import _maybe_parallel_safe_open
        except ImportError:
            _maybe_parallel_safe_open = None
        if _maybe_parallel_safe_open is not None:
            wrapper = _maybe_parallel_safe_open(
                filename, framework, device, force_cpu=parallel
            )
            if wrapper is not None:
                return wrapper
    return _rust_safe_open(filename, framework=framework, device=device)
