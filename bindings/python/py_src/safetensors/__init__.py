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


def safe_open(filename, framework, device="cpu"):
    """Opens a safetensors file lazily and returns tensors as asked.

    For ``framework='pt'`` + ``device='mps'``, eagerly bulk-loads the
    file via parallel ``preadv(2)`` straight into shared MTLBuffers
    (see ``safetensors._mps_host_alias._MPSHostAliasSafeOpen``) on first
    tensor access. All other framework/device combinations delegate to
    the underlying Rust loader unchanged."""
    if framework == "pt":
        try:
            from ._mps_host_alias import _maybe_mps_safe_open
        except ImportError:
            _maybe_mps_safe_open = None
        if _maybe_mps_safe_open is not None:
            wrapper = _maybe_mps_safe_open(filename, framework, device)
            if wrapper is not None:
                return wrapper
    return _rust_safe_open(filename, framework=framework, device=device)
