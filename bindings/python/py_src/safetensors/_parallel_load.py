# Unified parallel-preadv fast path for safetensors loading.
#
# A single loader core + small per-device adapters. Currently supports:
#   - CPU: preadv straight into torch CPU tensors (opt-in, see _parallel_cpu_enabled).
#   - MPS: preadv straight into shared MTLBuffers via torch.mps._host_alias_storage,
#          bypassing CPU staging and mmap demand-paging.
# CUDA would want a pinned-staging adapter with a different shape and lives
# outside this module.

import ctypes
import errno
import json
import os
import struct
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, NamedTuple, Optional, Tuple, Union

import torch

# Four threads tend to saturate SSD bandwidth in practice.
_NUM_THREADS = 4


# ----- low-level preadv -----

class _Iovec(ctypes.Structure):
    _fields_ = [
        ("iov_base", ctypes.c_void_p),
        ("iov_len", ctypes.c_size_t),
    ]


_libc_preadv = None
if sys.platform in ("darwin", "linux"):
    try:
        _libc = ctypes.CDLL(None, use_errno=True)
        _libc.preadv.argtypes = [
            ctypes.c_int,
            ctypes.POINTER(_Iovec),
            ctypes.c_int,
            ctypes.c_int64,  # off_t: 64-bit on Darwin and modern Linux
        ]
        _libc.preadv.restype = ctypes.c_int64
        _libc_preadv = _libc.preadv
    except (OSError, AttributeError):
        pass


def _preadv_full(fd: int, ptr: int, nbytes: int, offset: int) -> None:
    # Loop over partial reads / EINTR until nbytes have landed at ptr.
    iov = _Iovec()
    n = 0
    while n < nbytes:
        iov.iov_base = ptr + n
        iov.iov_len = nbytes - n
        result = _libc_preadv(fd, ctypes.byref(iov), 1, offset + n)
        if result < 0:
            err = ctypes.get_errno()
            if err == errno.EINTR:
                continue
            raise OSError(err, os.strerror(err))
        if result == 0:
            raise OSError(
                f"unexpected EOF at offset {offset + n}, "
                f"{nbytes - n} bytes still needed"
            )
        n += result


def _parse_safetensors_header(
    filename: Union[str, os.PathLike],
) -> Tuple[int, dict, dict]:
    """Returns (data_start, metadata, info_by_name)."""
    with open(filename, "rb") as fh:
        (header_len,) = struct.unpack("<Q", fh.read(8))
        header = json.loads(fh.read(header_len))
    metadata = header.pop("__metadata__", None) or {}
    return 8 + header_len, metadata, header


# ----- device adapter -----

@dataclass(frozen=True)
class _DeviceAdapter:
    """Device-specific bits for the parallel loader.

    allocate:  build a destination tensor on the target device.
    host_ptr:  (writable_host_ptr, keepalive) for that tensor. The keepalive
               is held for the duration of the preadv so any host alias
               (e.g. an MPS host-alias storage) isn't GC'd mid-read.
    pre_sync/post_sync: optional device barriers around the bulk write.
    """
    name: str
    allocate: Callable[[List[int], torch.dtype], torch.Tensor]
    host_ptr: Callable[[torch.Tensor], Tuple[int, Any]]
    pre_sync:  Callable[[], None] = field(default=lambda: None)
    post_sync: Callable[[], None] = field(default=lambda: None)


_CPU_ADAPTER = _DeviceAdapter(
    name="cpu",
    allocate=lambda shape, dtype: torch.empty(shape, dtype=dtype, device="cpu"),
    host_ptr=lambda t: (t.data_ptr(), None),
)


def _mps_host_ptr(t: torch.Tensor) -> Tuple[int, Any]:
    alias = torch.mps._host_alias_storage(t.untyped_storage())
    return alias.data_ptr(), alias


_MPS_ADAPTER = _DeviceAdapter(
    name="mps",
    allocate=lambda shape, dtype: torch.empty(shape, dtype=dtype, device="mps"),
    host_ptr=_mps_host_ptr,
    pre_sync=lambda: torch.mps.synchronize(),
    post_sync=lambda: torch.mps.synchronize(),
)


# ----- dispatch -----

def _parallel_cpu_enabled() -> bool:
    # CPU parallel load is opt-in: replacing lazy mmap with an eager bulk
    # read would regress callers that only want a few tensors out of a
    # large file. Toggle globally with the env var, or per-call via the
    # ``parallel=True`` kwarg on load_file / safe_open.
    return os.environ.get("SAFETENSORS_PARALLEL_CPU", "").lower() in ("1", "true", "yes")


def _pick_adapter(
    device: Union[str, int, "torch.device"],
    *,
    force_cpu: bool = False,
) -> Optional[_DeviceAdapter]:
    if _libc_preadv is None:
        return None
    try:
        kind = torch.device(device).type
    except (RuntimeError, TypeError):
        return None
    if kind == "cpu" and (force_cpu or _parallel_cpu_enabled()):
        return _CPU_ADAPTER
    if kind == "mps" and hasattr(getattr(torch, "mps", None), "_host_alias_storage"):
        return _MPS_ADAPTER
    return None


# ----- core loader -----

class _ReadJob(NamedTuple):
    ptr: int
    offset: int
    nbytes: int


def _load_file_parallel(
    filename: Union[str, os.PathLike],
    adapter: _DeviceAdapter,
    header: Optional[Tuple[int, dict, dict]] = None,
) -> Dict[str, torch.Tensor]:
    from .torch import _TYPES

    if header is None:
        header = _parse_safetensors_header(filename)
    data_start, _, info_by_name = header

    adapter.pre_sync()

    result: Dict[str, torch.Tensor] = {}
    work: List[_ReadJob] = []
    keepalive: list = []

    for name, info in info_by_name.items():
        dtype = _TYPES.get(info["dtype"])
        if dtype is None:
            raise ValueError(f"unsupported safetensors dtype: {info['dtype']}")
        shape = info["shape"]
        start, end = info["data_offsets"]
        nbytes = end - start
        t = adapter.allocate(shape, dtype)
        result[name] = t
        if nbytes == 0:
            continue
        ptr, alive = adapter.host_ptr(t)
        keepalive.append(alive)
        work.append(_ReadJob(ptr=ptr, offset=data_start + start, nbytes=nbytes))

    fd = os.open(filename, os.O_RDONLY)
    try:
        if len(work) > 1:
            with ThreadPoolExecutor(max_workers=_NUM_THREADS) as pool:
                for _ in pool.map(
                    lambda w: _preadv_full(fd, w.ptr, w.nbytes, w.offset),
                    work,
                ):
                    pass
        else:
            for w in work:
                _preadv_full(fd, w.ptr, w.nbytes, w.offset)
    finally:
        os.close(fd)

    adapter.post_sync()
    del keepalive
    return result


# ----- safe_open-compatible wrapper -----

class _ParallelSafeSlice:
    """Stand-in for the Rust PySafeSlice, backed by an already-loaded tensor."""

    __slots__ = ("_tensor",)

    def __init__(self, tensor: torch.Tensor) -> None:
        self._tensor = tensor

    def __getitem__(self, idx):
        return self._tensor[idx]

    def get_shape(self):
        return list(self._tensor.shape)

    def get_dtype(self):
        from .torch import _TYPES
        return next(
            (s for s, dt in _TYPES.items() if dt == self._tensor.dtype),
            str(self._tensor.dtype),
        )


class _ParallelSafeOpen:
    """safe_open-compatible wrapper that bulk-loads a safetensors file on
    first tensor access. keys()/metadata() don't trigger I/O. Eager bulk
    load matches full-model loaders; for single-tensor extraction, prefer
    the default Rust mmap path."""

    __slots__ = (
        "_filename", "_adapter", "_data_start",
        "_metadata", "_info", "_tensors", "_lock",
    )

    def __init__(
        self,
        filename: Union[str, os.PathLike],
        adapter: _DeviceAdapter,
        header: Tuple[int, dict, dict],
    ) -> None:
        self._filename = filename
        self._adapter = adapter
        self._data_start, self._metadata, self._info = header
        self._tensors: Optional[Dict[str, torch.Tensor]] = None
        self._lock = threading.Lock()

    def _ensure_loaded(self) -> Dict[str, torch.Tensor]:
        if self._tensors is None:
            with self._lock:
                if self._tensors is None:
                    self._tensors = _load_file_parallel(
                        self._filename,
                        self._adapter,
                        header=(self._data_start, self._metadata, self._info),
                    )
        return self._tensors

    def __enter__(self):
        return self

    def __exit__(self, _exc_type, _exc_value, _traceback):
        self._tensors = None
        return False

    def keys(self):
        return list(self._info.keys())

    def offset_keys(self):
        return sorted(
            self._info.keys(),
            key=lambda k: self._info[k]["data_offsets"][0],
        )

    def metadata(self):
        return self._metadata

    def get_tensor(self, name: str) -> torch.Tensor:
        return self._ensure_loaded()[name]

    def get_slice(self, name: str) -> _ParallelSafeSlice:
        return _ParallelSafeSlice(self._ensure_loaded()[name])


def _maybe_parallel_safe_open(
    filename: Union[str, os.PathLike],
    framework: str,
    device: Union[str, int],
    *,
    force_cpu: bool = False,
):
    """Returns a _ParallelSafeOpen wrapper if the fast path applies, else
    None so the caller falls through to the Rust safe_open. Pre-validates
    the header so the wrapper has no failure mode for unsupported dtypes."""
    from .torch import _TYPES

    if framework != "pt":
        return None
    adapter = _pick_adapter(device, force_cpu=force_cpu)
    if adapter is None:
        return None
    try:
        header = _parse_safetensors_header(filename)
    except (OSError, ValueError):
        return None
    if any(_TYPES.get(info["dtype"]) is None for info in header[2].values()):
        return None
    return _ParallelSafeOpen(filename, adapter, header)
