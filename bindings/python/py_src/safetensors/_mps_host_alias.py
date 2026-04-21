# MPS host-alias fast path: when loading a safetensors file onto Apple
# Silicon MPS, bulk-preadv straight into shared MTLBuffers via
# torch.mps._host_alias_storage, bypassing CPU staging and mmap demand-
# paging.

import ctypes
import errno
import json
import os
import struct
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, NamedTuple, Optional, Tuple, Union

import torch

# Threads for parallel preadv. Four tend to saturate the SSD bandwidth.
_NUM_THREADS = 4


class _Iovec(ctypes.Structure):
    _fields_ = [
        ("iov_base", ctypes.c_void_p),
        ("iov_len", ctypes.c_size_t),
    ]


_libc_preadv = None
if sys.platform == "darwin":
    try:
        _libc = ctypes.CDLL(None, use_errno=True)
        _libc.preadv.argtypes = [
            ctypes.c_int,
            ctypes.POINTER(_Iovec),
            ctypes.c_int,
            ctypes.c_int64,  # off_t (64-bit on Darwin)
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


def _can_use_mps_host_alias(device: Union[str, int]) -> bool:
    if _libc_preadv is None:
        return False
    if not hasattr(getattr(torch, "mps", None), "_host_alias_storage"):
        return False
    try:
        return torch.device(device).type == "mps"
    except (RuntimeError, TypeError):
        return False


def _parse_safetensors_header(filename: Union[str, os.PathLike]) -> Tuple[int, dict, dict]:
    """Returns (data_start, metadata, info_by_name). info_by_name is the
    raw header dict minus __metadata__; data_start is the offset where
    tensor bytes begin in the file."""
    with open(filename, "rb") as fh:
        (header_len,) = struct.unpack("<Q", fh.read(8))
        header = json.loads(fh.read(header_len))
    metadata = header.pop("__metadata__", None) or {}
    return 8 + header_len, metadata, header


class _ReadJob(NamedTuple):
    ptr: int
    offset: int
    nbytes: int


def _load_file_mps_host_alias(
    filename: Union[str, os.PathLike],
    header: Optional[Tuple[int, dict, dict]] = None,
) -> Dict[str, torch.Tensor]:
    """Bulk-load every tensor in a safetensors file onto MPS via the
    host-alias path: allocate destinations + take aliases serially,
    then parallel-preadv the file bytes straight into the MTLBuffers."""
    from .torch import _TYPES

    if header is None:
        header = _parse_safetensors_header(filename)
    data_start, _, info_by_name = header

    # Drain any in-flight MPS work before we start writing per
    # torch.mps._host_alias_storage's documented contract.
    torch.mps.synchronize()

    result: Dict[str, torch.Tensor] = {}
    work: List[_ReadJob] = []
    for name, info in info_by_name.items():
        dtype = _TYPES.get(info["dtype"])
        if dtype is None:
            raise ValueError(f"unsupported safetensors dtype: {info['dtype']}")
        shape = info["shape"]
        start, end = info["data_offsets"]
        nbytes = end - start
        if nbytes == 0:
            result[name] = torch.empty(shape, dtype=dtype, device="mps")
            continue
        mps_t = torch.empty(shape, dtype=dtype, device="mps")
        host_storage = torch.mps._host_alias_storage(mps_t.untyped_storage())
        result[name] = mps_t
        work.append(_ReadJob(
            ptr=host_storage.data_ptr(),
            offset=data_start + start,
            nbytes=nbytes,
        ))

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

    torch.mps.synchronize()
    return result


class _MPSHostAliasSafeSlice:
    """Stand-in for the Rust ``PySafeSlice``, backed by an already-loaded
    MPS tensor. Doesn't save memory but matches the API surface."""

    __slots__ = ("_tensor",)

    def __init__(self, tensor: torch.Tensor) -> None:
        self._tensor = tensor

    def __getitem__(self, idx):
        return self._tensor[idx]

    def get_shape(self):
        return list(self._tensor.shape)

    def get_dtype(self):
        from .torch import _TYPES  # lazy
        return next(
            (s for s, dt in _TYPES.items() if dt == self._tensor.dtype),
            str(self._tensor.dtype),
        )


class _MPSHostAliasSafeOpen:
    """``safe_open``-compatible wrapper that bulk-loads a safetensors file
    onto MPS via the host-alias fast path on first ``get_tensor`` /
    ``get_slice`` call. ``keys()``/``metadata()`` don't trigger I/O.
    Eager bulk load matches full-model loaders' access pattern; for
    single-tensor MPS extraction, prefer 
    ``safe_open(device='cpu').get_tensor(name).to('mps')``."""

    __slots__ = (
        "_filename", "_device", "_data_start",
        "_metadata", "_info", "_tensors", "_lock",
    )

    def __init__(
        self,
        filename: Union[str, os.PathLike],
        device: Union[str, int],
        header: Tuple[int, dict, dict],
    ) -> None:
        self._filename = filename
        self._device = device
        self._data_start, self._metadata, self._info = header
        self._tensors: Optional[Dict[str, torch.Tensor]] = None
        self._lock = threading.Lock()

    def _ensure_loaded(self) -> Dict[str, torch.Tensor]:
        # Double-checked locking against concurrent first-load races.
        if self._tensors is None:
            with self._lock:
                if self._tensors is None:
                    self._tensors = _load_file_mps_host_alias(
                        self._filename,
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

    def get_slice(self, name: str) -> _MPSHostAliasSafeSlice:
        return _MPSHostAliasSafeSlice(self._ensure_loaded()[name])


def _maybe_mps_safe_open(
    filename: Union[str, os.PathLike],
    framework: str,
    device: Union[str, int],
):
    """Returns an ``_MPSHostAliasSafeOpen`` wrapper if the host-alias fast
    path applies, else ``None`` so the caller falls through to the Rust
    ``safe_open``. Pre-validates the header so the wrapper has no
    failure mode for unsupported dtypes."""
    from .torch import _TYPES

    if framework != "pt" or not _can_use_mps_host_alias(device):
        return None
    try:
        header = _parse_safetensors_header(filename)
    except (OSError, ValueError):
        return None
    if any(_TYPES.get(info["dtype"]) is None for info in header[2].values()):
        return None
    return _MPSHostAliasSafeOpen(filename, device, header)
