from __future__ import annotations

import os
from enum import Enum
from typing import Any, Optional

import numpy as np


class BackendType(Enum):
    CPU = "cpu"
    CUDA = "cuda"
    CUPY = "cupy"


class Backend:
    _instance: Optional["Backend"] = None
    _backend_type: BackendType
    _xp: Any
    _linalg: Any
    _fft: Any
    _random: Any

    def __new__(cls, backend_type: Optional[BackendType] = None) -> "Backend":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialize(backend_type)
        return cls._instance

    def _initialize(self, backend_type: Optional[BackendType] = None) -> None:
        if backend_type is None:
            backend_type = self._detect_best_backend()

        self._backend_type = backend_type

        if backend_type == BackendType.CPU:
            self._xp = np
            self._linalg = np.linalg
            self._fft = np.fft
            self._random = np.random
        elif backend_type in (BackendType.CUDA, BackendType.CUPY):
            self._xp = self._import_cupy()
            self._linalg = self._xp.linalg
            self._fft = self._xp.fft
            self._random = self._xp.random
        else:
            raise ValueError(f"Unknown backend type: {backend_type}")

    @staticmethod
    def _detect_best_backend() -> BackendType:
        env_backend = os.environ.get("QUANTUM_SIM_BACKEND", "").lower()
        if env_backend == "cpu":
            return BackendType.CPU
        if env_backend in ("cuda", "cupy", "gpu"):
            try:
                import cupy

                return BackendType.CUPY
            except ImportError:
                pass

        try:
            import cupy

            if cupy.cuda.runtime.getDeviceCount() > 0:
                return BackendType.CUPY
        except (ImportError, RuntimeError):
            pass

        return BackendType.CPU

    @staticmethod
    def _import_cupy() -> Any:
        try:
            import cupy

            return cupy
        except ImportError as e:
            raise ImportError(
                "CuPy is not installed. Install it with: pip install cupy-cuda11x (or appropriate version)"
            ) from e

    @property
    def type(self) -> BackendType:
        return self._backend_type

    @property
    def xp(self) -> Any:
        return self._xp

    @property
    def linalg(self) -> Any:
        return self._linalg

    @property
    def fft(self) -> Any:
        return self._fft

    @property
    def random(self) -> Any:
        return self._random

    def is_gpu(self) -> bool:
        return self._backend_type in (BackendType.CUDA, BackendType.CUPY)

    def to_numpy(self, array: Any) -> np.ndarray:
        if self.is_gpu():
            return array.get()
        return np.asarray(array)

    def to_device(self, array: Any) -> Any:
        if self.is_gpu():
            return self._xp.asarray(array)
        return np.asarray(array)

    def synchronize(self) -> None:
        if self.is_gpu():
            self._xp.cuda.Device().synchronize()

    def get_memory_info(self) -> tuple[int, int]:
        if self.is_gpu():
            free, total = self._xp.cuda.runtime.memGetInfo()
            return free, total
        import psutil

        mem = psutil.virtual_memory()
        return mem.available, mem.total

    def get_num_gpus(self) -> int:
        if self.is_gpu():
            return self._xp.cuda.runtime.getDeviceCount()
        return 0

    def set_device(self, device_id: int) -> None:
        if self.is_gpu():
            num_gpus = self.get_num_gpus()
            if device_id < 0 or device_id >= num_gpus:
                raise ValueError(f"Device ID {device_id} out of range (0-{num_gpus-1})")
            self._xp.cuda.Device(device_id).use()

    def get_device(self) -> int:
        if self.is_gpu():
            return self._xp.cuda.runtime.getDevice()
        return 0

    def __repr__(self) -> str:
        return f"Backend(type={self._backend_type.value}, gpu={self.is_gpu()})"


_global_backend: Optional[Backend] = None


def get_backend() -> Backend:
    global _global_backend
    if _global_backend is None:
        _global_backend = Backend()
    return _global_backend


def set_backend(backend_type: str | BackendType) -> Backend:
    global _global_backend
    if isinstance(backend_type, str):
        backend_type = BackendType(backend_type.lower())
    _global_backend = Backend(backend_type)
    return _global_backend
