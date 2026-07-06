"""Fixed-size observation history buffers."""

from __future__ import annotations

from collections import deque

import numpy as np


class HistoryBuffer:
    def __init__(self, length: int, dim: int):
        self.length = int(length)
        self.dim = int(dim)
        self._items: deque[np.ndarray] = deque(maxlen=self.length)

    def reset(self, value: np.ndarray | None = None) -> None:
        self._items.clear()
        if value is None:
            value = np.zeros(self.dim, dtype=np.float32)
        value = np.asarray(value, dtype=np.float32).reshape(self.dim)
        for _ in range(self.length):
            self._items.append(value.copy())

    def append(self, value: np.ndarray) -> None:
        self._items.append(np.asarray(value, dtype=np.float32).reshape(self.dim).copy())

    def as_array(self) -> np.ndarray:
        if len(self._items) != self.length:
            self.reset()
        return np.stack(list(self._items), axis=0)

    def flatten(self) -> np.ndarray:
        return self.as_array().reshape(-1).astype(np.float32)
