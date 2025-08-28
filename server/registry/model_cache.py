# server/registry/model_cache.py
from __future__ import annotations
import threading
import time
from typing import Any, Dict, Tuple
class ModelCache:
    def __init__(self, capacity: int = 2) -> None:
        self.capacity = max(1, capacity)
        self._lock = threading.Lock()
        self._store: Dict[str, Tuple[float, Any]] = {}
    def get(self, key: str):
        with self._lock:
            item = self._store.get(key)
            if not item:
                return None
            self._store[key] = (time.time(), item[1])
            return item[1]
    def put(self, key: str, value: Any):
        with self._lock:
            self._store[key] = (time.time(), value)
            self._evict_if_needed()
    def _evict_if_needed(self):
        if len(self._store) <= self.capacity:
            return
        oldest_key = min(self._store.items(), key=lambda kv: kv[1][0])[0]
        _, obj = self._store.pop(oldest_key)
        try:
            if hasattr(obj, "to"):
                obj.to("cpu")
        except Exception:
            pass
        try:
            del obj
        except Exception:
            pass
    def clear(self):
        with self._lock:
            keys = list(self._store.keys())
            for k in keys:
                _, obj = self._store.pop(k)
                try:
                    if hasattr(obj, "to"):
                        obj.to("cpu")
                except Exception:
                    pass
                try:
                    del obj
                except Exception:
                    pass
