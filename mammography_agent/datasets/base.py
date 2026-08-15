from __future__ import annotations
from abc import ABC, abstractmethod
from pathlib import Path

class DatasetAdapter(ABC):
    def __init__(self, key: str, cfg: dict): self.key, self.cfg = key, cfg
    @abstractmethod
    def status(self) -> dict: ...
    @abstractmethod
    def download(self) -> dict: ...
    @abstractmethod
    def verify_integrity(self) -> dict: ...
    @abstractmethod
    def prepare(self) -> dict: ...
