"""可插拔向量库实现（接口见 ``core.interfaces.VectorStore``）。"""
from .memory import FileBackedVectorStore, InMemoryVectorStore
from .sqlite import SqliteVectorStore

__all__ = ["InMemoryVectorStore", "FileBackedVectorStore", "SqliteVectorStore"]
