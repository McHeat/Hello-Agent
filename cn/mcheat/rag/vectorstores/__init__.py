"""可插拔向量库实现（接口见 ``core.interfaces.VectorStore``）。"""
from .memory import FileBackedVectorStore, InMemoryVectorStore

__all__ = ["InMemoryVectorStore", "FileBackedVectorStore"]
