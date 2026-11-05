"""内置实现注册入口。

各实现模块用 ``@XXX_REGISTRY.register("name")`` 装饰器在 **import 时**完成
自注册。集中在此处一次性导入，好处是：

* 配置层（``config.py``）不需要 import 任何实现，保持依赖单向；
* 可选的第三方依赖（LangChain / sentence-transformers）只在适配器模块
  被**使用**时才真正 import，import 本包永远不会因缺依赖而失败；
* 用户新增组件时只需在应用启动时 ``import`` 自己的模块，注册表自然生效。
"""
from __future__ import annotations

import importlib

from .._utils import get_logger

logger = get_logger(__name__)

_BOOTSTRAPPED = False

# 相对当前包 ``cn.mcheat.rag.core`` 的模块列表
_REGISTRATION_MODULES: tuple[str, ...] = (
    "..loaders",                 # 5 种 Loader（由 build_default_loader_registry 组装）
    "..cleaning",                # Cleaner 注册
    "..indexing",                # 父子切分器 / ParentStore 注册
    "..embeddings",              # HashingEmbedder 注册
    "..embeddings.adapters",     # LangChainEmbedder 注册（延迟第三方 import）
    "..vectorstores",            # InMemory / FileBacked VectorStore 注册
    "..retrieval",               # BM25 / 融合 / 精排 / 上下文 / 兜底 注册
    "..evaluation",              # 评测模块（无注册，仅保证可导入）
)


def bootstrap(force: bool = False) -> None:
    """导入全部内置实现模块以完成注册（幂等）。

    Args:
        force: ``True`` 时强制重新导入。一般不需要；仅在测试中动态
            ``unregister`` 后想恢复内置实现时使用。
    """
    global _BOOTSTRAPPED
    if _BOOTSTRAPPED and not force:
        return
    package = __package__  # "cn.mcheat.rag.core"
    for relative_name in _REGISTRATION_MODULES:
        importlib.import_module(relative_name, package=package)
    _BOOTSTRAPPED = True
    logger.debug("内置组件注册完成")


def is_bootstrapped() -> bool:
    """是否已经完成注册（供测试与诊断使用）。"""
    return _BOOTSTRAPPED


__all__: list[str] = ["bootstrap", "is_bootstrapped"]
