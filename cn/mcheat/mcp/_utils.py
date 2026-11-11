"""包内部通用工具（X2，零第三方依赖）。

与 ``cn.mcheat.rag._utils`` 同一约定：本模块位于依赖链最底层，
**不得 import 本包内任何其他模块**，以避免循环导入。
提供：统一日志入口、可选依赖的友好导入检查。
"""
from __future__ import annotations

import importlib.util
import logging

# ---------------------------------------------------------------------------
# 日志
# ---------------------------------------------------------------------------

_PACKAGE_LOGGER = "cn.mcheat.mcp"
_ROOT_LOGGER = logging.getLogger(_PACKAGE_LOGGER)
# 库代码永远不自行配置 handler（不调用 basicConfig），由应用侧决定输出方式；
# 加 NullHandler 只是为了避免 "No handler could be found" 告警。
_ROOT_LOGGER.addHandler(logging.NullHandler())


def get_logger(name: str) -> logging.Logger:
    """返回一个位于 ``cn.mcheat.mcp`` 命名空间下的 logger。

    Args:
        name: 一般传 ``__name__``；会自动剥掉包前缀，避免重复。

    Returns:
        ``logging.Logger`` 实例。应用侧可用
        ``logging.getLogger("cn.mcheat.mcp").setLevel(logging.DEBUG)`` 打开调试日志。
    """
    suffix = name.rsplit(_PACKAGE_LOGGER, 1)[-1].lstrip(".")
    return _ROOT_LOGGER.getChild(suffix) if suffix else _ROOT_LOGGER


# ---------------------------------------------------------------------------
# 可选依赖检查（导入失败给「怎么修」的提示，而不是裸 ImportError）
# ---------------------------------------------------------------------------


def require_module(module: str, feature: str, hint: str = "") -> None:
    """检查可选依赖是否已安装，缺依赖时抛出带修复提示的 ``ImportError``。

    Args:
        module: 模块名，如 ``"mcp"`` / ``"yaml"``。
        feature: 缺了它哪个功能不可用，用于错误信息。
        hint: 额外的安装提示（可空）。

    Raises:
        ImportError: 未安装时抛出，消息含安装命令。
    """
    if importlib.util.find_spec(module) is None:
        message = f"{feature} 需要可选依赖 '{module}'，但当前环境未安装。请执行：pip install {module}"
        if hint:
            message += f"（{hint}）"
        raise ImportError(message)
