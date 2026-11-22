"""包内部通用工具（零第三方依赖）。

本模块位于依赖链最底层，**不得 import 本包内任何其他模块**，以避免循环导入。
提供：统一日志入口。其余通用逻辑按需补充，保持与 ``cn.mcheat.rag._utils``
相同的分层约束。
"""
from __future__ import annotations

import logging

# ---------------------------------------------------------------------------
# 日志
# ---------------------------------------------------------------------------

_PACKAGE_LOGGER = "cn.mcheat.agent"
_ROOT_LOGGER = logging.getLogger(_PACKAGE_LOGGER)
# 库代码永远不自行配置 handler（不调用 basicConfig），由应用侧决定输出方式；
# 加 NullHandler 只是为了避免 "No handler could be found" 告警。
_ROOT_LOGGER.addHandler(logging.NullHandler())


def get_logger(name: str) -> logging.Logger:
    """返回一个位于 ``cn.mcheat.agent`` 命名空间下的 logger。

    Args:
        name: 一般传 ``__name__``；会自动剥掉包前缀，避免重复。

    Returns:
        ``logging.Logger`` 实例。应用侧可用
        ``logging.getLogger("cn.mcheat.agent").setLevel(logging.DEBUG)`` 打开调试日志，
        或设置环境变量 ``AGENT_LOG_LEVEL=DEBUG``（见 config.py）。
    """
    suffix = name.rsplit(_PACKAGE_LOGGER, 1)[-1].lstrip(".")
    return _ROOT_LOGGER.getChild(suffix) if suffix else _ROOT_LOGGER
