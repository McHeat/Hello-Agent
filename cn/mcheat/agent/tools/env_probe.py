"""本机 Python 环境探针 + LangChain 工具封装。

设计要点：

* 只用标准库（``subprocess`` / ``sys`` / ``platform`` / ``shutil``）；
* 每个探测项（版本 / 解释器 / pip / pip list）独立捕获异常并记入
  ``EnvReport.issues``，**单项失败绝不中断整轮对话**；
* 所有 subprocess 都带超时（默认 10s），防止 pip 挂起阻塞 LangGraph 图；
* ``pip list`` 输出按配置截断（默认 30 行），防止 token 膨胀。
"""
from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys
from typing import Callable

from langchain_core.tools import tool

from .._utils import get_logger
from ..models import EnvReport
from ..core.interfaces import EnvProbe

logger = get_logger(__name__)


def _run_cmd(cmd: list[str], timeout: float) -> tuple[str, str | None]:
    """执行命令并返回 ``(stdout, error)``；error 非 None 表示失败。

    失败原因转成友好文本（命令不存在 / 超时 / OS 错误），不抛异常。
    """
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout, check=False
        )
    except FileNotFoundError:
        return "", f"命令不存在：{' '.join(cmd)}"
    except subprocess.TimeoutExpired:
        return "", f"命令超时（>{timeout}s）：{' '.join(cmd)}"
    except OSError as exc:
        return "", f"执行失败：{' '.join(cmd)}（{exc}）"
    output = (proc.stdout or proc.stderr or "").strip()
    return output, None


class LocalEnvProbe(EnvProbe):
    """默认环境探针：探测当前进程所在的 Python 安装环境。

    Args:
        timeout: subprocess 超时秒数。
        include_pip_list: 是否执行 ``pip list``（相对较慢，可关）。
        pip_list_limit: ``pip list`` 截断行数（0 表示不截断、只报数量）。
    """

    def __init__(
        self,
        timeout: float = 10.0,
        include_pip_list: bool = True,
        pip_list_limit: int = 30,
    ) -> None:
        self.timeout = float(timeout)
        self.include_pip_list = bool(include_pip_list)
        self.pip_list_limit = int(pip_list_limit)

    def probe(self) -> EnvReport:
        report = EnvReport(
            python_version=platform.python_version(),
            executable=sys.executable or "未知",
            platform_info=platform.platform(),
        )

        # -- 虚拟环境判定 ---------------------------------------------------
        # sys.prefix != sys.base_prefix 是 venv/virtualenv 的标准判别法
        if sys.prefix != sys.base_prefix:
            report.virtual_env = True
            report.venv_detail = f"{sys.prefix}（base: {sys.base_prefix}）"
        elif "CONDA_DEFAULT_ENV" in os.environ:
            report.virtual_env = True
            report.venv_detail = f"conda 环境 {os.environ['CONDA_DEFAULT_ENV']}"

        # -- PATH 上的 python（可能与当前解释器不一致，常见踩坑点） -----------
        which_python = shutil.which("python") or shutil.which("python3")
        if which_python and report.executable and which_python != report.executable:
            report.issues.append(
                f"注意：PATH 上的 python 是 {which_python}，与当前解释器 "
                f"{report.executable} 不一致，可能是虚拟环境未激活。"
            )

        # -- pip 版本 --------------------------------------------------------
        version_out, err = _run_cmd(
            [sys.executable, "-m", "pip", "--version"], self.timeout
        )
        if err is None and version_out:
            report.pip_available = True
            report.pip_version = version_out
        else:
            report.pip_available = False
            report.issues.append(err or "pip 不可用（--version 无输出）")

        # -- 已安装包清单（可选，截断） --------------------------------------
        if self.include_pip_list and report.pip_available:
            list_out, err = _run_cmd([sys.executable, "-m", "pip", "list"], self.timeout)
            if err is not None:
                report.issues.append(f"pip list 探测失败：{err}")
            elif list_out:
                rows = [line for line in list_out.splitlines() if line.strip()]
                header, body = rows[:2], rows[2:]
                shown = body[: self.pip_list_limit] if self.pip_list_limit > 0 else body
                report.packages = header + shown
                if len(body) > len(shown):
                    report.packages.append(
                        f"...（共 {len(body)} 个包，已省略 {len(body) - len(shown)} 行）"
                    )

        logger.debug(
            "环境探测完成：python=%s venv=%s pip=%s issues=%d",
            report.python_version, report.virtual_env,
            report.pip_available, len(report.issues),
        )
        return report


# ---------------------------------------------------------------------------
# LangChain 工具封装（供自建 ReAct Agent 复用）
# ---------------------------------------------------------------------------


def build_env_tool(probe: EnvProbe) -> Callable[[], str]:
    """把任意探针实现包成 LangChain ``@tool``。"""

    @tool
    def check_python_env() -> str:
        """检测当前机器的 Python 安装环境：版本、解释器路径、pip、虚拟环境、已装包。"""
        return probe.probe().to_display()

    return check_python_env


@tool
def check_python_env() -> str:
    """检测当前机器的 Python 安装环境：版本、解释器路径、pip、虚拟环境、已装包。"""
    return LocalEnvProbe().probe().to_display()
