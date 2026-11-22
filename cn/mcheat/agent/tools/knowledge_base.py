"""本地 Python 语法知识库 + LangChain 工具封装。

设计要点：

* 内置 ``SYNTAX_KNOWLEDGE`` 是**带代码示例的主题式知识库**（dict，O(1) 查找 +
  别名最长匹配），刻意不做向量检索 —— 覆盖主题有限、可离线复现、测试可断言；
* ``search`` 未命中时返回 ``matched=False`` 并附推荐主题，技能层据此**如实说明**，
  而不是让 LLM 编造（与 rag 框架「兜底链路」同一哲学）；
* ``build_syntax_tool`` 把任意 :class:`SyntaxKnowledgeBase` 实例包成 LangChain
  ``@tool``，供自建 ReAct Agent 复用；``search_python_syntax`` 是基于默认
  知识库的模块级现成工具。
"""
from __future__ import annotations

import unicodedata
from typing import Any, Callable

from langchain_core.tools import tool

from ..core.interfaces import SyntaxKnowledgeBase
from ..models import KnowledgeHit

# ---------------------------------------------------------------------------
# 内置知识库数据（主题 key -> 条目）
# ---------------------------------------------------------------------------

SYNTAX_KNOWLEDGE: dict[str, dict[str, Any]] = {
    "list_comprehension": {
        "title": "列表推导式",
        "aliases": ["列表推导", "list comprehension", "推导式"],
        "content": """\
列表推导式（list comprehension）是构建列表的简洁语法：

    squares = [x * x for x in range(10)]

带条件过滤：

    evens = [x for x in range(20) if x % 2 == 0]

带 if/else 表达式（注意位置在 for 之前）：

    labels = ["偶" if x % 2 == 0 else "奇" for x in range(5)]

嵌套循环（外层在前）：

    pairs = [(x, y) for x in "ab" for y in (1, 2)]

常见坑：
1. 过滤的 if 在 for 之后，选择的 if/else 在 for 之前，二者不可混用；
2. 推导式在 Python 3 有独立作用域，内部变量不泄漏到外部；
3. 只用它表达简单的映射/过滤，逻辑复杂时请写普通 for 循环，可读性优先。""",
    },
    "dict_comprehension": {
        "title": "字典与集合推导式",
        "aliases": ["字典推导", "dict comprehension", "集合推导", "set comprehension"],
        "content": """\
字典推导式（dict comprehension）：

    lengths = {name: len(name) for name in ["python", "langgraph"]}

反转映射：

    reversed_map = {v: k for k, v in original.items()}

集合推导式（set comprehension）：

    unique_exts = {p.rsplit(".", 1)[-1] for p in filenames}

常见坑：
1. 字典推导式的键冲突时后者覆盖前者，需要保留首个请用 dict.setdefault 或循环；
2. 集合推导式的元素必须可哈希（list 不行，tuple 可以）。""",
    },
    "generators": {
        "title": "生成器与 yield",
        "aliases": ["生成器", "generator", "yield", "惰性求值"],
        "content": """\
生成器（generator）用 yield 逐个产出值，惰性求值、内存 O(1)：

    def countdown(n):
        while n > 0:
            yield n
            n -= 1

生成器表达式（注意是圆括号，方括号是列表推导式）：

    total = sum(x * x for x in range(1_000_000))  # 不会先建出百万元素列表

yield from 委托子生成器：

    def chain(*iterables):
        for it in iterables:
            yield from it

常见坑：
1. 生成器只能消费一次，遍历完就耗尽，需要重复使用请转 list 或重新创建；
2. 生成器函数调用时**不执行任何代码**，返回生成器对象，next() 才开始执行；
3. 在生成器里 return value 等价于 StopIteration(value)。""",
    },
    "decorators": {
        "title": "装饰器",
        "aliases": ["装饰器", "decorator", "@", " functools.wraps"],
        "content": """\
装饰器是「接收函数、返回新函数」的高阶函数，@ 语法是语法糖：

    import functools

    def log_calls(func):
        @functools.wraps(func)          # 保留原函数的 __name__ / docstring
        def wrapper(*args, **kwargs):
            print(f"调用 {func.__name__}")
            return func(*args, **kwargs)
        return wrapper

    @log_calls                          # 等价于 add = log_calls(add)
    def add(a, b):
        return a + b

带参数的装饰器需要再包一层（三层嵌套）：

    def repeat(times):
        def decorator(func):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                for _ in range(times):
                    result = func(*args, **kwargs)
                return result
            return wrapper
        return decorator

    @repeat(times=3)
    def greet(): print("hi")

类也可以做装饰器（实现 __call__）；functools.cache / functools.lru_cache
是标准库自带的缓存装饰器。常见坑：忘记 @functools.wraps 会导致被装饰函数的
元信息（名字、文档）丢失，影响调试与序列化。""",
    },
    "context_manager": {
        "title": "上下文管理器 with",
        "aliases": ["上下文管理器", "with 语句", "with", "context manager", "__enter__", "__exit__"],
        "content": """\
with 语句保证「进入/退出」成对执行，即使中途抛异常也会执行清理：

    with open("data.txt", encoding="utf-8") as f:
        text = f.read()
    # 离开 with 块时 f 自动 close，无论是否发生异常

自定义上下文管理器（类方式）：

    class Timer:
        def __enter__(self):
            self.start = time.perf_counter()
            return self                       # as 后面拿到的就是返回值
        def __exit__(self, exc_type, exc, tb):
            self.elapsed = time.perf_counter() - self.start
            return False                      # True 表示吞掉异常，一般返回 False

更轻量的方式是 contextlib.contextmanager 装饰器 + yield：

    from contextlib import contextmanager

    @contextmanager
    def timer():
        start = time.perf_counter()
        try:
            yield                             # yield 前是 __enter__，后是 __exit__
        finally:
            print(time.perf_counter() - start)

多个上下文可叠写：with open(a) as fa, open(b) as fb: ...""",
    },
    "args_kwargs": {
        "title": "*args 与 **kwargs",
        "aliases": ["*args", "**kwargs", "可变参数", "关键字参数"],
        "content": """\
函数参数的四种形态与顺序（缺省参数必须在位置参数之后）：

    def func(pos, /, normal, *args, kw_only=0, **kwargs):
        #     ^^^ 仅位置   ^^^^^^^ 可位置可关键字
        ...

*args 收集多余位置参数为 tuple；**kwargs 收集多余关键字参数为 dict：

    def flexible(*args, **kwargs):
        print(args, kwargs)

    flexible(1, 2, name="py")   # (1, 2) {'name': 'py'}

调用时的解包（与定义正好相反）：

    args, kwargs = (1, 2), {"name": "py"}
    flexible(*args, **kwargs)

Python 3.8+ 的 / 表示其左侧参数只能按位置传，* 或 *name 表示其右侧只能按
关键字传。常见坑：**kwargs 会吞掉拼写错误的关键字，关键字参数尽量显式声明。""",
    },
    "closures": {
        "title": "闭包与 nonlocal",
        "aliases": ["闭包", "closure", "nonlocal", "自由变量"],
        "content": """\
闭包：内层函数引用外层函数作用域的变量，且外层函数已返回后仍存活：

    def make_counter():
        count = 0
        def counter():
            nonlocal count        # 声明修改外层变量；不写则报 UnboundLocalError
            count += 1
            return count
        return counter

    c = make_counter()
    c(); c()   # 1, 2 —— count 活在闭包单元里，不被回收

查看闭包变量：c.__closure__ / c.__code__.co_freevars。

常见坑：
1. 循环里创建闭包共享同一个变量，延迟求值时拿到的是最终值；
   解决：用默认参数固定 snapshot，或用 functools.partial；
2. nonlocal 只作用于函数作用域，模块级变量要用 global（尽量都别用）。""",
    },
    "slicing": {
        "title": "切片",
        "aliases": ["切片", "slice", "序列截取"],
        "content": """\
切片语法 seq[start:stop:step]，左闭右开，任何一项都可省略：

    s = "hello, world"
    s[0:5]      # 'hello'
    s[-6:-1]    # 'worl'（负索引从尾部数，-1 是最后一个字符）
    s[::2]      # 'hlowrd'（步长 2）
    s[::-1]     # 'dlrow ,olleh'（经典反转写法）

切片是**浅拷贝**，产生新对象（与视图型语言不同）；但元素是引用：

    copy1 = lst[:]              # 浅拷贝，等价于 list(lst) / lst.copy()
    copy1[0] is lst[0]          # True —— 第一层元素仍是同一对象

对可变序列可原地赋值切片（长度可不等）：

    lst = [1, 2, 3, 4]
    lst[1:3] = [20, 30, 35]     # [1, 20, 30, 35, 4]

常见坑：s[a:b] 越界不会抛 IndexError，只会得到较短的切片。""",
    },
    "fstring": {
        "title": "f-string 与字符串格式化",
        "aliases": ["f-string", "f字符串", "格式化", "format"],
        "content": """\
f-string（3.6+）是首选的格式化方式，效率与可读性俱佳：

    name, score = "python", 95.678
    f"{name}: {score:.1f}"          # 'python: 95.7'
    f"{name!r}"                     # "'python'"（调用 repr）
    f"{value=}"                     # 'value=42'（3.8+ 调试写法）
    f"{'py':>10}"                   # 右对齐宽度 10

日期与数字字面量分隔符：

    f"{1_000_000:,}"                # '1,000,000'

常见坑：
1. f-string 内不能出现反斜杠（3.12 之前），需要换行符先赋给变量；
2. format() 方法与 % 格式化在旧代码里常见，新代码统一用 f-string；
3. 国际化文案不要硬编码 f-string，换 gettext 之类的占位符方案。""",
    },
    "type_hints": {
        "title": "类型注解",
        "aliases": ["类型注解", "类型标注", "type hint", "typing", "mypy"],
        "content": """\
类型注解不影响运行时行为，但支撑 mypy / pyright 静态检查与 IDE 补全：

    def process(items: list[str], limit: int = 10) -> dict[str, int]:
        return {item: len(item) for item in items[:limit]}

常用类型（3.9+ 直接用内置容器，不再需要 typing.List）：

    from typing import Optional, Union, Callable, Protocol
    x: int | None = None            # 3.10+ 可用 | 代替 Optional/Union
    handler: Callable[[str], bool]

更严格的检查建议打开：from __future__ import annotations（延迟求值注解，
避免前向引用问题）。运行时取注解：typing.get_type_hints(func)。

常见坑：注解是「提示」不是「约束」，运行期不会自动校验，需要 pydantic /
beartype 等库才 enforcement。""",
    },
    "exceptions": {
        "title": "异常处理",
        "aliases": ["异常", "try", "except", "finally", "raise", "异常处理"],
        "content": """\
标准结构（finally 无论如何都执行，常用于清理）：

    try:
        risky()
    except (ValueError, KeyError) as exc:   # 捕获多个类型
        logger.warning("已知问题：%s", exc)
    except Exception:                        # 兜底要窄，别裸 except
        raise                                # 原样重新抛出，保留 traceback
    else:
        pass                                 # 无异常才执行
    finally:
        cleanup()

抛出与链式：

    raise ValueError(f"非法输入: {value}") from original_exc   # 显式因果链

自定义异常：继承 Exception（不要继承 BaseException）：

    class AgentError(Exception):
        def __init__(self, message, hint=None): ...

最佳实践：
1. 捕获你能处理的，其余放行；不要用异常吞错（except: pass）；
2. 用 try 覆盖尽量窄的代码块；
3. EAFP（先斩后奏）是 Python 惯用法，比反复 hasattr 检查更地道。""",
    },
    "oop_property": {
        "title": "类、继承与 property",
        "aliases": ["面向对象", "类", "class", "继承", "property", "封装", "魔术方法"],
        "content": """\
类基础：所有方法第一个参数是 self；类体直接写属性默认值。

    class Stack:
        def __init__(self):
            self._items: list[int] = []

        def push(self, item: int) -> None:
            self._items.append(item)

        @property
        def size(self) -> int:               # property 让方法像属性一样访问
            return len(self._items)

    s = Stack()
    s.size                                   # 不带括号

继承与 super()：

    class BoundedStack(Stack):
        def __init__(self, capacity: int):
            super().__init__()               # 必须显式调用父类初始化
            self.capacity = capacity

常用魔术方法：__repr__（调试输出）、__eq__、__len__、__iter__、
__enter__/__exit__、__call__（让实例可像函数一样调用）。

dataclass（3.7+）自动生成 __init__ / __repr__ / __eq__：

    from dataclasses import dataclass
    @dataclass
    class Point:
        x: float
        y: float = 0.0

常见坑：类属性是全体实例共享的（可变默认值要用 __init__ 里的实例属性）。""",
    },
    "asyncio": {
        "title": "异步编程 asyncio",
        "aliases": ["异步", "async", "await", "asyncio", "协程"],
        "content": """\
协程用 async def 定义，await 挂起等待结果，事件循环负责调度：

    import asyncio

    async def fetch(name: str, delay: float) -> str:
        await asyncio.sleep(delay)           # 模拟 IO；await 只能出现在 async 里
        return f"{name} done"

    async def main():
        results = await asyncio.gather(      # 并发执行，总耗时 ≈ 最长者
            fetch("a", 1.0), fetch("b", 2.0)
        )
        print(results)

    asyncio.run(main())                      # 程序唯一入口

要点：
1. async def 调用返回协程对象，不 await / 不 gather 就不会执行；
2. CPU 密集型任务用 asyncio.to_thread / ProcessPoolExecutor，不要阻塞事件循环；
3. asyncio.gather 并发等待多个；as_completed 先完成先处理；
4. httpx / aiohttp 是异步 HTTP 客户端，requests 是同步的。""",
    },
    "gil": {
        "title": "GIL 与并发模型",
        "aliases": ["GIL", "全局解释器锁", "多线程", "multiprocessing", "并发"],
        "content": """\
GIL（Global Interpreter Lock）：CPython 解释器同一时刻只允许一个线程执行
字节码。影响与选型：

* CPU 密集型：多线程无加速（甚至更慢），用 multiprocessing（多进程）
  或 C 扩展 / numpy（计算时释放 GIL）；
* IO 密集型：线程在 IO 等待时释放 GIL，多线程 / asyncio 都能提升吞吐；

    from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

    with ThreadPoolExecutor(max_workers=8) as pool:   # IO 密集
        results = list(pool.map(fetch_url, urls))

    with ProcessPoolExecutor() as pool:               # CPU 密集
        results = list(pool.map(heavy_compute, data))

注意：
1. 进程池的参数与返回值必须可 pickle；
2. free-threading（无 GIL）构建在 3.13 以实验特性提供，生产仍以标准 CPython 为准；
3. 「多线程 + 共享可变状态」需要 threading.Lock，或改用队列传递消息。""",
    },
    "venv": {
        "title": "虚拟环境",
        "aliases": ["虚拟环境", "venv", "virtualenv", "环境隔离"],
        "content": """\
虚拟环境为每个项目隔离独立的包目录，避免全局污染与版本冲突：

    python -m venv .venv             # 创建
    source .venv/bin/activate        # 激活（macOS / Linux）
    .venv\\Scripts\\activate          # 激活（Windows PowerShell）
    deactivate                       # 退出

激活后 pip install 的包只落在 .venv 里；pip freeze > requirements.txt
导出依赖清单，pip install -r requirements.txt 复现。

判别当前是否在虚拟环境：

    import sys
    in_venv = sys.prefix != sys.base_prefix

常见坑：
1. conda 环境与 venv 是两套体系，混用容易装错位置；
2. IDE / 脚本执行器务必指向虚拟环境里的解释器（.venv/bin/python），
   否则 import 的是全局包。""",
    },
    "pip_usage": {
        "title": "pip 与依赖管理",
        "aliases": ["pip", "安装包", "pip install", "依赖管理", "requirements"],
        "content": """\
pip 常用命令（建议统一 python -m pip 形式，确保装进当前解释器）：

    python -m pip install requests            # 安装最新版
    python -m pip install "requests>=2.31,<3" # 版本约束
    python -m pip install -r requirements.txt # 按清单安装
    python -m pip list                        # 已安装列表
    python -m pip show requests               # 包详情与依赖
    python -m pip uninstall requests          # 卸载

导出与复现：

    python -m pip freeze > requirements.txt

常见坑：
1. pip install 装到了系统 Python —— 99% 是没激活虚拟环境；
2. 版本冲突时优先用 pip install pkg==x.y.z 精确回退，再用 pip check 验证；
3. 大型项目建议 pyproject.toml（现代标准）管理依赖，pip install -e . 可安装本地包。""",
    },
}


# ---------------------------------------------------------------------------
# 知识库实现
# ---------------------------------------------------------------------------


class InMemorySyntaxKnowledgeBase(SyntaxKnowledgeBase):
    """内置本地知识库：别名最长匹配，O(条目数) 扫描，零外部依赖。

    Args:
        entries: 自定义条目（结构同 ``SYNTAX_KNOWLEDGE``）；
            不传则使用内置的 16 个主题。
    """

    def __init__(self, entries: dict[str, dict[str, Any]] | None = None) -> None:
        # 深拷贝一份，避免调用方通过引用改掉全局数据
        self._entries: dict[str, dict[str, Any]] = {
            key: {
                "title": value["title"],
                "aliases": list(value.get("aliases", [])),
                "content": value["content"],
            }
            for key, value in (entries or SYNTAX_KNOWLEDGE).items()
        }

    @property
    def topics(self) -> list[str]:
        """全部主题的展示名（字典序）。"""
        return sorted(entry["title"] for entry in self._entries.values())

    def search(self, query: str) -> KnowledgeHit:
        """在用户问题里找主题词：命中的**最长别名**优先，避免「推导式」
        抢走「字典推导式」这类前缀歧义。
        """
        normalized = unicodedata.normalize("NFKC", str(query or "")).lower()
        best_key: str | None = None
        best_alias_len = 0
        for key, entry in self._entries.items():
            for alias in (entry["title"], *entry.get("aliases", [])):
                a = alias.lower()
                if a in normalized and len(a) > best_alias_len:
                    best_key, best_alias_len = key, len(a)

        if best_key is None:
            return KnowledgeHit(matched=False, suggestions=self.topics)

        entry = self._entries[best_key]
        return KnowledgeHit(
            matched=True,
            topic=best_key,
            title=entry["title"],
            content=entry["content"],
        )


# ---------------------------------------------------------------------------
# LangChain 工具封装（供自建 ReAct Agent 复用）
# ---------------------------------------------------------------------------


def _format_hit(hit: KnowledgeHit) -> str:
    if hit.matched:
        return f"【{hit.title}】\n{hit.content}"
    topics = "、".join(hit.suggestions)
    return f"知识库未命中。可问的主题：{topics}"


def build_syntax_tool(kb: SyntaxKnowledgeBase) -> Callable[[str], str]:
    """把任意知识库实现包成 LangChain ``@tool``（便于 bind_tools 复用）。"""

    @tool
    def search_python_syntax(topic: str) -> str:
        """在 Python 语法知识库中查询指定主题（如：列表推导式、装饰器、yield）的讲解。"""
        return _format_hit(kb.search(topic))

    return search_python_syntax


# 模块级现成工具：基于默认内置知识库（skills 层不经过它，走接口注入）
_DEFAULT_KB = InMemorySyntaxKnowledgeBase()


@tool
def search_python_syntax(topic: str) -> str:
    """在 Python 语法知识库中查询指定主题（如：列表推导式、装饰器、yield）的讲解。"""
    return _format_hit(_DEFAULT_KB.search(topic))
