"""三个能力分支（Skill 实现）。

- :class:`~cn.mcheat.agent.skills.syntax.SyntaxSkill`：语法知识查询
- :class:`~cn.mcheat.agent.skills.environment.EnvSkill`：环境状态检测
- :class:`~cn.mcheat.agent.skills.review.ReviewSkill`：代码修改指导
"""
from .environment import EnvSkill
from .review import ReviewSkill
from .syntax import SyntaxSkill

__all__ = ["EnvSkill", "ReviewSkill", "SyntaxSkill"]
