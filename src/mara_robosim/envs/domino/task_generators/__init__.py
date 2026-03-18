"""Task generators for the domino environment.

Task generators create EnvironmentTask instances with initial states and
goals. Different generators can be composed with different component
combinations.
"""

from mara_robosim.envs.domino.task_generators.base_generator import TaskGenerator
from mara_robosim.envs.domino.task_generators.domino_task_generator import (
    DominoTaskGenerator,
)

__all__ = [
    "TaskGenerator",
    "DominoTaskGenerator",
]
