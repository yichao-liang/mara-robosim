"""Gymnasium wrapper for mara-robosim PyBullet environments."""

import importlib
from typing import Any, Dict, Optional, Tuple
from typing import Type as TypingType
from typing import Union

import gymnasium
import numpy as np
from gymnasium import spaces
from numpy.typing import NDArray

from mara_robosim.config import PyBulletConfig
from mara_robosim.envs.base_env import PyBulletEnv
from mara_robosim.structs import Action, State


def _resolve_cls(
    env_cls: Union[str, TypingType[PyBulletEnv]],
) -> TypingType[PyBulletEnv]:
    """Resolve an env class from a string like
    'mara_robosim.envs.ants:PyBulletAntsEnv'."""
    if isinstance(env_cls, str):
        module_path, cls_name = env_cls.rsplit(":", 1)
        module = importlib.import_module(module_path)
        return getattr(module, cls_name)
    return env_cls


class MARARoboSimEnv(gymnasium.Env):
    """Wraps a mara-robosim PyBulletEnv as a standard gymnasium.Env.

    Observation: flattened State features as a Box space.
    Action: robot joint action space (forwarded from the underlying env).
    Reward: sparse +1 when all goal predicates are satisfied, 0 otherwise.
    """

    metadata = {"render_modes": ["rgb_array"]}

    def __init__(
        self,
        env_cls: Union[str, TypingType[PyBulletEnv]],
        config: Optional[PyBulletConfig] = None,
        render_mode: Optional[str] = None,
        **env_kwargs: Any,
    ) -> None:
        super().__init__()
        self.render_mode = render_mode
        use_gui = render_mode == "human"
        # Resolve string entry points to actual classes.
        resolved_cls = _resolve_cls(env_cls)

        # Create the underlying PyBullet environment.
        # If no config is provided, let the env class create its own
        # domain-specific default (e.g. AntsConfig, BlocksConfig).
        self._env = resolved_cls(config=config, use_gui=use_gui, **env_kwargs)
        self._config = self._env._config

        # Convert gym.spaces.Box → gymnasium.spaces.Box (the underlying env
        # uses the old gym API).
        old_as = self._env.action_space
        self.action_space = spaces.Box(
            low=old_as.low, high=old_as.high, dtype=old_as.dtype
        )

        # Build observation space from a sample reset.
        self._train_or_test = "train"
        self._task_idx = 0
        sample_obs = self._env.reset(self._train_or_test, self._task_idx)
        assert isinstance(sample_obs, State)
        self._obs_objects = sorted(sample_obs.data.keys(), key=lambda o: o.name)
        self._obs_features = self._build_feature_list(sample_obs)
        obs_dim = sum(len(feats) for feats in self._obs_features)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        self._max_episode_length = 500
        self._step_count = 0

    def _build_feature_list(self, state: State) -> list[list[str]]:
        """Get ordered feature names per object."""
        feature_lists = []
        for obj in self._obs_objects:
            feats = [
                f for f in obj.type.feature_names if f not in obj.type.sim_features
            ]
            feature_lists.append(feats)
        return feature_lists

    def _state_to_obs(self, state: State) -> NDArray:
        """Flatten a State into a 1-D numpy observation."""
        parts = []
        for obj, feats in zip(self._obs_objects, self._obs_features):
            for f in feats:
                parts.append(state.get(obj, f))
        return np.array(parts, dtype=np.float32)

    def _get_info(self, state: State) -> Dict[str, Any]:
        """Build info dict for gymnasium."""
        return {
            "state": state,
            "goal_reached": self._env.goal_reached(),
        }

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[NDArray, Dict[str, Any]]:
        if seed is not None:
            self._env._set_seed(seed)
        if options:
            self._train_or_test = options.get("train_or_test", self._train_or_test)
            self._task_idx = options.get("task_idx", self._task_idx)

        obs = self._env.reset(self._train_or_test, self._task_idx)
        assert isinstance(obs, State)
        self._step_count = 0
        return self._state_to_obs(obs), self._get_info(obs)

    def step(
        self, action: NDArray
    ) -> Tuple[NDArray, float, bool, bool, Dict[str, Any]]:
        action_obj = Action(np.array(action, dtype=np.float32))
        obs = self._env.step(action_obj)
        assert isinstance(obs, State)

        goal_reached = self._env.goal_reached()
        reward = 1.0 if goal_reached else 0.0
        terminated = goal_reached
        self._step_count += 1
        truncated = self._step_count >= self._max_episode_length

        return (
            self._state_to_obs(obs),
            reward,
            terminated,
            truncated,
            self._get_info(obs),
        )

    def render(self) -> Optional[NDArray]:
        if self.render_mode == "rgb_array":
            obs = self._env.get_observation(render=True)
            if hasattr(obs, "state_image"):
                img = obs.state_image
                if img is not None:
                    return np.array(img, dtype=np.uint8)
        return None

    def close(self) -> None:
        pass
