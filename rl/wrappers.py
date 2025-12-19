from __future__ import annotations

from collections import deque

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class HistoryObsWrapper(gym.Wrapper):
    """
    Converts an env with obs shape (D,) into an env with obs shape (T, D),
    where T is the number of most recent observations (including current).
    """

    def __init__(self, env: gym.Env, *, seq_len: int):
        super().__init__(env)
        if seq_len < 1:
            raise ValueError("seq_len must be >= 1")
        self.seq_len = int(seq_len)

        base = env.observation_space
        if not isinstance(base, spaces.Box) or len(base.shape) != 1:
            raise TypeError("HistoryObsWrapper expects a 1D Box observation space")

        self._d = int(base.shape[0])
        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.seq_len, self._d),
            dtype=np.float32,
        )

        self._buf: deque[np.ndarray] = deque(maxlen=self.seq_len)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._buf.clear()
        for _ in range(self.seq_len):
            self._buf.append(np.asarray(obs, dtype=np.float32))
        return self._stack(), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._buf.append(np.asarray(obs, dtype=np.float32))
        return self._stack(), reward, terminated, truncated, info

    def _stack(self) -> np.ndarray:
        return np.stack(list(self._buf), axis=0).astype(np.float32, copy=False)


