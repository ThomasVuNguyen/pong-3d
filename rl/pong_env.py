from __future__ import annotations

import math
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
from gymnasium import spaces


@dataclass(frozen=True)
class PongConfig:
    width: float = 800.0
    height: float = 450.0
    dt: float = 1.0 / 60.0

    paddle_w: float = 12.0
    paddle_h: float = 90.0
    paddle_speed: float = 420.0
    paddle_inset: float = 22.0

    ball_r: float = 6.0
    serve_speed: float = 360.0
    max_speed: float = 820.0
    speed_up_per_hit: float = 18.0

    ai_max_speed: float = 360.0
    ai_reaction: float = 0.22
    ai_dead_zone: float = 10.0
    
    # Curriculum learning: scale opponent difficulty (0.0 = easiest, 1.0 = hardest)
    ai_difficulty: float = 1.0  # Full difficulty (100%)

    terminate_on_point: bool = True
    
    # Domain randomization: enable/disable and set variation ranges
    enable_domain_randomization: bool = True
    # Randomization ranges (as multipliers, e.g., 0.9 means 90% to 110% of base value)
    rand_serve_speed: tuple[float, float] = (0.85, 1.15)  # ±15% serve speed
    rand_paddle_speed: tuple[float, float] = (0.9, 1.1)  # ±10% paddle speed
    rand_power_boost: tuple[float, float] = (0.8, 1.2)  # ±20% power shot boost
    rand_speed_up: tuple[float, float] = (0.7, 1.3)  # ±30% speed-up per hit
    rand_field_width: tuple[float, float] = (0.95, 1.05)  # ±5% field width
    rand_field_height: tuple[float, float] = (0.95, 1.05)  # ±5% field height
    rand_paddle_h: tuple[float, float] = (0.9, 1.1)  # ±10% paddle height


class PongEnv(gym.Env[np.ndarray, int]):
    """
    Headless Pong environment (agent controls left paddle, right paddle is scripted AI).

    Observation (state-based, normalized to [-1, 1]):
      [ball_x, ball_y, ball_vx, ball_vy, paddle_y, opp_y, paddle_charge]

    Action space:
      0: noop
      1: up
      2: down

    Reward (sparse):
      +1 when agent scores (ball exits right side)
      -1 when agent concedes (ball exits left side)
      0 otherwise
    """

    metadata = {"render_modes": ["rgb_array"], "render_fps": 60}

    def __init__(self, cfg: PongConfig | None = None, *, render_mode: str | None = None):
        super().__init__()
        self.cfg = cfg or PongConfig()
        self.render_mode = render_mode

        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(7,),  # Added paddle_charge
            dtype=np.float32,
        )

        self._x_l = self.cfg.paddle_inset
        self._x_r = self.cfg.width - self.cfg.paddle_inset - self.cfg.paddle_w

        self._paddle_y = 0.0
        self._paddle_vy = 0.0
        self._paddle_charge = 0.0  # Charge for power shots (0.0 to 1.0)
        self._opp_y = 0.0
        self._opp_target_y = 0.0
        self._opp_vy = 0.0
        self._opp_charge = 0.0  # Opponent also has charge for power shots

        self._ball_x = 0.0
        self._ball_y = 0.0
        self._ball_vx = 0.0
        self._ball_vy = 0.0
        self._ball_speed = 0.0

        self._ep_return = 0.0
        self._ep_len = 0
        self._last_hit_by_agent = False
        self._paddle_charge = 0.0
        
        # Domain randomization: episode-specific parameters (set on reset)
        self._ep_width = self.cfg.width
        self._ep_height = self.cfg.height
        self._ep_paddle_h = self.cfg.paddle_h
        self._ep_paddle_speed = self.cfg.paddle_speed
        self._ep_serve_speed = self.cfg.serve_speed
        self._ep_power_boost = 50.0  # Base power shot boost
        self._ep_speed_up = self.cfg.speed_up_per_hit

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        _ = options

        # Domain randomization: sample new parameters for this episode
        if self.cfg.enable_domain_randomization:
            self._ep_width = self.cfg.width * self.np_random.uniform(*self.cfg.rand_field_width)
            self._ep_height = self.cfg.height * self.np_random.uniform(*self.cfg.rand_field_height)
            self._ep_paddle_h = self.cfg.paddle_h * self.np_random.uniform(*self.cfg.rand_paddle_h)
            self._ep_paddle_speed = self.cfg.paddle_speed * self.np_random.uniform(*self.cfg.rand_paddle_speed)
            self._ep_serve_speed = self.cfg.serve_speed * self.np_random.uniform(*self.cfg.rand_serve_speed)
            self._ep_power_boost = 50.0 * self.np_random.uniform(*self.cfg.rand_power_boost)
            self._ep_speed_up = self.cfg.speed_up_per_hit * self.np_random.uniform(*self.cfg.rand_speed_up)
        else:
            self._ep_width = self.cfg.width
            self._ep_height = self.cfg.height
            self._ep_paddle_h = self.cfg.paddle_h
            self._ep_paddle_speed = self.cfg.paddle_speed
            self._ep_serve_speed = self.cfg.serve_speed
            self._ep_power_boost = 50.0
            self._ep_speed_up = self.cfg.speed_up_per_hit
        
        # Update paddle positions based on randomized field
        self._x_l = self.cfg.paddle_inset
        self._x_r = self._ep_width - self.cfg.paddle_inset - self.cfg.paddle_w

        self._ep_return = 0.0
        self._ep_len = 0
        self._last_hit_by_agent = False
        self._paddle_charge = 0.0
        self._opp_charge = 0.0

        self._paddle_y = (self._ep_height - self._ep_paddle_h) * 0.5
        self._paddle_vy = 0.0
        self._opp_y = (self._ep_height - self._ep_paddle_h) * 0.5
        self._opp_target_y = self._opp_y
        self._opp_vy = 0.0

        self._ball_x = self._ep_width * 0.5
        self._ball_y = self._ep_height * 0.5

        dir_sign = -1.0 if self.np_random.random() < 0.5 else 1.0
        angle = (self.np_random.random() * 0.55 - 0.275) * math.pi

        self._ball_speed = self._ep_serve_speed
        self._ball_vx = math.cos(angle) * self._ball_speed * dir_sign
        self._ball_vy = math.sin(angle) * self._ball_speed

        return self._obs(), {}

    def step(self, action: int):
        cfg = self.cfg
        dt = cfg.dt

        # Player paddle with charge system
        if action == 1:
            self._paddle_vy = -self._ep_paddle_speed
            # Build charge when moving (for power shots)
            self._paddle_charge = min(1.0, self._paddle_charge + dt * 2.0)
        elif action == 2:
            self._paddle_vy = self._ep_paddle_speed
            # Build charge when moving (for power shots)
            self._paddle_charge = min(1.0, self._paddle_charge + dt * 2.0)
        else:
            self._paddle_vy = 0.0
            # Decay charge when not moving
            self._paddle_charge = max(0.0, self._paddle_charge - dt * 1.5)

        self._paddle_y = float(
            np.clip(self._paddle_y + self._paddle_vy * dt, 0.0, self._ep_height - self._ep_paddle_h)
        )

        # Opponent AI (with curriculum difficulty scaling and power shot charge)
        target_center = self._ball_y - self._ep_paddle_h * 0.5
        # Scale reaction speed and max speed based on difficulty
        effective_reaction = cfg.ai_reaction * cfg.ai_difficulty
        effective_max_speed = cfg.ai_max_speed * cfg.ai_difficulty
        effective_dead_zone = cfg.ai_dead_zone * (2.0 - cfg.ai_difficulty)  # Larger dead zone when easier
        
        self._opp_target_y = self._opp_target_y + (target_center - self._opp_target_y) * effective_reaction

        delta = self._opp_target_y - self._opp_y
        if abs(delta) < effective_dead_zone:
            move = 0.0
            self._opp_vy = 0.0
            # Decay charge when not moving
            self._opp_charge = max(0.0, self._opp_charge - dt * 1.5)
        else:
            move = float(np.clip(delta, -effective_max_speed * dt, effective_max_speed * dt))
            self._opp_vy = move / dt  # Track velocity for charge building
            # Build charge when moving (same as agent)
            self._opp_charge = min(1.0, self._opp_charge + dt * 2.0)

        self._opp_y = float(np.clip(self._opp_y + move, 0.0, self._ep_height - self._ep_paddle_h))

        # Ball motion
        self._ball_x += self._ball_vx * dt
        self._ball_y += self._ball_vy * dt

        # Wall bounce
        if self._ball_y - cfg.ball_r <= 0.0:
            self._ball_y = cfg.ball_r
            self._ball_vy = abs(self._ball_vy)
        elif self._ball_y + cfg.ball_r >= self._ep_height:
            self._ball_y = self._ep_height - cfg.ball_r
            self._ball_vy = -abs(self._ball_vy)

        # Paddle collisions
        self._collide_paddle(side=-1)
        self._collide_paddle(side=+1)

        terminated = False
        reward = 0.0

        # Reward shaping: small reward for hitting the ball (encourages engagement)
        if self._last_hit_by_agent:
            reward = +0.1  # Small reward for successful hit
            self._last_hit_by_agent = False  # Reset flag

        # Scoring (agent is left paddle)
        if self._ball_x + cfg.ball_r < 0.0:
            terminated = True
            reward = -1.0  # Override hit reward with loss
        elif self._ball_x - cfg.ball_r > self._ep_width:
            terminated = True
            reward = +1.0  # Override hit reward with win

        truncated = False

        self._ep_return += reward
        self._ep_len += 1

        info: dict = {}
        if terminated or truncated:
            info["episode_return"] = float(self._ep_return)
            info["episode_length"] = int(self._ep_len)
            info["winner"] = "agent" if reward > 0 else "opponent"

        # Optionally continue match (not used by default)
        if not cfg.terminate_on_point and terminated:
            terminated = False
            reward = 0.0
            self.reset(seed=None)

        return self._obs(), float(reward), bool(terminated), bool(truncated), info

    def _collide_paddle(self, side: int) -> None:
        cfg = self.cfg
        px = self._x_l if side == -1 else self._x_r
        py = self._paddle_y if side == -1 else self._opp_y

        hit_x = (self._ball_x + cfg.ball_r > px) and (self._ball_x - cfg.ball_r < px + cfg.paddle_w)
        hit_y = (self._ball_y + cfg.ball_r > py) and (self._ball_y - cfg.ball_r < py + self._ep_paddle_h)
        if not (hit_x and hit_y):
            return

        # Only bounce if moving towards that paddle
        if side == -1 and self._ball_vx >= 0.0:
            return
        if side == +1 and self._ball_vx <= 0.0:
            return

        center = py + self._ep_paddle_h * 0.5
        offset = float(np.clip((self._ball_y - center) / (self._ep_paddle_h * 0.5), -1.0, 1.0))
        # More strategic angle control: edge hits create sharper angles
        max_bounce = 0.85 * math.pi  # Slightly wider angle range
        # Non-linear mapping: edges create sharper angles (more strategic)
        offset_power = offset * abs(offset)  # Quadratic for sharper edge angles
        angle = offset_power * (max_bounce * 0.5)

        # Power shot: charge adds extra speed boost (both agent and opponent)
        power_boost = 0.0
        if side == -1:  # Agent's paddle
            power_boost = self._paddle_charge * self._ep_power_boost  # Randomized power boost
            self._paddle_charge = 0.0  # Consume charge
            self._last_hit_by_agent = True
        else:  # Opponent's paddle
            power_boost = self._opp_charge * self._ep_power_boost  # Same randomized power shot capability
            self._opp_charge = 0.0  # Consume charge
            self._last_hit_by_agent = False

        self._ball_speed = float(np.clip(
            self._ball_speed + self._ep_speed_up + power_boost,
            self._ep_serve_speed,
            cfg.max_speed
        ))
        away = +1.0 if side == -1 else -1.0

        # Nudge out of paddle to avoid sticking
        if side == -1:
            self._ball_x = px + cfg.paddle_w + cfg.ball_r
        else:
            self._ball_x = px - cfg.ball_r

        self._ball_vx = math.cos(angle) * self._ball_speed * away
        self._ball_vy = math.sin(angle) * self._ball_speed

        # Tiny "english" from player paddle motion
        if side == -1:
            self._ball_vy += self._paddle_vy * 0.12

    def _obs(self) -> np.ndarray:
        cfg = self.cfg
        # Normalize to [-1, 1] using randomized field dimensions
        bx = (self._ball_x / self._ep_width) * 2.0 - 1.0
        by = (self._ball_y / self._ep_height) * 2.0 - 1.0
        bvx = float(np.clip(self._ball_vx / cfg.max_speed, -1.0, 1.0))
        bvy = float(np.clip(self._ball_vy / cfg.max_speed, -1.0, 1.0))

        py = (self._paddle_y / (self._ep_height - self._ep_paddle_h)) * 2.0 - 1.0
        oy = (self._opp_y / (self._ep_height - self._ep_paddle_h)) * 2.0 - 1.0
        
        # Paddle charge (0.0 to 1.0, normalized to [-1, 1])
        charge = self._paddle_charge * 2.0 - 1.0

        return np.array([bx, by, bvx, bvy, py, oy, charge], dtype=np.float32)

    def render(self):
        if self.render_mode is None:
            raise RuntimeError("render_mode is None. Create env with render_mode='rgb_array'.")
        if self.render_mode != "rgb_array":
            raise NotImplementedError(f"Unsupported render_mode={self.render_mode!r}")
        return self.render_rgb()

    def render_rgb(self, *, scale: int = 1) -> np.ndarray:
        """
        Returns an RGB frame (H, W, 3) uint8 for videos/viewers.
        Uses randomized field dimensions for accurate visualization.
        Dimensions are rounded to even numbers for H.264 compatibility.
        """
        cfg = self.cfg
        # Round to even numbers for H.264 encoding compatibility
        w = (int(self._ep_width) * scale) // 2 * 2
        h = (int(self._ep_height) * scale) // 2 * 2
        img = np.zeros((h, w, 3), dtype=np.uint8)

        # Background
        img[:, :] = (6, 7, 11)

        # Midline dashed
        dash_h = int(16 * scale)
        gap = int(10 * scale)
        mx = w // 2  # Use rounded width for midline
        for y in range(int(14 * scale), h - int(14 * scale), dash_h + gap):
            img[y : y + dash_h, mx - 2 * scale : mx + 2 * scale] = (130, 150, 190)

        # Paddles + ball
        def rect(x0: float, y0: float, rw: float, rh: float, color: tuple[int, int, int]):
            x1 = int(round((x0 + rw) * scale))
            y1 = int(round((y0 + rh) * scale))
            x0i = int(round(x0 * scale))
            y0i = int(round(y0 * scale))
            x0i = max(0, min(w, x0i))
            y0i = max(0, min(h, y0i))
            x1 = max(0, min(w, x1))
            y1 = max(0, min(h, y1))
            img[y0i:y1, x0i:x1] = color

        white = (232, 240, 255)
        rect(self._x_l, self._paddle_y, cfg.paddle_w, self._ep_paddle_h, white)
        rect(self._x_r, self._opp_y, cfg.paddle_w, self._ep_paddle_h, white)
        rect(self._ball_x - cfg.ball_r, self._ball_y - cfg.ball_r, cfg.ball_r * 2, cfg.ball_r * 2, white)

        return img


