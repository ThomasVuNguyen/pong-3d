from __future__ import annotations

import threading
import time
from dataclasses import dataclass

import numpy as np
import torch
from torch.distributions.categorical import Categorical

from .model import TransformerACConfig, TransformerActorCritic
from .pong_env import PongConfig, PongEnv
from .wrappers import HistoryObsWrapper


@dataclass
class LiveShared:
    """
    Thread-safe shared state between trainer thread and UI thread.
    """

    model_cfg: TransformerACConfig
    lock: threading.Lock
    stop_event: threading.Event

    # Updated by trainer:
    weights_version: int = 0
    state_dict: dict | None = None

    update: int = 0
    global_step: int = 0
    sps: int = 0
    ep_ret_mean: float = float("nan")
    ep_len_mean: float = float("nan")
    eval_return_mean: float = float("nan")
    eval_win_rate: float = float("nan")

    status: str = "Waiting for first update..."


class LiveTrainingUI:
    def __init__(
        self,
        shared: LiveShared,
        *,
        fps: int = 60,
        scale: int = 1,
        seed: int = 123,
        greedy: bool = False,
    ):
        self.shared = shared
        self.fps = max(1, int(fps))
        self.scale = max(1, int(scale))
        self.seed = int(seed)
        self.greedy = bool(greedy)

        self.device = torch.device("cpu")

        self.model = TransformerActorCritic(shared.model_cfg).to(self.device)
        self.model.eval()
        self._loaded_version = -1
        self._last_action = 0
        self._last_probs = None

        self.env = PongEnv(PongConfig(terminate_on_point=True), render_mode="rgb_array")
        self.env = HistoryObsWrapper(self.env, seq_len=shared.model_cfg.seq_len)
        self.rng = np.random.default_rng(self.seed)
        self.obs, _ = self.env.reset(seed=int(self.rng.integers(0, 2**31 - 1)))

        # Tk setup (import here so headless runs can skip)
        import tkinter as tk  # noqa: PLC0415

        self.tk = tk
        self.root = tk.Tk()
        self.root.title("Pong RL Training (Live)")
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        self.header = tk.Label(self.root, text="", font=("Menlo", 12), justify="left", anchor="w")
        self.header.pack(fill="x", padx=10, pady=(10, 0))

        self.canvas = tk.Canvas(
            self.root,
            width=int(800 * self.scale),
            height=int(450 * self.scale),
            highlightthickness=0,
        )
        self.canvas.pack(padx=10, pady=10)

        self.footer = tk.Label(self.root, text="", font=("Menlo", 11), justify="left", anchor="w")
        self.footer.pack(fill="x", padx=10, pady=(0, 10))

        self._photo = None
        self._img_id = None

        self._last_frame_t = time.time()

    def _on_close(self) -> None:
        self.shared.stop_event.set()
        try:
            self.root.destroy()
        except Exception:
            pass

    def _maybe_load_weights(self) -> None:
        with self.shared.lock:
            ver = self.shared.weights_version
            sd = self.shared.state_dict
        if sd is None or ver == self._loaded_version:
            return
        self.model.load_state_dict(sd, strict=True)
        self.model.eval()
        self._loaded_version = ver

    def _policy_action(self, obs: np.ndarray) -> int:
        with torch.no_grad():
            o = torch.from_numpy(obs[None, ...]).to(self.device)  # (1,T,D)
            logits, _ = self.model(o)
            dist = Categorical(logits=logits)
            if self.greedy:
                act = int(torch.argmax(logits, dim=-1).item())
            else:
                act = int(dist.sample().item())
            probs = dist.probs.squeeze(0).cpu().numpy()
            self._last_action = act
            self._last_probs = probs
            return act

    def _draw_frame(self) -> None:
        frame = self.env.unwrapped.render_rgb(scale=self.scale)

        # Tk no-deps path: PPM bytes -> PhotoImage
        h, w, _ = frame.shape
        header = f"P6 {w} {h} 255 ".encode("ascii")
        ppm = header + frame.tobytes()

        self._photo = self.tk.PhotoImage(data=ppm, format="PPM")
        if self._img_id is None:
            self._img_id = self.canvas.create_image(0, 0, anchor="nw", image=self._photo)
        else:
            self.canvas.itemconfig(self._img_id, image=self._photo)

    def _update_text(self) -> None:
        with self.shared.lock:
            txt = (
                f"update={self.shared.update}  step={self.shared.global_step}  sps={self.shared.sps}\n"
                f"train: ep_ret_mean={self.shared.ep_ret_mean:.3f}  ep_len_mean={self.shared.ep_len_mean:.1f}\n"
                f"eval:  return_mean={self.shared.eval_return_mean:.3f}  win_rate={self.shared.eval_win_rate:.3f}"
            )
            status = self.shared.status
        if self._last_probs is not None:
            p0, p1, p2 = self._last_probs.tolist()
            mode = "greedy" if self.greedy else "sample"
            status = f"{status} | ui={mode} | action={self._last_action} | probs=[{p0:.2f},{p1:.2f},{p2:.2f}]"
        self.header.config(text=txt)
        self.footer.config(text=status)

    def tick(self) -> None:
        if self.shared.stop_event.is_set():
            return

        self._maybe_load_weights()

        # Advance one env step per tick
        try:
            act = self._policy_action(self.obs) if self._loaded_version >= 0 else 0
            self.obs, _, term, trunc, _ = self.env.step(act)
            if term or trunc:
                self.obs, _ = self.env.reset(seed=int(self.rng.integers(0, 2**31 - 1)))
        except Exception as e:
            with self.shared.lock:
                self.shared.status = f"UI stepping error: {type(e).__name__}: {e}"

        self._draw_frame()
        self._update_text()

        delay_ms = max(1, int(1000 / self.fps))
        self.root.after(delay_ms, self.tick)

    def run(self) -> None:
        self.root.after(1, self.tick)
        self.root.mainloop()


