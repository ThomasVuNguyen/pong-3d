from __future__ import annotations

import argparse
import os
import threading
import time
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical

from .live_ui import LiveShared, LiveTrainingUI
from .web_ui import WebTrainingUI
from .logger import CsvLogger
from .model import TransformerACConfig, TransformerActorCritic
from .pong_env import PongConfig, PongEnv
from .video import can_write_mp4, write_mp4
from .wrappers import HistoryObsWrapper


@dataclass(frozen=True)
class PPOConfig:
    total_steps: int = 2_000_000
    n_envs: int = 32
    n_steps: int = 256
    n_epochs: int = 4
    n_minibatches: int = 8

    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: float = 0.03

    lr: float = 3e-4


def make_env(*, seq_len: int, seed: int, idx: int, ai_difficulty: float = 1.0) -> gym.Env:
    def _thunk():
        cfg = PongConfig(terminate_on_point=True, ai_difficulty=ai_difficulty)
        env = PongEnv(cfg)
        env = HistoryObsWrapper(env, seq_len=seq_len)
        env.reset(seed=seed + idx)
        return env

    return _thunk


def _extract_episode_stats(infos) -> tuple[list[float], list[int]]:
    rets: list[float] = []
    lens: list[int] = []
    if not infos:
        return rets, lens

    # Gymnasium vector envs may return:
    # - dict of arrays/lists (one entry per env), possibly with "final_info"
    # - list of dicts (one per env)
    def _consume_info(d: dict) -> None:
        if not d:
            return
        if "episode_return" in d:
            rets.append(float(d["episode_return"]))
        if "episode_length" in d:
            lens.append(int(d["episode_length"]))

    if isinstance(infos, (list, tuple)):
        for d in infos:
            if isinstance(d, dict):
                _consume_info(d)
        return rets, lens

    if isinstance(infos, dict):
        # If autoreset is enabled, terminal infos often land under "final_info"
        final_infos = infos.get("final_info", None)
        if final_infos is not None:
            for d in final_infos:
                if isinstance(d, dict):
                    _consume_info(d)

        # Or keys may be directly vectorized
        if "episode_return" in infos:
            for x in infos["episode_return"]:
                if x is not None:
                    rets.append(float(x))
        if "episode_length" in infos:
            for x in infos["episode_length"]:
                if x is not None:
                    lens.append(int(x))

    return rets, lens


def _reset_done_envs(envs: gym.vector.SyncVectorEnv, obs: np.ndarray, dones: np.ndarray, rng: np.random.Generator):
    """
    Gymnasium's SyncVectorEnv (0.29.x) does not provide reset_done().
    Manually reset only the sub-envs that are done and patch their observations in-place.
    """
    if not np.any(dones):
        return obs
    # SyncVectorEnv keeps a list of underlying envs at .envs
    for i, d in enumerate(dones):
        if not d:
            continue
        o, _ = envs.envs[i].reset(seed=int(rng.integers(0, 2**31 - 1)))
        obs[i] = o
    return obs


def evaluate(
    *,
    model: TransformerActorCritic,
    model_cfg: TransformerACConfig,
    seed: int,
    episodes: int,
    max_steps: int,
    video_path: str | None,
    video_fps: int,
    greedy: bool = False,
    ai_difficulty: float = 1.0,  # Use full difficulty for evaluation
) -> dict:
    device = next(model.parameters()).device
    env = PongEnv(PongConfig(terminate_on_point=True, ai_difficulty=ai_difficulty), render_mode="rgb_array")
    env = HistoryObsWrapper(env, seq_len=model_cfg.seq_len)

    rng = np.random.default_rng(seed)
    returns: list[float] = []
    wins = 0
    frames: list[np.ndarray] = []

    model.eval()
    for ep in range(episodes):
        obs, _ = env.reset(seed=int(rng.integers(0, 2**31 - 1)))
        done = False
        ep_ret = 0.0
        steps = 0
        while not done and steps < max_steps:
            with torch.no_grad():
                o = torch.from_numpy(obs[None, ...]).to(device)
                logits, _ = model(o)
                if greedy:
                    act = int(torch.argmax(logits, dim=-1).item())
                else:
                    # Sample from policy distribution (matches training behavior)
                    dist = Categorical(logits=logits)
                    act = int(dist.sample().item())
            obs, r, term, trunc, info = env.step(act)
            done = term or trunc
            ep_ret += float(r)
            steps += 1

            if video_path is not None and ep == 0:
                frames.append(env.unwrapped.render_rgb(scale=1))

        returns.append(ep_ret)
        if ep_ret > 0:
            wins += 1

    if video_path is not None and frames:
        try:
            write_mp4(video_path, frames, fps=video_fps)
            print(f"[rl.train] Wrote video: {video_path}")
        except Exception as e:
            # Don't crash training if video writing fails, but warn the user.
            print(f"[rl.train] Failed to write video {video_path}: {type(e).__name__}: {e}")
            video_path = None
    elif video_path is not None:
        print(f"[rl.train] No frames collected for video {video_path}")
        video_path = None

    env.close()
    model.train()
    return {
        "eval_return_mean": float(np.mean(returns)) if returns else float("nan"),
        "eval_return_std": float(np.std(returns)) if returns else float("nan"),
        "eval_win_rate": float(wins / max(1, episodes)),
        "eval_episodes": int(episodes),
    }


def train(args: argparse.Namespace) -> None:
    torch.set_num_threads(max(1, args.torch_threads))
    device = torch.device("cpu")

    ppo = PPOConfig(
        total_steps=args.total_steps,
        n_envs=args.n_envs,
        n_steps=args.n_steps,
        n_epochs=args.n_epochs,
        n_minibatches=args.n_minibatches,
        lr=args.lr,
    )

    # Curriculum learning: start at full difficulty (can be adjusted based on win rate if needed)
    current_ai_difficulty = 1.0  # Start at 100% difficulty
    difficulty_update_interval = 10  # Update difficulty every N updates
    
    # Create envs with initial difficulty (will be updated via curriculum)
    def _make_envs_with_difficulty(difficulty: float):
        return [
            make_env(seq_len=args.seq_len, seed=args.seed, idx=i, ai_difficulty=difficulty)
            for i in range(ppo.n_envs)
        ]
    
    envs = gym.vector.SyncVectorEnv(
        _make_envs_with_difficulty(current_ai_difficulty),
        # Keep behavior predictable: we will handle resets ourselves.
        **(
            {"autoreset_mode": gym.vector.AutoresetMode.DISABLED}
            if hasattr(gym.vector, "AutoresetMode")
            else {}
        ),
    )
    obs, _ = envs.reset(seed=args.seed)
    reset_rng = np.random.default_rng(args.seed + 999)

    obs_dim = int(envs.single_observation_space.shape[-1])
    n_actions = int(envs.single_action_space.n)

    # If a live UI is running, reuse its exact model config to avoid mismatch.
    if getattr(args, "_live_shared", None) is not None:
        model_cfg = args._live_shared.model_cfg
        if model_cfg.obs_dim != obs_dim or model_cfg.n_actions != n_actions or model_cfg.seq_len != args.seq_len:
            raise RuntimeError("Live UI model config does not match environment/model args.")
    else:
        model_cfg = TransformerACConfig(
            obs_dim=obs_dim,
            n_actions=n_actions,
            seq_len=args.seq_len,
            d_model=args.d_model,
            n_layers=args.n_layers,
            n_heads=args.n_heads,
            d_ff=args.d_ff,
            dropout=0.0,
        )
    model = TransformerActorCritic(model_cfg).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=ppo.lr, eps=1e-5)

    # Progress counters (may be overwritten by resume)
    global_step = 0

    # Disable videos if MP4 writing isn't available (don't crash training).
    if args.video_every > 0 and not can_write_mp4():
        print("[rl.train] MP4 writer not available; disabling videos (install imageio-ffmpeg to enable).")
        args.video_every = 0

    # Resume from checkpoint (model + optimizer + progress)
    start_update = 1
    if args.resume and args.save_path and os.path.exists(args.save_path):
        ckpt = torch.load(args.save_path, map_location="cpu")
        model.load_state_dict(ckpt["model_state_dict"], strict=True)
        if "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if "global_step" in ckpt:
            global_step = int(ckpt["global_step"])
        if "update" in ckpt:
            start_update = int(ckpt["update"]) + 1
        print(f"Resuming from {args.save_path} at update={start_update} step={global_step}")

        # Push resumed weights to UI immediately
        if getattr(args, "_live_shared", None) is not None:
            sd = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            with args._live_shared.lock:
                args._live_shared.state_dict = sd
                args._live_shared.weights_version += 1
                args._live_shared.status = "Resumed from checkpoint."

    n_params = model.n_params()
    print(f"Model params: {n_params:,} (~{n_params/1e3:.1f}k)")

    batch_size = ppo.n_envs * ppo.n_steps
    minibatch_size = batch_size // ppo.n_minibatches

    print(
        f"Rollout: n_envs={ppo.n_envs}, n_steps={ppo.n_steps} -> batch={batch_size}\n"
        f"Training until 100% win rate or cancellation. Epochs/update={ppo.n_epochs}, minibatch={minibatch_size}"
    )
    ep_returns_window: list[float] = []
    ep_lens_window: list[int] = []

    csv = None
    if args.log_csv:
        csv = CsvLogger(
            path=args.log_csv,
            fieldnames=[
                "time_sec",
                "update",
                "global_step",
                "sps",
                "ep_ret_mean",
                "ep_len_mean",
                "kl",
                "clipfrac",
                "pg_loss",
                "v_loss",
                "entropy",
                "eval_return_mean",
                "eval_return_std",
                "eval_win_rate",
                "eval_episodes",
                "video_path",
            ],
        )

    # Graceful cancellation for headless runs: Ctrl+C sets a stop flag (so we can save a checkpoint).
    stop_requested = False
    if threading.current_thread() is threading.main_thread():
        try:
            import signal

            def _handle_stop(_signum, _frame):
                nonlocal stop_requested
                stop_requested = True
                if getattr(args, "_live_shared", None) is not None:
                    with args._live_shared.lock:
                        args._live_shared.status = "Stopping..."

            signal.signal(signal.SIGINT, _handle_stop)
            signal.signal(signal.SIGTERM, _handle_stop)
        except Exception:
            pass

    start_time = time.time()
    last_update_done = start_update - 1
    update = start_update
    target_win_rate = 1.0  # 100% win rate target
    
    while True:
        if stop_requested:
            break
        if args._stop_event is not None and args._stop_event.is_set():
            if args._live_shared is not None:
                with args._live_shared.lock:
                    args._live_shared.status = "Stopped (window closed)."
            break
        # Storage
        obs_buf = np.zeros((ppo.n_steps, ppo.n_envs, args.seq_len, obs_dim), dtype=np.float32)
        act_buf = np.zeros((ppo.n_steps, ppo.n_envs), dtype=np.int64)
        logp_buf = np.zeros((ppo.n_steps, ppo.n_envs), dtype=np.float32)
        val_buf = np.zeros((ppo.n_steps, ppo.n_envs), dtype=np.float32)
        rew_buf = np.zeros((ppo.n_steps, ppo.n_envs), dtype=np.float32)
        done_buf = np.zeros((ppo.n_steps, ppo.n_envs), dtype=np.float32)

        for t in range(ppo.n_steps):
            global_step += ppo.n_envs
            obs_buf[t] = obs

            with torch.no_grad():
                obs_t = torch.from_numpy(obs).to(device)
                logits, values = model(obs_t)
                dist = Categorical(logits=logits)
                actions = dist.sample()
                logp = dist.log_prob(actions)

            act = actions.cpu().numpy()
            next_obs, rewards, terminated, truncated, infos = envs.step(act)
            dones = np.logical_or(terminated, truncated)

            act_buf[t] = act
            logp_buf[t] = logp.cpu().numpy()
            val_buf[t] = values.cpu().numpy()
            rew_buf[t] = rewards
            done_buf[t] = dones.astype(np.float32)

            # Track episodic stats (env adds episode_return/episode_length on termination)
            r, l = _extract_episode_stats(infos)
            ep_returns_window.extend(r)
            ep_lens_window.extend(l)
            ep_returns_window = ep_returns_window[-200:]
            ep_lens_window = ep_lens_window[-200:]

            obs = next_obs
            obs = _reset_done_envs(envs, obs, dones, reset_rng)

        with torch.no_grad():
            obs_t = torch.from_numpy(obs).to(device)
            _, last_values = model(obs_t)
            last_values = last_values.cpu().numpy()  # (n_envs,)

        # GAE-Lambda advantages
        adv_buf = np.zeros_like(rew_buf, dtype=np.float32)
        last_gae = np.zeros((ppo.n_envs,), dtype=np.float32)
        for t in reversed(range(ppo.n_steps)):
            next_nonterminal = 1.0 - done_buf[t]
            next_values = last_values if t == ppo.n_steps - 1 else val_buf[t + 1]
            delta = rew_buf[t] + ppo.gamma * next_values * next_nonterminal - val_buf[t]
            last_gae = delta + ppo.gamma * ppo.gae_lambda * next_nonterminal * last_gae
            adv_buf[t] = last_gae
        ret_buf = adv_buf + val_buf

        # Flatten
        b_obs = obs_buf.reshape((batch_size, args.seq_len, obs_dim))
        b_act = act_buf.reshape((batch_size,))
        b_logp = logp_buf.reshape((batch_size,))
        b_adv = adv_buf.reshape((batch_size,))
        b_ret = ret_buf.reshape((batch_size,))
        b_val = val_buf.reshape((batch_size,))

        # Normalize advantages
        b_adv = (b_adv - b_adv.mean()) / (b_adv.std() + 1e-8)

        # PPO updates
        idxs = np.arange(batch_size)
        clip_fracs: list[float] = []
        approx_kls: list[float] = []
        pg_losses: list[float] = []
        v_losses: list[float] = []
        entropies: list[float] = []

        for epoch in range(ppo.n_epochs):
            np.random.shuffle(idxs)
            for start in range(0, batch_size, minibatch_size):
                mb_idx = idxs[start : start + minibatch_size]

                obs_mb = torch.from_numpy(b_obs[mb_idx]).to(device)
                act_mb = torch.from_numpy(b_act[mb_idx]).to(device)
                old_logp_mb = torch.from_numpy(b_logp[mb_idx]).to(device)
                adv_mb = torch.from_numpy(b_adv[mb_idx]).to(device)
                ret_mb = torch.from_numpy(b_ret[mb_idx]).to(device)
                old_val_mb = torch.from_numpy(b_val[mb_idx]).to(device)

                logits, value = model(obs_mb)
                dist = Categorical(logits=logits)
                new_logp = dist.log_prob(act_mb)
                entropy = dist.entropy().mean()

                log_ratio = new_logp - old_logp_mb
                ratio = log_ratio.exp()

                # Policy loss
                pg_loss1 = -adv_mb * ratio
                pg_loss2 = -adv_mb * torch.clamp(ratio, 1.0 - ppo.clip_coef, 1.0 + ppo.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss (clipped)
                v_clipped = old_val_mb + (value - old_val_mb).clamp(-ppo.clip_coef, ppo.clip_coef)
                v_loss_unclipped = (value - ret_mb).pow(2)
                v_loss_clipped = (v_clipped - ret_mb).pow(2)
                v_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()

                loss = pg_loss - ppo.ent_coef * entropy + ppo.vf_coef * v_loss

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), ppo.max_grad_norm)
                optimizer.step()

                approx_kl = (old_logp_mb - new_logp).mean().item()
                approx_kls.append(approx_kl)

                clipped = (ratio - 1.0).abs() > ppo.clip_coef
                clip_fracs.append(clipped.float().mean().item())
                pg_losses.append(float(pg_loss.item()))
                v_losses.append(float(v_loss.item()))
                entropies.append(float(entropy.item()))

                if ppo.target_kl is not None and approx_kl > ppo.target_kl:
                    break
            if ppo.target_kl is not None and (approx_kls and approx_kls[-1] > ppo.target_kl):
                break

        # Logging
        sps = int(global_step / max(1e-9, time.time() - start_time))
        mean_ret = float(np.mean(ep_returns_window)) if ep_returns_window else float("nan")
        mean_len = float(np.mean(ep_lens_window)) if ep_lens_window else float("nan")
        mean_kl = float(np.mean(approx_kls)) if approx_kls else 0.0
        mean_clip = float(np.mean(clip_fracs)) if clip_fracs else 0.0
        mean_pg = float(np.mean(pg_losses)) if pg_losses else 0.0
        mean_v = float(np.mean(v_losses)) if v_losses else 0.0
        mean_ent = float(np.mean(entropies)) if entropies else 0.0

        eval_stats = {}
        video_path = ""
        if args.eval_every > 0 and (update % args.eval_every == 0):
            do_video = args.video_every > 0 and (update % args.video_every == 0)
            if do_video:
                video_path = os.path.join(args.videos_dir, f"eval_update_{update:05d}.mp4")
                print(f"[rl.train] Writing video at update {update}: {video_path}")
            eval_stats = evaluate(
                model=model,
                model_cfg=model_cfg,
                seed=args.seed + 10_000 + update,
                episodes=args.eval_episodes,
                max_steps=args.eval_max_steps,
                video_path=video_path if do_video else None,
                video_fps=args.video_fps,
                greedy=getattr(args, "eval_greedy", False),
                ai_difficulty=1.0,  # Always use full difficulty for evaluation
            )
            
            # Check win rate and break if target reached
            if "eval_win_rate" in eval_stats:
                win_rate = eval_stats["eval_win_rate"]
                if win_rate >= target_win_rate:
                    print(f"\n[rl.train] Target win rate ({target_win_rate:.0%}) reached! Win rate: {win_rate:.1%}")
                    if args._live_shared is not None:
                        with args._live_shared.lock:
                            args._live_shared.status = f"Training complete: {win_rate:.1%} win rate"
                    break
                
                # Curriculum learning: increase difficulty as agent improves
                # Scale difficulty from 0.3 (easy) to 1.0 (hard) based on win rate
                if update % difficulty_update_interval == 0:
                    # Gradually increase difficulty: 0.3 at 0% win rate, 1.0 at 70%+ win rate
                    new_difficulty = min(1.0, 0.3 + (win_rate / 0.7) * 0.7)
                    if abs(new_difficulty - current_ai_difficulty) > 0.05:  # Only update if significant change
                        current_ai_difficulty = new_difficulty
                        print(f"[rl.train] Curriculum: adjusting opponent difficulty to {current_ai_difficulty:.1%} (win_rate={win_rate:.1%})")
                        # Recreate environments with new difficulty
                        envs.close()
                        envs = gym.vector.SyncVectorEnv(
                            _make_envs_with_difficulty(current_ai_difficulty),
                            **(
                                {"autoreset_mode": gym.vector.AutoresetMode.DISABLED}
                                if hasattr(gym.vector, "AutoresetMode")
                                else {}
                            ),
                        )
                        obs, _ = envs.reset(seed=args.seed + update)

        # Display win rate in console output if available
        win_rate_str = ""
        if "eval_win_rate" in eval_stats:
            win_rate = eval_stats["eval_win_rate"]
            win_rate_str = f" | win_rate {win_rate:.1%}"
        
        print(
            f"update {update:6d} | step {global_step:9d} | sps {sps:5d} | "
            f"ep_ret {mean_ret:6.3f} | ep_len {mean_len:6.1f} | kl {mean_kl:7.5f} | clip {mean_clip:5.3f}{win_rate_str}"
        )

        if csv is not None:
            csv.log(
                {
                    "time_sec": float(time.time() - start_time),
                    "update": int(update),
                    "global_step": int(global_step),
                    "sps": int(sps),
                    "ep_ret_mean": float(mean_ret),
                    "ep_len_mean": float(mean_len),
                    "kl": float(mean_kl),
                    "clipfrac": float(mean_clip),
                    "pg_loss": float(mean_pg),
                    "v_loss": float(mean_v),
                    "entropy": float(mean_ent),
                    **eval_stats,
                    "video_path": video_path,
                }
            )

        # Stream weights + metrics to live UI
        if args._live_shared is not None and args.ui_model_every > 0:
            if update % args.ui_model_every == 0:
                # Copy small state_dict safely for UI thread
                sd = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                with args._live_shared.lock:
                    args._live_shared.state_dict = sd
                    args._live_shared.weights_version += 1
                    args._live_shared.update = int(update)
                    args._live_shared.global_step = int(global_step)
                    args._live_shared.sps = int(sps)
                    args._live_shared.ep_ret_mean = float(mean_ret)
                    args._live_shared.ep_len_mean = float(mean_len)
                    if "eval_return_mean" in eval_stats:
                        args._live_shared.eval_return_mean = float(eval_stats["eval_return_mean"])
                        args._live_shared.eval_win_rate = float(eval_stats.get("eval_win_rate", float("nan")))
                    args._live_shared.status = "Training..."

        if args.save_path and (update % args.save_every == 0):
            os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "model_cfg": model_cfg.__dict__,
                    "seed": args.seed,
                    "global_step": int(global_step),
                    "update": int(update),
                },
                args.save_path,
            )
        last_update_done = update
        update += 1

    envs.close()
    if csv is not None:
        csv.close()

    # Always save a final checkpoint so you can resume after cancellation/window close.
    if args.save_path:
        os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "model_cfg": model_cfg.__dict__,
                "seed": args.seed,
                "global_step": int(global_step),
                "update": int(last_update_done),
            },
            args.save_path,
        )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()

    # Training
    p.add_argument("--total-steps", type=int, default=2_000_000)
    p.add_argument("--n-envs", type=int, default=32)
    p.add_argument("--n-steps", type=int, default=256)
    p.add_argument("--n-epochs", type=int, default=4)
    p.add_argument("--n-minibatches", type=int, default=8)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--torch-threads", type=int, default=0, help="0 = let PyTorch decide")

    # Model (defaults tuned to ~200k params)
    p.add_argument("--seq-len", type=int, default=16)
    p.add_argument("--d-model", type=int, default=64)
    p.add_argument("--n-layers", type=int, default=4)
    p.add_argument("--n-heads", type=int, default=4)
    p.add_argument("--d-ff", type=int, default=256)

    # Saving
    p.add_argument("--save-path", type=str, default="runs/pong_transformer_200k.pt")
    p.add_argument("--save-every", type=int, default=10)
    p.add_argument("--resume", action="store_true", help="Resume from --save-path if it exists")
    p.add_argument("--no-resume", dest="resume", action="store_false", help="Do not resume even if checkpoint exists")
    p.set_defaults(resume=True)

    # Progressive logging / eval / video
    p.add_argument("--log-csv", type=str, default="runs/metrics.csv")
    p.add_argument("--eval-every", type=int, default=5)
    p.add_argument("--eval-episodes", type=int, default=40)
    p.add_argument("--eval-max-steps", type=int, default=400)
    p.add_argument("--videos-dir", type=str, default="runs/videos")
    p.add_argument("--video-every", type=int, default=20)
    p.add_argument("--video-fps", type=int, default=30)
    p.add_argument("--eval-greedy", action="store_true", help="Use greedy (argmax) actions in evaluation/videos (default: sample like training)")

    # Live UI (enabled by default; disable for headless runs)
    p.add_argument("--no-ui", action="store_true", help="Disable realtime UI window")
    p.add_argument("--ui-fps", type=int, default=60)
    p.add_argument("--ui-scale", type=int, default=1)
    p.add_argument("--ui-model-every", type=int, default=1, help="Push weights to UI every N updates")
    p.add_argument("--ui-greedy", action="store_true", help="Use argmax actions in the UI (default: sample)")

    # Web UI args
    p.add_argument("--web-ui", action="store_true", help="Use Web UI instead of Tkinter")
    p.add_argument("--port", type=int, default=1306, help="Port for Web UI")

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.torch_threads <= 0:
        args.torch_threads = max(1, os.cpu_count() or 1)

    # Attach optional live UI plumbing onto args to keep changes minimal.
    args._live_shared = None
    args._stop_event = None

    if args.no_ui:
        train(args)
    else:
        # Auto-detect if we should default to web UI (headless env)
        use_web = args.web_ui
        if not use_web and os.environ.get("DISPLAY") is None:
            print("[rl.train] No DISPLAY detected. Defaulting to Web UI.")
            use_web = True

        # Try to start UI; if Tk isn't available, fall back to headless training.
        try:
            # Derive obs/action dims from a single env
            # Derive obs/action dims from a single env
            tmp = PongEnv(PongConfig(terminate_on_point=True))
            tmp = HistoryObsWrapper(tmp, seq_len=args.seq_len)
            obs_dim = int(tmp.observation_space.shape[-1])
            n_actions = int(tmp.action_space.n)
            tmp.close()

            model_cfg = TransformerACConfig(
                obs_dim=obs_dim,
                n_actions=n_actions,
                seq_len=args.seq_len,
                d_model=args.d_model,
                n_layers=args.n_layers,
                n_heads=args.n_heads,
                d_ff=args.d_ff,
                dropout=0.0,
            )

            shared = LiveShared(
                model_cfg=model_cfg,
                lock=threading.Lock(),
                stop_event=threading.Event(),
            )
            args._live_shared = shared
            args._stop_event = shared.stop_event

            def _trainer():
                try:
                    train(args)
                    with shared.lock:
                        shared.status = "Training finished."
                except Exception as e:
                    with shared.lock:
                        shared.status = f"Training error: {type(e).__name__}: {e}"
                    raise

            t = threading.Thread(target=_trainer, daemon=True)
            t.start()

            if use_web:
                # Web UI mode
                ui = WebTrainingUI(shared, port=args.port, fps=args.ui_fps, scale=args.ui_scale, greedy=args.ui_greedy)
                ui.run()
            else:
                # Desktop Tk mode
                ui = LiveTrainingUI(shared, fps=args.ui_fps, scale=args.ui_scale, greedy=args.ui_greedy)
                ui.run()
        except Exception as e:
            print(f"[rl.train] Live UI unavailable ({type(e).__name__}: {e}). Running headless.")
            train(args)


