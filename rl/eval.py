from __future__ import annotations

import argparse

import gymnasium as gym
import numpy as np
import torch
from torch.distributions.categorical import Categorical

from .model import TransformerACConfig, TransformerActorCritic
from .pong_env import PongConfig, PongEnv
from .wrappers import HistoryObsWrapper


def make_env(seq_len: int, seed: int) -> gym.Env:
    env = PongEnv(PongConfig(terminate_on_point=True))
    return HistoryObsWrapper(env, seq_len=seq_len)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, default="runs/pong_transformer_200k.pt")
    p.add_argument("--episodes", type=int, default=100)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--greedy", action="store_true", help="Use greedy (argmax) actions (default: sample like training)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cpu")

    ckpt = torch.load(args.ckpt, map_location="cpu")
    cfg = TransformerACConfig(**ckpt["model_cfg"])
    model = TransformerActorCritic(cfg).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    env = make_env(seq_len=cfg.seq_len, seed=args.seed)
    rng = np.random.default_rng(args.seed)

    returns = []
    for ep in range(args.episodes):
        obs, _ = env.reset(seed=int(rng.integers(0, 2**31 - 1)))
        done = False
        ep_ret = 0.0
        while not done:
            with torch.no_grad():
                o = torch.from_numpy(obs[None, ...]).to(device)  # (1,T,D)
                logits, _ = model(o)
                if args.greedy:
                    act = int(torch.argmax(logits, dim=-1).item())
                else:
                    # Sample from policy distribution (matches training behavior)
                    dist = Categorical(logits=logits)
                    act = int(dist.sample().item())
            obs, r, term, trunc, _ = env.step(act)
            done = term or trunc
            ep_ret += float(r)
        returns.append(ep_ret)
        if (ep + 1) % 20 == 0:
            print(f"episode {ep+1:4d}/{args.episodes} | avg_return(last 20)={np.mean(returns[-20:]):.3f}")

    print(f"avg_return={np.mean(returns):.3f} over {args.episodes} episodes")


if __name__ == "__main__":
    main()


