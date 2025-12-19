from __future__ import annotations

import argparse

from .model import TransformerACConfig, TransformerActorCritic


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--obs-dim", type=int, default=6)
    p.add_argument("--n-actions", type=int, default=3)
    p.add_argument("--seq-len", type=int, default=16)
    p.add_argument("--d-model", type=int, default=64)
    p.add_argument("--n-layers", type=int, default=4)
    p.add_argument("--n-heads", type=int, default=4)
    p.add_argument("--d-ff", type=int, default=256)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = TransformerACConfig(
        obs_dim=args.obs_dim,
        n_actions=args.n_actions,
        seq_len=args.seq_len,
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        d_ff=args.d_ff,
        dropout=0.0,
    )
    model = TransformerActorCritic(cfg)
    n = model.n_params()
    print(f"{n:,} params (~{n/1e3:.1f}k)")


if __name__ == "__main__":
    main()


