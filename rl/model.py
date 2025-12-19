from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass(frozen=True)
class TransformerACConfig:
    obs_dim: int
    n_actions: int
    seq_len: int = 16

    d_model: int = 64
    n_layers: int = 4
    n_heads: int = 4
    d_ff: int = 256
    dropout: float = 0.0


class TransformerActorCritic(nn.Module):
    """
    Small transformer for PPO (CPU-friendly).
    Input:  (B, T, obs_dim)
    Output: logits (B, n_actions), value (B,)
    """

    def __init__(self, cfg: TransformerACConfig):
        super().__init__()
        self.cfg = cfg

        self.obs_embed = nn.Linear(cfg.obs_dim, cfg.d_model)
        self.pos_embed = nn.Parameter(torch.zeros(1, cfg.seq_len, cfg.d_model))

        layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.n_heads,
            dim_feedforward=cfg.d_ff,
            dropout=cfg.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        # Torch warns about nested tensor fast-path being disabled with norm_first=True.
        # Disabling nested tensors here keeps logs clean and has no functional impact for us.
        try:
            self.encoder = nn.TransformerEncoder(
                layer, num_layers=cfg.n_layers, enable_nested_tensor=False
            )
        except TypeError:
            # Older torch versions don't have enable_nested_tensor.
            self.encoder = nn.TransformerEncoder(layer, num_layers=cfg.n_layers)

        self.actor = nn.Linear(cfg.d_model, cfg.n_actions)
        self.critic = nn.Linear(cfg.d_model, 1)

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.normal_(self.pos_embed, mean=0.0, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=1.0)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        # Slightly smaller actor head init helps early stability
        nn.init.orthogonal_(self.actor.weight, gain=0.01)

    def forward(self, obs: torch.Tensor):
        """
        obs: float tensor (B, T, obs_dim)
        """
        if obs.ndim != 3:
            raise ValueError(f"Expected obs shape (B,T,D), got {tuple(obs.shape)}")
        if obs.shape[1] != self.cfg.seq_len:
            raise ValueError(f"Expected seq_len={self.cfg.seq_len}, got T={obs.shape[1]}")

        x = self.obs_embed(obs)
        x = x + self.pos_embed
        x = self.encoder(x)
        h = x[:, -1, :]  # use most recent token

        logits = self.actor(h)
        value = self.critic(h).squeeze(-1)
        return logits, value

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


