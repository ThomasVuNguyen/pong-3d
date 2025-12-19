# pong
A game of pong

## play (browser)

- **Fastest**: open `index.html` in your browser.
- **Recommended** (local server):

```bash
cd /Users/tungnguyen/Documents/GitHub/pong
python3 -m http.server 5173
```

Then open `http://localhost:5173`.

## model

We also wanted to train a transformers model through Reinforcement Learning to play pong well.

Design decisions

- Observation space: game state

- Action space: Discrete action

- Reward design: Sparse

- Environment engineering:
  - Fixed-timestep `step(action)` loop (deterministic physics; avoid variable `dt` during training)
  - Standard RL API: `obs, reward, done, info = step(action)` and `obs = reset(seed=...)`
  - Episode boundaries: end on point, end on game (e.g. first-to-11), or use fixed-horizon rollouts
  - Randomization knobs (seeded): serve direction/angle, initial ball speed, slight paddle/ball speed jitter
  - Headless mode: run physics without rendering for high throughput; render only for evaluation/videos
  - Observations: normalize to [-1, 1] (positions / dimensions, velocities / max speed)
  - Actions: map {0: noop, 1: up, 2: down} (optionally include {3: serve} or auto-serve on reset)
  - Opponent policy: start with a fixed scripted opponent; later vary difficulty or sample from a pool

- Algorithm: PPO

- Stabilizers:
  - Observation normalization (running mean/std) and/or manual normalization to stable ranges
  - Advantage normalization, GAE(\(\lambda\)) (common defaults: \(\gamma=0.99\), \(\lambda=0.95\))
  - Gradient clipping, value-function loss coefficient, entropy bonus (prevents premature collapse)
  - Reward scaling (keep returns in a reasonable magnitude); keep sparse scoring as the main signal
  - Learning-rate schedule (linear decay) + early stopping based on KL divergence (PPO-specific)
  - Deterministic evaluation mode (no exploration noise) and fixed evaluation seeds

- Curriculum: domain randomization

- Metrics to be recorded always + videos saved

## training (CPU, transformer ~200k params)

### Setup (always use `venv/`)

```bash
# Setup virtual environment
python3 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

All RL commands below assume:
- You are in the repo root (`pong/`)
- Your virtualenv is activated (`source venv/bin/activate`)

### Verify parameter count (~200k)

```bash
python -m rl.count_params
```

Default config is ~200k params (around ~201k).

### Train (PPO)

python -m rl.train
```

By default this **opens a realtime UI window** (and trains at the same time). It also writes a checkpoint to `runs/pong_transformer_200k.pt`.

**Server / Web UI (Headless Mode)**:
If you are running on a server (no display) or want to view the training remotely:

```bash
# Auto-detected if no DISPLAY is present
python -m rl.train --port 1306
```

Then open `http://<your-server-ip>:1306` in your browser.
- **Tailscale**: If using Tailscale, use your Tailscale IP (e.g., `http://100.x.y.z:1306`).
- **Flags**: Use `--web-ui` to force web mode, `--port` to change the port.

To disable the UI completely:

```bash
python -m rl.train --no-ui
```

### Evaluate (greedy policy)

```bash
python -m rl.eval --ckpt runs/pong_transformer_200k.pt --episodes 200
```

### Watch training progress (metrics + videos)

- **Metrics** are appended every update to `runs/metrics.csv`:

```bash
tail -f runs/metrics.csv
```

- **Videos** are written progressively to `runs/videos/` (**MP4** rollouts every N updates; configurable via flags):

```bash
ls -lt runs/videos | head
```

Useful flags:
- `--log-csv runs/metrics.csv`
- `--eval-every 5`
- `--video-every 20`
- `--eval-episodes 40`

### Observe the model in realtime (auto-reload latest checkpoint)

Run training in one terminal and the watcher in another. The watcher **reloads** the checkpoint whenever it updates.

```bash
python -m rl.watch --ckpt runs/pong_transformer_200k.pt
```

**Web UI Watcher**:
To watch on a server/remote browser:

```bash
python -m rl.watch --web-ui --port 1306 --ckpt runs/pong_transformer_200k.pt
```

Note: `rl.watch` (desktop) uses **Tkinter**. If `import tkinter` fails, use the `--web-ui` flag or `--no-ui` mode.