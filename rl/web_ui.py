from __future__ import annotations

import io
import threading
import time
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import ThreadingMixIn

import numpy as np
import torch
from PIL import Image
from torch.distributions.categorical import Categorical

from .live_ui import LiveShared
from .model import TransformerACConfig, TransformerActorCritic
from .pong_env import PongConfig, PongEnv
from .wrappers import HistoryObsWrapper


import json

class ThreadingHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True


class WebOutput:
    def __init__(self):
        self.frame = None
        self.lock = threading.Lock()

    def set_frame(self, frame_bytes: bytes):
        with self.lock:
            self.frame = frame_bytes

    def get_frame(self):
        with self.lock:
            return self.frame


OUTPUT = WebOutput()


class StreamingHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/":
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            content = """
            <html>
            <head>
                <title>Pong RL Web UI</title>
                <style>
                    body { background: #111; color: #eee; font-family: monospace; text-align: center; margin: 0; padding: 20px; }
                    h1 { margin-bottom: 10px; }
                    .container { display: inline-block; position: relative; }
                    img { border: 2px solid #444; box-shadow: 0 0 20px rgba(0,0,0,0.5); background: #000; }
                    .stats-box { 
                        margin-top: 20px; 
                        padding: 15px; 
                        background: #222; 
                        border: 1px solid #333; 
                        border-radius: 8px; 
                        text-align: left; 
                        display: inline-block;
                        min-width: 600px;
                        font-family: 'Consolas', 'Monaco', monospace;
                    }
                    .stat-row { display: flex; justify-content: space-between; margin-bottom: 5px; }
                    .stat-label { color: #888; }
                    .stat-val { color: #fff; font-weight: bold; }
                    .highlight { color: #4CAF50; }
                    .status-line { margin-bottom: 15px; padding-bottom: 10px; border-bottom: 1px solid #333; font-style: italic; color: #aaa;}
                </style>
                <script>
                    function updateStats() {
                        fetch('/status')
                            .then(response => response.json())
                            .then(data => {
                                document.getElementById('status-text').textContent = data.status;
                                document.getElementById('update').textContent = data.update;
                                document.getElementById('step').textContent = data.global_step;
                                document.getElementById('sps').textContent = data.sps;
                                document.getElementById('ep_ret').textContent = Number(data.ep_ret_mean).toFixed(3);
                                document.getElementById('ep_len').textContent = Number(data.ep_len_mean).toFixed(1);
                                document.getElementById('eval_ret').textContent = Number(data.eval_return_mean).toFixed(3);
                                
                                const wr = Number(data.eval_win_rate);
                                const wrEl = document.getElementById('eval_wr');
                                wrEl.textContent = (wr * 100).toFixed(1) + '%';
                            })
                            .catch(err => console.error(err));
                    }
                    setInterval(updateStats, 1000);
                </script>
            </head>
            <body>
                <h1>Pong RL Training Stream</h1>
                <div class="container">
                    <img src="/stream.mjpg" width="800" height="450" />
                </div>
                <br/>
                <div class="stats-box">
                    <div id="status-text" class="status-line">Waiting for connection...</div>
                    <div class="stat-row"><span class="stat-label">Update:</span> <span id="update" class="stat-val">0</span></div>
                    <div class="stat-row"><span class="stat-label">Global Step:</span> <span id="step" class="stat-val">0</span></div>
                    <div class="stat-row"><span class="stat-label">Steps/Sec:</span> <span id="sps" class="stat-val">0</span></div>
                    <hr style="border-color: #333; opacity: 0.3; margin: 10px 0;">
                    <div class="stat-row"><span class="stat-label">Train Return (Mean):</span> <span id="ep_ret" class="stat-val">0.000</span></div>
                    <div class="stat-row"><span class="stat-label">Train Length (Mean):</span> <span id="ep_len" class="stat-val">0.0</span></div>
                    <hr style="border-color: #333; opacity: 0.3; margin: 10px 0;">
                    <div class="stat-row"><span class="stat-label">Eval Return:</span> <span id="eval_ret" class="stat-val">0.000</span></div>
                    <div class="stat-row"><span class="stat-label">Eval Win Rate:</span> <span id="eval_wr" class="stat-val highlight">0.0%</span></div>
                </div>
            </body>
            </html>
            """
            self.wfile.write(content.encode("utf-8"))
        elif self.path == "/stream.mjpg":
            self.send_response(200)
            self.send_header("Age", "0")
            self.send_header("Cache-Control", "no-cache, private")
            self.send_header("Pragma", "no-cache")
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=FRAME")
            self.end_headers()
            try:
                while True:
                    frame = OUTPUT.get_frame()
                    if frame:
                        self.wfile.write(b"--FRAME\r\n")
                        self.send_header("Content-Type", "image/jpeg")
                        self.send_header("Content-Length", str(len(frame)))
                        self.end_headers()
                        self.wfile.write(frame)
                        self.wfile.write(b"\r\n")
                    time.sleep(1.0 / 30.0)  # Cap streaming FPS
            except Exception:
                pass
        elif self.path == "/status":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            
            # Extract metrics from shared state
            shared = self.server.shared
            data = {}
            if shared:
                with shared.lock:
                    data = {
                        "status": shared.status,
                        "update": shared.update,
                        "global_step": shared.global_step,
                        "sps": shared.sps,
                        "ep_ret_mean": shared.ep_ret_mean,
                        "ep_len_mean": shared.ep_len_mean,
                        "eval_return_mean": shared.eval_return_mean,
                        "eval_win_rate": shared.eval_win_rate,
                    }
            else:
                 data = {"status": "Watcher Mode (Metrics unavailable)", "update":0, "global_step": 0, "sps": 0, "ep_ret_mean": 0, "ep_len_mean": 0, "eval_return_mean": 0, "eval_win_rate": 0}
            
            self.wfile.write(json.dumps(data).encode("utf-8"))
        else:
            self.send_error(404)


class WebTrainingUI:
    def __init__(
        self,
        shared: LiveShared | None,
        *,
        port: int = 1306,
        fps: int = 60,
        scale: int = 1,
        seed: int = 123,
        greedy: bool = False,
        model: TransformerActorCritic | None = None, # For Watcher mode
        env: PongEnv | None = None, # For Watcher mode
    ):
        self.shared = shared
        self.port = port
        self.fps = max(1, int(fps))
        self.scale = max(1, int(scale))
        self.seed = int(seed)
        self.greedy = bool(greedy)

        self.device = torch.device("cpu")
        self.rng = np.random.default_rng(self.seed)

        # Allow passing existing model/env (for Watcher) or creating new ones (for Trainer)
        if model:
            self.model = model
            self.env = env
            self.is_watcher = True
            # For watcher, we assume env is already wrapped if needed, but we check/wrap if we own it
        else:
            self.is_watcher = False
            self.model = TransformerActorCritic(shared.model_cfg).to(self.device) # type: ignore
            self.model.eval()
            self.env = PongEnv(PongConfig(terminate_on_point=True), render_mode="rgb_array")
            self.env = HistoryObsWrapper(self.env, seq_len=shared.model_cfg.seq_len) # type: ignore
            
        if not self.is_watcher:
             self.obs, _ = self.env.reset(seed=int(self.rng.integers(0, 2**31 - 1)))
        
        self._loaded_version = -1
        self._last_action = 0
        
        # Start server
        self.server = ThreadingHTTPServer(("0.0.0.0", self.port), StreamingHandler)
        self.server.shared = self.shared # type: ignore
        self.server_thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.server_thread.start()
        print(f"[WebUI] Server started at http://localhost:{self.port}")

    def _maybe_load_weights(self) -> None:
        if self.is_watcher or self.shared is None:
            return
            
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
            o = torch.from_numpy(obs[None, ...]).to(self.device)
            logits, _ = self.model(o)
            dist = Categorical(logits=logits)
            if self.greedy:
                return int(torch.argmax(logits, dim=-1).item())
            else:
                return int(dist.sample().item())

    def _render_to_buffer(self):
        # Render to RGB array
        frame_rgb = self.env.unwrapped.render_rgb(scale=self.scale) # type: ignore
        
        # Convert to JPEG bytes via PIL
        img = Image.fromarray(frame_rgb)
        
        # Draw some text on the image using PIL (optional, but good for status)
        # For now, just raw frame is fine.
        
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=80)
        OUTPUT.set_frame(buf.getvalue())

    def tick(self) -> None:
        if self.shared and self.shared.stop_event.is_set():
            return

        self._maybe_load_weights()

        # Step environment
        try:
            if not self.is_watcher:
                act = self._policy_action(self.obs) if self._loaded_version >= 0 else 0
                self.obs, _, term, trunc, _ = self.env.step(act)
                if term or trunc:
                    self.obs, _ = self.env.reset(seed=int(self.rng.integers(0, 2**31 - 1)))
        except Exception as e:
            print(f"[WebUI] Error stepping env: {e}")

        # Render
        self._render_to_buffer()

        # Schedule next tick
        pass # Loop handled in run()

    def run(self) -> None:
        try:
            while True:
                if self.shared and self.shared.stop_event.is_set():
                    break
                
                start_t = time.time()
                self.tick()
                
                # Sleep to maintain FPS
                elapsed = time.time() - start_t
                delay = max(0, (1.0 / self.fps) - elapsed)
                time.sleep(delay)
        except KeyboardInterrupt:
            pass
        finally:
            self.server.shutdown()
