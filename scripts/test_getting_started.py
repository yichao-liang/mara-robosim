"""Smoke-test that mirrors the getting_started notebook."""

from pathlib import Path

import numpy as np
from PIL import Image

import mara_robosim
from mara_robosim.config import PyBulletConfig
from mara_robosim.structs import State

OUT_DIR = Path(__file__).resolve().parent / "getting_started_output"
OUT_DIR.mkdir(exist_ok=True)

# ── 1. Discover environments ──────────────────────────────────────────

mara_robosim.register_all_environments()
env_ids = sorted(mara_robosim.get_all_env_ids())
print(f"[1/6] Found {len(env_ids)} environments")
assert len(env_ids) == 15, f"Expected 15 environments, got {len(env_ids)}"
for eid in env_ids:
    print(f"       {eid}")

# ── 2. Create environment ────────────────────────────────────────────

config = PyBulletConfig(camera_width=1674, camera_height=900)
env = mara_robosim.make("mara/Blocks-v0", render_mode="rgb_array", config=config)
obs, info = env.reset()
print(f"\n[2/6] Created mara/Blocks-v0")
assert isinstance(obs, np.ndarray)
assert obs.shape == env.observation_space.shape

# ── 3. Observation and action spaces ─────────────────────────────────

print(f"\n[3/6] Observation shape: {env.observation_space.shape}")
print(f"       Action shape:      {env.action_space.shape}")
assert len(env.observation_space.shape) == 1
assert env.observation_space.shape[0] > 0
assert len(env.action_space.shape) >= 1

action = env.action_space.sample()
obs, reward, terminated, truncated, info = env.step(action)
assert isinstance(reward, float)
assert isinstance(terminated, bool)
assert isinstance(truncated, bool)
print(f"       Step OK — reward={reward}, terminated={terminated}")

# ── 4. Structured state in info dict ─────────────────────────────────

state = info["state"]
assert isinstance(state, State), f"Expected State, got {type(state)}"
assert "goal_reached" in info
assert isinstance(info["goal_reached"], bool)
print(f"\n[4/6] Structured state OK — goal_reached={info['goal_reached']}")
print(state.pretty_str())

# ── 5. Rendering ─────────────────────────────────────────────────────

frame = env.render()
assert frame is not None, "render() returned None — rendering is broken"
assert isinstance(frame, np.ndarray), f"Expected ndarray, got {type(frame)}"
assert frame.ndim == 3, f"Expected 3D image array, got shape {frame.shape}"
assert frame.shape[2] == 3, f"Expected RGB (3 channels), got {frame.shape[2]}"
img_path = OUT_DIR / "blocks_initial.png"
Image.fromarray(frame).save(img_path)
print(f"\n[5/6] Render OK — frame shape {frame.shape}, saved to {img_path}")

# ── 6. Multi-step rollout with rendering ─────────────────────────────

obs, info = env.reset()
frames = []
frame = env.render()
if frame is not None:
    frames.append(frame)

for _ in range(50):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    frame = env.render()
    if frame is not None:
        frames.append(frame)
    if terminated or truncated:
        break

assert len(frames) > 0, "No frames captured during rollout"

gif_path = OUT_DIR / "blocks_rollout.gif"
pil_frames = [Image.fromarray(f) for f in frames]
pil_frames[0].save(
    gif_path,
    format="GIF",
    save_all=True,
    append_images=pil_frames[1:],
    duration=100,
    loop=0,
)
print(f"\n[6/6] Rollout OK — collected {len(frames)} frames, saved to {gif_path}")

env.close()
print("\nAll checks passed!")
