# BalanceBot — Self-Balancing Two-Wheel Robot (MuJoCo + PPO)

A two-wheeled, self-balancing (inverted-pendulum style) robot trained with
reinforcement learning in [MuJoCo](https://mujoco.org/), using a custom
[Gymnasium](https://gymnasium.farama.org/) environment and
[Stable-Baselines3](https://stable-baselines3.readthedocs.io/) PPO.

The robot learns to stand upright and hold its position from IMU (accelerometer
+ gyro) and wheel sensor readings alone — no external position tracking, no
scripted controller (PID/LQR), just a learned policy.

## Demo

The trained policy balances indefinitely from a randomized initial tilt/angular
velocity, correcting for drift using wheel odometry and resisting yaw/position
disturbance. See `watch_trained.ipynb` to load a checkpoint and watch it run in
the MuJoCo passive viewer.

## Project structure

```
.
├── robot.xml              # MJCF model: differential-drive robot, wheels, IMU, sensors
├── env_for_bbot.py         # Custom Gymnasium environment (BalanceBotEnv)
├── train.py                # PPO training script (Stable-Baselines3, SubprocVecEnv)
├── watch_trained.ipynb      # Load a trained checkpoint and visualize it in the viewer
├── env_test.py / env_test2.ipynb   # Environment sanity checks / manual testing
├── mujoco_interface.py     # Low-level MuJoCo helper utilities
├── meshes/                  # STL meshes referenced by robot.xml
├── working_model/           # Known-good checkpoint + VecNormalize stats (tracked in git)
├── check_points/            # Periodic training checkpoints (gitignored)
├── models/                  # Final saved models (gitignored)
└── tb_logs/                 # TensorBoard logs (gitignored)
```

`check_points/`, `models/`, and `tb_logs/` are excluded from version control
(see `.gitignore`) since they contain large, frequently-regenerated binary
artifacts. `working_model/` holds a specific checkpoint known to perform well
and is committed intentionally.

## Environment

`BalanceBotEnv` (in `env_for_bbot.py`) wraps a MuJoCo simulation of the robot
defined in `robot.xml`.

**Observation space** (5-dim, continuous):
| Index | Quantity | Source |
|---|---|---|
| 0 | Pitch (rad) | Complementary filter over accelerometer + gyro |
| 1 | Pitch rate (rad/s) | Gyro |
| 2 | Left wheel velocity (rad/s) | `left_wheel_vel` sensor |
| 3 | Right wheel velocity (rad/s) | `right_wheel_vel` sensor |
| 4 | Average wheel position delta (rad) | `left_wheel_pos` / `right_wheel_pos` sensors, relative to the pose at episode reset |

The last observation (wheel position delta) gives the policy a way to perceive
net drift over time — without it, the agent can react to instantaneous tilt
but has no signal for "I've been rolling forward," which was an early failure
mode during development.

**Action space** (2-dim, continuous, normalized `[-1, 1]`): left/right wheel
motor torque commands.

**Reward** combines:
- an alive bonus for staying upright each step,
- a penalty on pitch² (discourages leaning),
- a small penalty on action magnitude² (discourages jitter),
- a penalty on distance drifted from the start position,
- a penalty on yaw rate (discourages spinning in place),
- a light penalty on wheel speed² (discourages unnecessary spinning, tuned low
  enough not to suppress the corrective motion needed to catch a fall).

**Episode termination:** the episode ends if pitch exceeds a configurable tilt
threshold (default 30°), or is truncated at `max_steps`.

## Training

```bash
python train.py
```

Trains PPO (`MlpPolicy`) across 4 parallel `SubprocVecEnv` workers with
`VecNormalize` for observation/reward normalization. Checkpoints (model +
VecNormalize stats) are saved every 50k steps to `check_points/`, and
TensorBoard logs are written to `tb_logs/`.

```bash
tensorboard --logdir tb_logs
```

Key hyperparameters (see `train.py` for the full list): `n_steps=2048`,
`batch_size=256`, `n_epochs=10`, `learning_rate=3e-4`, `gamma=0.99`,
`gae_lambda=0.95`, `clip_range=0.2`.

### Training results

Across a 10M-step run, `ep_len_mean` and `ep_rew_mean` rose steadily and
peaked around **step ~4.45M**, where the robot reliably balanced for full
10,000-step episodes. Training continued past that point degraded — a
catastrophic-forgetting dip in both metrics before partially recovering by
10M, likely from continued policy updates on an already-converged task
without a decaying learning rate.

**The checkpoint at step 4.45M (`working_model/`) is the recommended model to
use** — later checkpoints in the same run are not guaranteed to perform
better and were empirically worse. If resuming training to push further,
resume from this checkpoint specifically, with a decayed learning rate rather
than continuing from the final 10M-step model.

## Watching a trained model

Open `watch_trained.ipynb` and set:

```python
MODEL_PATH   = "working_model/bbot_ppo_4450000_steps"
VECNORM_PATH = "working_model/bbot_ppo_vecnormalize_4450000_steps.pkl"
XML_PATH     = "robot.xml"
```

This loads the PPO policy and matching `VecNormalize` statistics, runs it
deterministically in the MuJoCo passive viewer, and resets automatically on
episode termination.

> **Note:** the model and VecNormalize file must come from the *same*
> training checkpoint — their observation space shapes are tied together, and
> `VecNormalize.load()` will raise an `AssertionError` on any mismatch.

## Requirements

- Python 3.13
- `mujoco`
- `gymnasium`
- `stable-baselines3`
- `numpy`

```bash
pip install mujoco gymnasium stable-baselines3 numpy
```

## Known limitations / next steps

- The robot's torque authority (`gear` in `robot.xml`) has not been
  systematically tuned against the body mass — if action saturation is
  observed during training, this is a likely place to look.
- No sim-to-real transfer has been attempted; the policy has only been
  validated in simulation.
- Training used a fixed learning rate for the full run; a linear decay
  schedule may avoid the late-training instability seen after ~4.5M steps.