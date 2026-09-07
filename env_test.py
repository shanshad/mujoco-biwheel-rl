from pathlib import Path
import sys
import time

from gymnasium.utils.env_checker import check_env
import numpy as np
from env_for_bbot import BalanceBotEnv
MJCF_PATH=Path(r"C:\Users\LENOVO\Desktop\mujoco_rl_biwheel\robot.xml")
SEED=42
env=BalanceBotEnv(mjcf_path=MJCF_PATH,render_mode="human")
print(env.observation_space)
print(env.action_space)
obs,info=env.reset(seed=SEED)
print(obs)