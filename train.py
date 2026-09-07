import os
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv,VecNormalize
from env_for_bbot import BalanceBotEnv
N_ENVS=4
XML_PATH=os.path.abspath("robot.xml")
TOTAL_TIME_STEPS=6_000_000

def make_env():
    def _init():
        env=BalanceBotEnv(mjcf_path=XML_PATH)
        env=Monitor(env)
        return env
    return _init

if __name__ == "__main__":
    os.makedirs("tb_logs",exist_ok=True)
    os.makedirs("check_points",exist_ok=True)
    os.makedirs("models",exist_ok=True)

    vec_env=SubprocVecEnv([make_env() for _ in range (N_ENVS)])
    vec_env = VecNormalize(vec_env,norm_obs=True,norm_reward=True,clip_obs=10.0)

    model=PPO(
        "MlpPolicy",
        vec_env,
        verbose=1,
        n_steps=2048,
        batch_size=256,
        n_epochs=10,
        learning_rate=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.005,
        vf_coef=0.5,
        max_grad_norm=0.5,
        tensorboard_log="./tb_logs",
        policy_kwargs=dict(log_std_init=-1.0),
    )
    check_point_callback=CheckpointCallback(
        save_freq=max(50_000//N_ENVS,1),
        save_path="./check_points",
        name_prefix="bbot_ppo",
        save_vecnormalize=True,
    )

    model.learn(
        total_timesteps=TOTAL_TIME_STEPS,
        callback=check_point_callback,
        tb_log_name="ppo_run",
    )
    model.save("models/bbot_ppo_final")
    vec_env.save("models/vecnormalize_final.pkl")
    print("Training complete. Saved to models/quadruped_ppo_final.zip")