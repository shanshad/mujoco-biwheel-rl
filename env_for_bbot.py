import math
from pathlib import Path
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import mujoco
import mujoco.viewer

class BalanceBotEnv(gym.Env):
    metadata={"render_modes": ["human","rgb_array"],"render_fps":30}
    def __init__(
            self,
            mjcf_path="robot.xml",max_steps=10000,
            render_mode=None,
            alpha=0.99,
            sensor_imu_accel="imu_accel",
            sensor_imu_gyro="imu_gyro",
            sensor_left_wheel_vel="left_wheel_vel",
            sensor_right_wheel_vel="right_wheel_vel",
            actuator_left_motor="left_motor",
            actuator_right_motor="right_motor",
            sensor_left_wheel_pos="left_wheel_pos",
            sensor_right_wheel_pos="right_wheel_pos",
            alive_bonus=1.0,
            pitch_penalty=5.0,
            action_penalty_coef=0.01,
            position_penalty_coef=0.02,
            yaw_penalty_coef=0.1,
            tip_threshold_deg=30.0,
    ):
        self.model=mujoco.MjModel.from_xml_path(str(mjcf_path))
        self.data=mujoco.MjData(self.model)
        self.render_mode=render_mode
        self.alpha=alpha
        self.sensor_imu_accel=sensor_imu_accel
        self.sensor_imu_gyro=sensor_imu_gyro
        self.sensor_left_wheel_vel=sensor_left_wheel_vel
        self.sensor_right_wheel_vel=sensor_right_wheel_vel
        self.left_motor_id=mujoco.mj_name2id(self.model,mujoco.mjtObj.mjOBJ_ACTUATOR,actuator_left_motor)
        self.right_motor_id=mujoco.mj_name2id(self.model,mujoco.mjtObj.mjOBJ_ACTUATOR,actuator_right_motor)
        self.base_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "base")

        #Define observation space (ie what the agent can see) and limits
        # [pitch, pitch_rate, wheel_vel_left, wheel_vel_right, avg_wheel_pos_delta]
        obs_low=np.array([-np.pi,-20,-50,-50,-1e4],dtype=np.float32)
        obs_high=np.array([np.pi,20.0,50.0,50.0,1e4],dtype=np.float32)
        self.observation_space=spaces.Box(obs_low,obs_high,dtype=np.float32)

        self.sensor_left_wheel_pos=sensor_left_wheel_pos
        self.sensor_right_wheel_pos=sensor_right_wheel_pos

        #define action space (i.e what the agent can do) and limits, normalized to [-1,1]
        #[left_wheel_torque,right_wheel_torque]
        actions_low=np.array([-1.0,-1.0],dtype=np.float32)
        actions_high=np.array([1.0,1.0],dtype=np.float32)
        self.action_space=spaces.Box(actions_low,actions_high,dtype=np.float32)

        # reward coefficients
        self.alive_bonus=alive_bonus
        self.pitch_penlaty_coef=pitch_penalty
        self.action_penalty_coef=action_penalty_coef
        self.position_penalty_coef=position_penalty_coef
        self.yaw_penalty_coef=yaw_penalty_coef

        self.tip_threshold_deg=tip_threshold_deg
        self.max_steps=max_steps
        self.pitch=0
        self._viewer=None
        self._step=0

    def _get_obs(self):
        accel_x,_,accel_z=self.data.sensor(self.sensor_imu_accel).data
        pitch_rate=self.data.sensor(self.sensor_imu_gyro).data[1]
        #acceleromeer-derived pitch estimate
        accel_pitch=-1*math.atan2(accel_x,accel_z)
        #complematary filter to estimate pitch from accelerometer and gyroscope
        self._pitch=self.alpha*(self._pitch+pitch_rate*self.model.opt.timestep)+(1-self.alpha)*accel_pitch

        wheel_vel_left=self.data.sensor(self.sensor_left_wheel_vel).data[0]
        wheel_vel_right=self.data.sensor(self.sensor_right_wheel_vel).data[0]

        left_pos=self.data.sensor(self.sensor_left_wheel_pos).data[0]
        right_pos=self.data.sensor(self.sensor_right_wheel_pos).data[0]
        avg_wheel_pos_delta = 0.5*(left_pos+right_pos) - self._wheel_pos_ref

        return np.array([self._pitch,pitch_rate,wheel_vel_left,wheel_vel_right,avg_wheel_pos_delta],dtype=np.float32)

    def reset(self,seed=None,options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model,self.data)
        self._pitch=0.0
        #import an initial angular velocity around the y axis so the agent learns to recover
        #qvel[4]=wy (rad/s)
        self.data.qvel[4]+=self.np_random.uniform(-0.5,0.5)
        mujoco.mj_forward(self.model,self.data)
        left_pos0 = self.data.sensor(self.sensor_left_wheel_pos).data[0]
        right_pos0 = self.data.sensor(self.sensor_right_wheel_pos).data[0]
        self._wheel_pos_ref = 0.5 * (left_pos0 + right_pos0)
        self._step=0
        return self._get_obs(),{}

    def step(self,action):
        #advance the simulation by one step and return the resulte
        #set motors to given (normalized torque)
        self.data.ctrl[self.left_motor_id]=action[0]
        self.data.ctrl[self.right_motor_id]=action[1]

        #advance simulation by one step
        mujoco.mj_step(self.model,self.data)
        self._step+=1

        #get observation
        obs=self._get_obs()
        pitch=obs[0]
        pitch_rate=obs[1]
        wheel_vel_left=obs[2]
        wheel_vel_right=obs[3]

        #reward function: alive - (A*pitch^2)-(B*action^2) - (c*(x^2)+y^2))-d*abs(yaw)
        #Note: qpos (simulation state) and qvel only available during training
        #alive: reward for staying upright each step
        #pitch :penal;ty for leaning
        #action : penalty for jittery motor commands
        #position : penalty for drifting from the starting position
        #yaw penalty for rotating around z axis
        pitch_penalty=self.pitch_penlaty_coef*pitch**2
        action_penalty=self.action_penalty_coef*np.sum(action**2)
        # x_pos=self.data.qpos[0]
        # y_pos=self.data.qpos[1]
        x_pos = self.data.xpos[self.base_body_id][0]
        y_pos = self.data.xpos[self.base_body_id][1]
        distance_from_center = math.sqrt(x_pos**2 + y_pos**2)
        position_penalty = self.position_penalty_coef * distance_from_center
        wheel_speed_penalty = 0.005 * (wheel_vel_left**2 + wheel_vel_right**2)
        #yaw_rate=self.data.qvel[5]
        yaw_rate = self.data.sensor(self.sensor_imu_gyro).data[2]
        yaw_penalty=self.yaw_penalty_coef*abs(yaw_rate)
        reward = self.alive_bonus - pitch_penalty - action_penalty - position_penalty - yaw_penalty - wheel_speed_penalty
        terminated = abs(pitch) > math.radians(self.tip_threshold_deg)
        truncated=self._step >= self.max_steps
        return obs,reward,terminated,truncated, {}
    def render(self):
        #render the current simulation state to the mujoco viewer window
        if self.render_mode !="human":
            return
        if self._viewer is None:
            self._viewer = mujoco.viewer.launch_passive(self.model,self.data)

            self._viewer.cam.type=mujoco.mjtCamera.mjCAMERA_FREE
            self._viewer.cam.lookat[:]=[0,0,0.05]
            self._viewer.cam.distance=0.8
            self._viewer.cam.elevation=-25
        self._viewer.sync()

    def close(self):
        if self._viewer is not None:
            self._viewer.close()
            self._viewer=None
