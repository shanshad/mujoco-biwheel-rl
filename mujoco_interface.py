import mujoco
import mujoco.viewer
import time

MOTOR_SPEED_STEP = 0.2
MOTOR_SPEED_LIMIT = 5.0
PRINT_EVERY = 50

LEFT_MOTOR = "left_motor"
RIGHT_MOTOR = "right_motor"
IMU_ACCEL="imu_accel"
IMU_GYRO="imu_gyro"
LEFT_WHEEL_POS="left_wheel_pos"
RIGHT_WHEEL_POS="right_wheel_pos"
LEFT_WHEEL_VEL="left_wheel_vel"
RIGHT_WHEEL_VEL="right_wheel_vel"


KEY_BACKSPACE = 259
KEY_UP = 265
KEY_DOWN = 264

ctrl = {"left": 0.0, "right": 0.0}

def clamp(x):
    return max(-MOTOR_SPEED_LIMIT, min(MOTOR_SPEED_LIMIT, x))

def key_callback(keycode):
    if keycode == KEY_BACKSPACE:
        ctrl["left"] = ctrl["right"] = 0.0
        return
    if keycode == KEY_UP:
        # Move forward: Add speed
        ctrl["left"] = clamp(ctrl["left"] + MOTOR_SPEED_STEP)
        ctrl["right"] = clamp(ctrl["right"] + MOTOR_SPEED_STEP)
    elif keycode == KEY_DOWN:
        # Reverse: Subtract speed
        ctrl["left"] = clamp(ctrl["left"] - MOTOR_SPEED_STEP)
        ctrl["right"] = clamp(ctrl["right"] - MOTOR_SPEED_STEP)

# 1. Load your XML file
model = mujoco.MjModel.from_xml_path(r"C:\Users\LENOVO\Desktop\mujoco_rl_biwheel\robot.xml")
data = mujoco.MjData(model)

left_motor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, LEFT_MOTOR)
right_motor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, RIGHT_MOTOR)

mujoco.mj_resetData(model, data)
steps = 0

# 2. Launch the interactive GUI window
with mujoco.viewer.launch_passive(model, data, key_callback=key_callback) as viewer:
    print("MuJoCo GUI launched! Press Ctrl+C in terminal to stop.")
    viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    viewer.cam.lookat[:] = [0, 0, 0.05]
    viewer.cam.distance = 0.8
    viewer.cam.azimuth = 45
    viewer.cam.elevation = -25
    
    while viewer.is_running():
        step_start = time.time()
        
        # Apply the control values to the actuator indices
        data.ctrl[left_motor_id] = ctrl["left"]
        data.ctrl[right_motor_id] = ctrl["right"]
        
        # Advance physics in the background
        mujoco.mj_step(model, data)

        # Sync the visual window with the physics engine
        viewer.sync()

        # 3. Read your sensor values while it simulates
        steps += 1
        if steps % PRINT_EVERY == 0:
            accel = data.sensor(IMU_ACCEL).data
            gyro = data.sensor(IMU_GYRO).data
            # Fixed names here to match XML "left_wheel_pos" / "right_wheel_pos"
            left_pos = data.sensor(LEFT_WHEEL_POS).data
            right_pos = data.sensor(RIGHT_WHEEL_POS).data

            print(f"Accel: {accel}")
            print(f"Gyro: {gyro}")
            print(f"Left Wheel Pos: {left_pos}")
            print(f"Right Wheel Pos: {right_pos}")

        slack = model.opt.timestep - (time.time() - step_start)
        if slack > 0:
            time.sleep(slack)
