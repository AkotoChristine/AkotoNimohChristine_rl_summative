# fire_rescue_env_discrete.py - FIXED VERSION
import math
import random
import time
import numpy as np
import pybullet as p
import pybullet_data
from gymnasium import Env, spaces
from typing import List, Tuple

# ---------------------------- FIRE PARTICLE SYSTEM ----------------------------
class FireParticle:
    def __init__(self, pos):
        self.radius = random.uniform(0.01, 0.03)
        self.visual = p.createVisualShape(
            p.GEOM_SPHERE, radius=self.radius, rgbaColor=[1,0.5,0,0.9]
        )
        self.multi = p.createMultiBody(baseMass=0, baseVisualShapeIndex=self.visual, basePosition=pos)
        self.birth = time.time()
        self.vx = random.uniform(-0.03,0.03)
        self.vy = random.uniform(-0.03,0.03)
        self.vz = random.uniform(0.02,0.08)
        self._destroyed = False  # Track if already destroyed

    def age(self):
        return time.time() - self.birth

    def alive(self):
        return not self._destroyed and self.age() < 0.8

    def step(self):
        if self._destroyed:
            return
        
        try:
            # Check if body still exists before accessing
            pos, _ = p.getBasePositionAndOrientation(self.multi)
            new_pos = [pos[0]+self.vx, pos[1]+self.vy, pos[2]+self.vz]
            p.resetBasePositionAndOrientation(self.multi, new_pos, [0,0,0,1])
        except Exception as e:
            # Body doesn't exist anymore, mark as destroyed
            self._destroyed = True

    def destroy(self):
        if not self._destroyed:
            self._destroyed = True
            try:
                p.removeBody(self.multi)
            except:
                pass  # Body already removed or doesn't exist

# ---------------------------- FIRE RENDERER ----------------------------
class FireRenderer:
    def __init__(self):
        self.env = None
        self.particles = []

    def bind_env(self, env):
        self.env = env

    def step(self):
        if not self.env:
            return
        
        # Spawn new particles
        for fpos in self.env.fire_positions:
            base = [float(fpos[0]), float(fpos[1]), 0.45]
            for _ in range(3):
                try:
                    self.particles.append(FireParticle(base))
                except Exception as e:
                    # If particle creation fails, skip it
                    continue

        # Update and clean up particles
        alive = []
        for ptl in self.particles:
            if ptl.alive():
                ptl.step()
                if ptl.alive():  # Check again after stepping
                    alive.append(ptl)
                else:
                    ptl.destroy()
            else:
                ptl.destroy()
        
        self.particles = alive

    def cleanup(self):
        """Safely remove all particles"""
        for ptl in self.particles:
            ptl.destroy()
        self.particles.clear()

# ---------------------------- FIRE RESCUE ENV (DISCRETE ACTIONS) ----------------------------
class FireRescueEnv(Env):
    """
    Fire Rescue Environment with DISCRETE action space.
    Compatible with DQN, A2C, PPO, and REINFORCE.
    
    Actions:
        0: Forward
        1: Backward
        2: Strafe Left
        3: Strafe Right
        4: Turn Left
        5: Turn Right
        6: Forward + Turn Left
        7: Forward + Turn Right
        8: Stop
    """
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 120}

    def __init__(self, render_mode=None, max_steps=500, headless=False):
        super().__init__()
        self.render_mode = render_mode
        self.headless = headless
        self.physics_client = None
        self.robot = None
        self.victim_id = None
        self.victim_parts = []
        self.victim_pos = None
        self.fire_ids = []
        self.fire_positions = []
        self.max_steps = max_steps
        self.episode_steps = 0
        self.cumulative_reward = 0.0
        self.fire_renderer = FireRenderer()

        # DISCRETE ACTION SPACE (9 actions)
        self.action_space = spaces.Discrete(9)

        # Observation: 10D vector
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(10,), dtype=np.float32
        )

        # Room info
        self._room_half = 4.0
        self._wall_penalty_distance = 0.4
        self._upright_threshold = 0.6

    def _action_to_continuous(self, action):
        """Convert discrete action to continuous [forward, strafe, turn]"""
        action_map = {
            0: [1.0, 0.0, 0.0],    # Forward
            1: [-1.0, 0.0, 0.0],   # Backward
            2: [0.0, 1.0, 0.0],    # Strafe Left
            3: [0.0, -1.0, 0.0],   # Strafe Right
            4: [0.0, 0.0, 1.0],    # Turn Left
            5: [0.0, 0.0, -1.0],   # Turn Right
            6: [1.0, 0.0, 0.5],    # Forward + Turn Left
            7: [1.0, 0.0, -0.5],   # Forward + Turn Right
            8: [0.0, 0.0, 0.0],    # Stop
        }
        return action_map.get(action, [0.0, 0.0, 0.0])

    # ---------------------------- ROOM ----------------------------
    def _build_room(self):
        wall_thickness = 0.1
        half = self._room_half
        walls = [
            ([0,-half,1],[half,wall_thickness,1]),
            ([0, half,1],[half,wall_thickness,1]),
            ([-half,0,1],[wall_thickness,half,1]),
            ([half,0,1],[wall_thickness,half,1])
        ]
        for pos, ext in walls:
            col = p.createCollisionShape(p.GEOM_BOX, halfExtents=ext)
            vis = p.createVisualShape(p.GEOM_BOX, halfExtents=ext, rgbaColor=[0.8,0.8,0.8,1])
            p.createMultiBody(baseCollisionShapeIndex=col, baseVisualShapeIndex=vis, basePosition=pos)

    # ---------------------------- ROBOT ----------------------------
    def _spawn_robot(self):
        try:
            self.robot = p.loadURDF("r2d2.urdf", basePosition=[0,0,0.12], globalScaling=0.5)
            num_joints = p.getNumJoints(self.robot)
            for j in range(num_joints):
                p.changeDynamics(self.robot, j, lateralFriction=1.0)
            p.changeDynamics(self.robot, -1, lateralFriction=1.0)
        except:
            col = p.createCollisionShape(p.GEOM_CYLINDER, radius=0.2, height=0.4)
            vis = p.createVisualShape(p.GEOM_CYLINDER, radius=0.2, length=0.4, rgbaColor=[0.2,0.5,1,1])
            self.robot = p.createMultiBody(baseMass=5, baseCollisionShapeIndex=col, baseVisualShapeIndex=vis,
                                           basePosition=[0,0,0.2])

        p.resetBasePositionAndOrientation(self.robot, [0,0,0.12], [0,0,0,1])
        p.resetBaseVelocity(self.robot, [0,0,0], [0,0,0])

    # ---------------------------- VICTIM ----------------------------
    def _spawn_victim(self):
        for _ in range(200):
            x = random.uniform(-3.2, 3.2)
            y = random.uniform(-3.2, 3.2)
            if np.linalg.norm(np.array([x,y])) > 0.8:
                break
        base_z = 0.2

        lying_orientation = p.getQuaternionFromEuler([0, math.pi/2, 0])
        parts = []

        head = p.createMultiBody(
            0,
            p.createCollisionShape(p.GEOM_SPHERE, radius=0.15),
            p.createVisualShape(p.GEOM_SPHERE, radius=0.15, rgbaColor=[1,0.8,0.6,1]),
            [x+0.6,y,base_z+0.15],
            lying_orientation
        )
        parts.append(head)

        body = p.createMultiBody(
            0,
            p.createCollisionShape(p.GEOM_CAPSULE, radius=0.2, height=0.5),
            p.createVisualShape(p.GEOM_CAPSULE, radius=0.2, length=0.5, rgbaColor=[0.2,0.6,1,1]),
            [x+0.2,y,base_z+0.2],
            lying_orientation
        )
        parts.append(body)

        leg_col = p.createCollisionShape(p.GEOM_CAPSULE, radius=0.08, height=0.6)
        leg_vis = p.createVisualShape(p.GEOM_CAPSULE, radius=0.08, length=0.6, rgbaColor=[0.3,0.3,0.8,1])
        left_leg = p.createMultiBody(0, leg_col, leg_vis, [x-0.35,y+0.1,base_z+0.08], lying_orientation)
        right_leg = p.createMultiBody(0, leg_col, leg_vis, [x-0.35,y-0.1,base_z+0.08], lying_orientation)
        parts.extend([left_leg,right_leg])

        self.victim_id = body
        self.victim_pos = np.array([x, y])
        self.victim_parts = parts
        return body, parts, self.victim_pos

    # ---------------------------- FIRE ----------------------------
    def _spawn_fire_boxes(self):
        self.fire_ids = []
        self.fire_positions = []
        n = random.randint(10,13)
        attempts = 0
        spawned = 0

        while spawned < n and attempts < 200:
            x = random.uniform(-2.5,2.5)
            y = random.uniform(-2.5,2.5)

            if self.victim_pos is not None and np.linalg.norm(np.array([x,y])-self.victim_pos) < 1.2:
                attempts += 1
                continue
            if np.linalg.norm(np.array([x,y])) < 0.8:
                attempts += 1
                continue

            too_close = False
            for f in self.fire_positions:
                if np.linalg.norm(np.array([x,y])-f) < 0.7:
                    too_close = True
                    break

            if too_close:
                attempts += 1
                continue

            vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.35,0.35,0.2], rgbaColor=[1,0.2,0,1])
            col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.35,0.35,0.2])
            fire_id = p.createMultiBody(0, col, vis, [x,y,0.2])
            self.fire_ids.append(fire_id)
            self.fire_positions.append(np.array([x,y]))
            spawned += 1
            attempts += 1

    # ---------------------------- INTERNAL RESET ----------------------------
    def _internal_reset_scene(self):
        # CRITICAL: Clean up particles FIRST before removing fire boxes
        self.fire_renderer.cleanup()
        
        for part in self.victim_parts:
            try: p.removeBody(part)
            except: pass

        for fid in self.fire_ids:
            try: p.removeBody(fid)
            except: pass

        p.resetBasePositionAndOrientation(self.robot, [0,0,0.12],[0,0,0,1])
        p.resetBaseVelocity(self.robot,[0,0,0],[0,0,0])

        self._spawn_victim()
        self._spawn_fire_boxes()

        rp,_ = p.getBasePositionAndOrientation(self.robot)
        self.prev_dist_to_victim = np.linalg.norm(np.array(rp[:2]) - self.victim_pos)
        self.episode_steps = 0

    # ---------------------------- RESET ----------------------------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Clean up particles before disconnecting
        if hasattr(self, 'fire_renderer'):
            self.fire_renderer.cleanup()

        if p.isConnected():
            p.disconnect()

        if self.headless or self.render_mode != "human":
            self.physics_client = p.connect(p.DIRECT)
        else:
            self.physics_client = p.connect(p.GUI)
            
        p.resetSimulation()
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0,0,-9.8)
        p.configureDebugVisualizer(p.COV_ENABLE_GUI,0)
        p.setTimeStep(1/240)

        p.loadURDF("plane.urdf")
        self._build_room()
        self._spawn_robot()
        self._spawn_victim()
        self._spawn_fire_boxes()
        self.fire_renderer.bind_env(self)

        rp,_ = p.getBasePositionAndOrientation(self.robot)
        self.prev_dist_to_victim = np.linalg.norm(np.array(rp[:2]) - self.victim_pos)

        self.episode_steps = 0
        self.cumulative_reward = 0.0

        return self._get_obs(), {}

    # ---------------------------- OBS ----------------------------
    def _get_obs(self):
        rp,_ = p.getBasePositionAndOrientation(self.robot)
        robot_xy = np.array(rp[:2])

        vp = self.victim_pos if self.victim_pos is not None else np.zeros(2)

        if self.fire_positions:
            dists = [np.linalg.norm(robot_xy - f) for f in self.fire_positions]
            idx = int(np.argmin(dists))
            nf = self.fire_positions[idx]
            nd = dists[idx]
        else:
            nf = np.zeros(2)
            nd = 10.0

        return np.array([
            robot_xy[0], robot_xy[1],
            vp[0], vp[1],
            nf[0], nf[1],
            nd,
            0,0,0
        ], dtype=np.float32)

    # ---------------------------- RENDER ----------------------------
    def render(self):
        if self.render_mode == "rgb_array":
            view_matrix = p.computeViewMatrixFromYawPitchRoll(
                cameraTargetPosition=[0, 0, 0],
                distance=8.0,
                yaw=45, pitch=-30, roll=0,
                upAxisIndex=2
            )

            proj_matrix = p.computeProjectionMatrixFOV(
                fov=60, aspect=1.0, nearVal=0.1, farVal=100.0
            )

            (_, _, px, _, _) = p.getCameraImage(
                width=640, height=480,
                viewMatrix=view_matrix,
                projectionMatrix=proj_matrix,
                renderer=p.ER_BULLET_HARDWARE_OPENGL
            )

            rgb_array = np.array(px, dtype=np.uint8)
            rgb_array = np.reshape(rgb_array, (480, 640, 4))
            return rgb_array[:, :, :3]

        return None

    # ---------------------------- STEP ----------------------------
    def step(self, action):
        # Convert discrete action to continuous
        forward, strafe, turn = self._action_to_continuous(action)

        pos, orn = p.getBasePositionAndOrientation(self.robot)
        yaw = p.getEulerFromQuaternion(orn)[2]

        base_speed = 6.0

        rp, _ = p.getBasePositionAndOrientation(self.robot)
        robot_xy = np.array(rp[:2])
        dist = np.linalg.norm(robot_xy - self.victim_pos)

        slow_radius = 1.2
        if dist < slow_radius:
            speed_scale = max(0.12, dist / slow_radius)
        else:
            speed_scale = 1.0

        speed = base_speed * speed_scale

        vx = speed*forward*math.cos(yaw) - speed*strafe*math.sin(yaw)
        vy = speed*forward*math.sin(yaw) + speed*strafe*math.cos(yaw)

        close_brake_radius = 0.75
        if dist < close_brake_radius:
            vx *= 0.35
            vy *= 0.35

        vel_vec = np.array([vx, vy])
        to_v = (self.victim_pos - robot_xy)
        to_v_norm = np.linalg.norm(to_v)
        if to_v_norm > 1e-6:
            to_v_unit = to_v / to_v_norm
            vel_norm = np.linalg.norm(vel_vec)
            if vel_norm > 1e-6:
                vel_unit = vel_vec / vel_norm
                if np.dot(vel_unit, to_v_unit) < -0.15 and dist < 1.0:
                    vx *= 0.2
                    vy *= 0.2

        p.resetBaseVelocity(self.robot, [vx, vy, 0], [0,0,turn*1.8])

        for _ in range(8):
            p.stepSimulation()

        rp, orn = p.getBasePositionAndOrientation(self.robot)
        robot_xy = np.array(rp[:2])
        robot_z = rp[2]
        base_v, base_omega = p.getBaseVelocity(self.robot)
        base_v = np.array(base_v[:2])

        reward = -0.003

        reward += max(0.0, 1.6 - dist) * 2.0

        dist_delta = (self.prev_dist_to_victim - dist)
        reward += np.clip(dist_delta, -0.05, 0.2) * 2.0
        self.prev_dist_to_victim = dist

        if np.linalg.norm(base_v) > 1e-4 and to_v_norm > 1e-6:
            v_unit = base_v / (np.linalg.norm(base_v) + 1e-8)
            to_v_unit_2 = to_v / (to_v_norm + 1e-8)
            alignment = np.dot(v_unit, to_v_unit_2)
            reward += max(0.0, alignment) * 1.6

        for f in self.fire_positions:
            df = np.linalg.norm(robot_xy - f)
            if df < 0.9:
                reward -= (0.9 - df) * 1.8

        half = self._room_half
        if half - abs(robot_xy[0]) < self._wall_penalty_distance:
            reward -= 0.6
        if half - abs(robot_xy[1]) < self._wall_penalty_distance:
            reward -= 0.6

        roll, pitch, _ = p.getEulerFromQuaternion(orn)
        if abs(roll) > self._upright_threshold or abs(pitch) > self._upright_threshold or robot_z < 0.05:
            reward -= 2.5
            p.resetBasePositionAndOrientation(self.robot, [robot_xy[0],robot_xy[1],0.12],[0,0,0,1])
            p.resetBaseVelocity(self.robot,[0,0,0],[0,0,0])

        rescued = False
        if dist < 0.55:
            rescued = True

        contacts = []
        for part in self.victim_parts:
            pts = p.getContactPoints(self.robot, part)
            if pts:
                contacts.extend(pts)
        if contacts:
            rescued = True

        if rescued:
            reward += 400.0
            print("\n VICTIM RESCUED — AUTO RESET!")
            self._internal_reset_scene()
            print(f"New victim at {tuple(self.victim_pos)}\n")
            terminated = True
        else:
            terminated = False

        if dist < 0.85:
            if np.linalg.norm(base_v) > 0.02 and to_v_norm > 1e-6:
                if np.dot(base_v, to_v) < -0.02:
                    p.resetBaseVelocity(self.robot, [base_v[0]*0.25, base_v[1]*0.25, 0], [0,0,0])

        # CRITICAL: Only step fire renderer if PyBullet is still connected
        if p.isConnected():
            self.fire_renderer.step()
        
        self.episode_steps += 1
        self.cumulative_reward += reward

        truncated = False
        if self.episode_steps >= self.max_steps:
            truncated = True

        return self._get_obs(), float(reward), terminated, truncated, {}

    # ---------------------------- CLOSE ----------------------------
    def close(self):
        # Clean up particles before closing
        if hasattr(self, 'fire_renderer'):
            self.fire_renderer.cleanup()
        
        if p.isConnected():
            p.disconnect()

# ---------------------------- DEMO ----------------------------
if __name__ == "__main__":
    env = FireRescueEnv(render_mode="human", headless=False)
    obs,_ = env.reset()
    print("Running refined approach demo...\n")

    for i in range(800):
        a = env.action_space.sample()
        obs, r, terminated, truncated, _ = env.step(a)
        if terminated:
            print(f"Victim saved at step {i}, auto-reset continuing...")
        time.sleep(1/240)

    env.close()