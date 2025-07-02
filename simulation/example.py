import numpy as np
import mujoco
import mujoco.viewer
import time
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional
import xml.etree.ElementTree as ET


class MuJoCoForwardKinematics:
    def __init__(self, mjcf_path: Optional[str] = None):
        if mjcf_path is None:
            self.mjcf_path = self._create_so_arm100_mjcf()
        else:
            self.mjcf_path = mjcf_path

        self.model = mujoco.MjModel.from_xml_path(self.mjcf_path)
        self.data = mujoco.MjData(self.model)

        self.joint_names = []
        self.joint_ids = []
        self.actuator_ids = list(range(self.model.nu))

        for i in range(self.model.njnt):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, i)
            if name and not name.startswith('free'):
                self.joint_names.append(name)
                self.joint_ids.append(i)

        self.end_effector_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'Moving_Jaw')
        if self.end_effector_id == -1:
            self.end_effector_id = self.model.nbody - 1

        print(f"Loaded model with {len(self.joint_names)} joints")
        print(f"Joint names: {self.joint_names}")
        print(f"End-effector body ID: {self.end_effector_id}")

    def _create_so_arm100_mjcf(self) -> str:
        mjcf_content = '''<?xml version="1.0" ?>
<mujoco model="so_arm100 scene">
  <include file="so_arm100.xml"/>
  <statistic center="0.1 -0.01 0.05" extent="0.5"/>
  <visual>
    <headlight diffuse="0.6 0.6 0.6" ambient="0.3 0.3 0.3" specular="0 0 0"/>
    <rgba haze="0.15 0.25 0.35 1"/>
    <global azimuth="45" elevation="-20"/>
  </visual>
  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.3 0.5 0.7" rgb2="0 0 0" width="512" height="3072"/>
    <texture type="2d" name="groundplane" builtin="checker" mark="edge" rgb1="0.2 0.3 0.4" rgb2="0.1 0.2 0.3"
      markrgb="0.8 0.8 0.8" width="300" height="300"/>
    <material name="groundplane" texture="groundplane" texuniform="true" texrepeat="5 5" reflectance="0.2"/>
  </asset>
  <worldbody>
    <light pos="0 0 1.5" dir="0 0 -1" directional="true"/>
    <geom name="floor" size="0 0 0.05" type="plane" material="groundplane"/>
  </worldbody>
</mujoco>
'''
        with open("so_arm100_temp.xml", 'w') as f:
            f.write(mjcf_content)
        return "so_arm100_temp.xml"

    def set_joint_angles(self, joint_angles: List[float]):
        for i, angle in enumerate(joint_angles):
            if i < len(self.joint_ids):
                self.data.qpos[i] = angle
        mujoco.mj_forward(self.model, self.data)

    def get_end_effector_pose(self) -> Tuple[np.ndarray, np.ndarray]:
        pos = self.data.body(self.end_effector_id).xpos.copy()
        rot = self.data.body(self.end_effector_id).xmat.copy().reshape(3, 3)
        return pos, rot

    def compute_jacobian(self) -> np.ndarray:
        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))
        mujoco.mj_jac(self.model, self.data, jacp, jacr, self.data.body(self.end_effector_id).xpos, self.end_effector_id)
        return np.vstack([jacp, jacr])[:, :len(self.joint_ids)]

    def test_forward_kinematics(self, configs: List[List[float]]):
        print("\n=== Testing Forward Kinematics ===")
        for i, config in enumerate(configs):
            print(f"\nConfig {i+1}: {[f'{a:.3f}' for a in config]}")
            self.set_joint_angles(config)
            ee_pos, _ = self.get_end_effector_pose()
            jac = self.compute_jacobian()
            print(f"EE position: [{ee_pos[0]:.4f}, {ee_pos[1]:.4f}, {ee_pos[2]:.4f}]")
            print(f"Jacobian condition number: {np.linalg.cond(jac):.2f}")

    def animate_trajectory(self, waypoints: List[List[float]], duration: float = 5.0):
        print(f"\n=== Animating Trajectory ({len(waypoints)} waypoints) ===")
        n_steps = int(duration * 100)
        with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
            for step in range(n_steps):
                t = step / n_steps
                idx = min(int(t * (len(waypoints) - 1)), len(waypoints) - 2)
                local_t = (t * (len(waypoints) - 1)) - idx
                q_start = np.array(waypoints[idx])
                q_end = np.array(waypoints[idx + 1])
                q_interp = (1 - local_t) * q_start + local_t * q_end
                self.set_joint_angles(q_interp)
                mujoco.mj_step(self.model, self.data)
                viewer.sync()
                time.sleep(0.01)

    def workspace_analysis(self, n_samples: int = 1000) -> np.ndarray:
        print(f"Analyzing workspace with {n_samples} samples...")
        joint_limits = [self.model.jnt_range[i] for i in self.joint_ids]
        positions = []
        for _ in range(n_samples):
            config = [np.random.uniform(low, high) for (low, high) in joint_limits]
            self.set_joint_angles(config)
            pos, _ = self.get_end_effector_pose()
            positions.append(pos)
        return np.array(positions)

    def visualize_workspace(self, n_samples: int = 1000):
        points = self.workspace_analysis(n_samples)
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')
        sc = ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=points[:, 2], cmap='viridis', s=2)
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        plt.colorbar(sc, ax=ax, label='Height (Z)')
        plt.title("SO-ARM100 Reachable Workspace")
        plt.tight_layout()
        plt.show()


def main():
    print("SO-ARM100 MuJoCo Forward Kinematics Simulation")
    sim = MuJoCoForwardKinematics()

    test_configs = [
        [0, 0, 0, 0, 0, 0],
        [0.5, 0.3, -0.5, 0.2, 0.1, 1],
        [-0.5, -0.3, 0.5, -0.2, -0.1, -1],
        [1.0, 0.8, -1.2, 0.5, 0.3, 0],
        [0, 1.0, -1.5, 0, 0.5, 1],
    ]

    sim.test_forward_kinematics(test_configs)

    input("Press Enter to animate trajectory...")
    sim.animate_trajectory(test_configs, duration=8.0)

    input("Press Enter to analyze workspace...")
    sim.visualize_workspace(n_samples=2000)


if __name__ == "__main__":
    main()
