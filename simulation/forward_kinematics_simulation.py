import mujoco
import mujoco.viewer
import numpy as np
import math
import time
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional


class MuJoCoForwardKinematics:
    def __init__(self, mjcf_path: Optional[str] = "scene.xml"):
        self.model = mujoco.MjModel.from_xml_path(mjcf_path)
        self.data = mujoco.MjData(self.model)

        self.joint_names = [
            "Rotation", "Pitch", "Elbow", "Wrist_Pitch", "Wrist_Roll", "Jaw"
        ]
        self.joint_ids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
                          for name in self.joint_names]
        #lo, hi = self.model.jnt_range[self.joint_ids]
        #print(f"{name:12s}: [{lo:.2f}, {hi:.2f}] rad")

        self.end_effector_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, 'Moving_Jaw')

        if self.end_effector_id == -1:
            self.end_effector_id = self.model.nbody - 1

        print("Loaded model with joints:", self.joint_names)
        print("End-effector body ID:", self.end_effector_id)
       
            


    def set_joint_angles(self, joint_angles: List[float]):
        for name, angle in zip(self.joint_names, joint_angles):
            j_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            qpos_id = self.model.jnt_qposadr[j_id]
            self.data.qpos[qpos_id] = angle
        mujoco.mj_forward(self.model, self.data)

    def get_end_effector_pose(self) -> Tuple[np.ndarray, np.ndarray]:
        pos = self.data.body(self.end_effector_id).xpos.copy()
        rot = self.data.body(self.end_effector_id).xmat.copy().reshape(3, 3)
        return pos, rot

    def interactive_fk_loop(self):
        print("\n=== Interactive Forward Kinematics ===")
        with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
            while viewer.is_running():
                try:
                    print("\nEnter joint angles (degrees):")
                    angles_deg = []
                    for name in self.joint_names[:-1]:  # all but gripper
                        angle = float(input(f"{name}: "))
                        angle = math.radians(angle)
                        j_id   = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
                        lo, hi = self.model.jnt_range[j_id]
                        if not (lo <= angle <= hi):
                            raise RuntimeError("Skipping input by request")

                        angles_deg.append(angle)
                    grip = float(input("Gripper (0 = close, 1 = open): "))
                    grip_angle = 0.8 if grip else 0.0
                    angles_deg.append(grip_angle)
                    #for name in ["Rotation","Pitch","Elbow","Wrist_Pitch","Wrist_Roll","Jaw"]:
                     #   j_id   = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
                    #
                       # print(f"{name:12s}: [{lo:.2f}, {hi:.2f}] rad")

                    self.set_joint_angles(angles_deg)
                    pos, _ = self.get_end_effector_pose()

                    print("\nEnd-effector position:")
                    print(f"x = {pos[0]:.5f}, y = {pos[1]:.5f}, z = {pos[2]:.5f}")
                except RuntimeError as ee:
                    print("Angle out of limits")
                    continue
                except Exception as e:
                    print("Invalid input:", e)
                    continue

                for _ in range(100):  # simulate for 1 second at ~100Hz
                    mujoco.mj_step(self.model, self.data)
                    viewer.sync()
                    time.sleep(0.01)

    def workspace_analysis(self, n_samples: int = 1000):
        print("\nAnalyzing workspace...")
        joint_limits = [self.model.jnt_range[i] for i in self.joint_ids]
        points = []
        for _ in range(n_samples):
            config = [np.random.uniform(low, high) for (low, high) in joint_limits]
            self.set_joint_angles(config)
            pos, _ = self.get_end_effector_pose()
            points.append(pos)

        points = np.array(points)
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        sc = ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=points[:, 2], cmap='viridis', s=2)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        plt.colorbar(sc, label="Z height")
        plt.title("SO-ARM100 Workspace")
        plt.tight_layout()
        plt.show()


def main():
    print("SO-ARM100 MuJoCo FK Interactive & Workspace Visualizer")
    sim = MuJoCoForwardKinematics()

    while True:
        print("\nOptions:")
        print("1. Interactive Forward Kinematics")
        print("2. Workspace Analysis")
        print("3. Exit")
        choice = input("Choose: ")

        if choice == "1":
            sim.interactive_fk_loop()
        elif choice == "2":
            sim.workspace_analysis(n_samples=2000)
        elif choice == "3":
            break
        else:
            print("Invalid choice.")


if __name__ == "__main__":
    main()
