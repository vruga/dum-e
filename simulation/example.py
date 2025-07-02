"""
MuJoCo-based Forward Kinematics Simulation for SO-ARM100
"""

import numpy as np
import mujoco
import mujoco.viewer
import time
import threading
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional
import xml.etree.ElementTree as ET


class MuJoCoForwardKinematics:
    """
    Forward kinematics implementation using MuJoCo physics engine
    """
    
    def __init__(self, mjcf_path: Optional[str] = None):
        """
        Initialize MuJoCo simulation
        
        Args:
            mjcf_path: Path to MJCF model file (if None, creates default SO-ARM100)
        """
        if mjcf_path is None:
            # Create default SO-ARM100 model
            self.mjcf_path = self._create_so_arm100_mjcf()
        else:
            self.mjcf_path = mjcf_path
            
        # Load model and create data
        self.model = mujoco.MjModel.from_xml_path(self.mjcf_path)
        self.data = mujoco.MjData(self.model)
        
        # Get joint and actuator information
        self.joint_names = []
        self.joint_ids = []
        self.actuator_ids = []
        
        for i in range(self.model.njnt):
            joint_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, i)
            if joint_name and not joint_name.startswith('free'):
                self.joint_names.append(joint_name)
                self.joint_ids.append(i)
        
        for i in range(self.model.nu):
            self.actuator_ids.append(i)
        
        # Get end-effector body ID
        self.end_effector_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, 'end_effector'
        )
        if self.end_effector_id == -1:
            # Fallback to last body if end_effector not found
            self.end_effector_id = self.model.nbody - 1
        
        print(f"Loaded model with {len(self.joint_names)} joints")
        print(f"Joint names: {self.joint_names}")
        print(f"End-effector body ID: {self.end_effector_id}")
        
        # Viewer for visualization
        self.viewer = None
        self.viewer_thread = None
        self.running = False
    
    def _create_so_arm100_mjcf(self) -> str:
        """
        Create a default MJCF model for SO-ARM100
        This is a template - replace with actual SO-ARM100 specifications
        """
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
        
        # Save to temporary file
        mjcf_file = "so_arm100_temp.xml"
        with open(mjcf_file, 'w') as f:
            f.write(mjcf_content)
        
        return mjcf_file
    
    def start_viewer(self):
        """Start MuJoCo viewer in separate thread"""
        if self.viewer is not None:
            return
        
        self.running = True
        self.viewer_thread = threading.Thread(target=self._viewer_loop, daemon=True)
        self.viewer_thread.start()
        time.sleep(1)  # Give viewer time to start
    
    def _viewer_loop(self):
        """Main viewer loop"""
        with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
            self.viewer = viewer
            
            while self.running and viewer.is_running():
                # Step physics
                mujoco.mj_step(self.model, self.data)
                
                # Sync viewer
                viewer.sync()
                
                # Control update rate
                time.sleep(0.01)  # 100 Hz
    
    def stop_viewer(self):
        """Stop the viewer"""
        self.running = False
        if self.viewer_thread:
            self.viewer_thread.join(timeout=2.0)
        self.viewer = None
    
    def set_joint_angles(self, joint_angles: List[float]):
        """
        Set joint angles in the simulation
        
        Args:
            joint_angles: List of joint angles in radians
        """
        if len(joint_angles) != len(self.joint_ids):
            raise ValueError(f"Expected {len(self.joint_ids)} angles, got {len(joint_angles)}")
        
        # Set joint positions
        for i, angle in enumerate(joint_angles):
            if i < len(self.joint_ids):
                self.data.qpos[self.joint_ids[i]] = angle
        
        # Set control targets (for position control)
        for i, angle in enumerate(joint_angles):
            if i < len(self.actuator_ids):
                self.data.ctrl[self.actuator_ids[i]] = angle
        
        # Forward kinematics computation
        mujoco.mj_forward(self.model, self.data)
    
    def get_end_effector_pose(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get current end-effector pose
        
        Returns:
            Tuple of (position, rotation_matrix)
        """
        # Get end-effector position
        ee_pos = self.data.body(self.end_effector_id).xpos.copy()
        
        # Get end-effector rotation matrix
        ee_rot_mat = self.data.body(self.end_effector_id).xmat.copy().reshape(3, 3)
        
        return ee_pos, ee_rot_mat
    
    def get_joint_positions(self) -> List[np.ndarray]:
        """
        Get positions of all joints for visualization
        
        Returns:
            List of 3D positions for each joint
        """
        positions = []
        
        # Add base position
        positions.append(np.array([0, 0, 0]))
        
        # Get positions of all bodies (links)
        for i in range(1, self.model.nbody):  # Skip world body
            body_pos = self.data.body(i).xpos.copy()
            positions.append(body_pos)
        
        return positions
    
    def compute_jacobian(self) -> np.ndarray:
        """
        Compute Jacobian matrix using MuJoCo's built-in function
        
        Returns:
            6xN Jacobian matrix
        """
        # Initialize Jacobian matrices
        jacp = np.zeros((3, self.model.nv))  # Position Jacobian
        jacr = np.zeros((3, self.model.nv))  # Rotation Jacobian
        
        # Compute Jacobian
        mujoco.mj_jac(self.model, self.data, jacp, jacr, 
                     self.data.body(self.end_effector_id).xpos, self.end_effector_id)
        
        # Combine position and rotation Jacobians
        jacobian = np.vstack([jacp, jacr])
        
        # Return only columns corresponding to active joints
        return jacobian[:, :len(self.joint_ids)]
    
    def test_forward_kinematics(self, test_configurations: List[List[float]]):
        """
        Test forward kinematics with multiple configurations
        
        Args:
            test_configurations: List of joint angle configurations to test
        """
        print("\n=== Testing Forward Kinematics in MuJoCo ===")
        
        results = []
        
        for i, config in enumerate(test_configurations):
            print(f"\nConfiguration {i+1}: {[f'{a:.3f}' for a in config]}")
            
            # Set joint angles
            self.set_joint_angles(config)
            
            # Get end-effector pose
            ee_pos, ee_rot = self.get_end_effector_pose()
            
            # Compute Jacobian
            jacobian = self.compute_jacobian()
            
            print(f"End-effector position: [{ee_pos[0]:.4f}, {ee_pos[1]:.4f}, {ee_pos[2]:.4f}]")
            print(f"Jacobian condition number: {np.linalg.cond(jacobian):.2f}")
            
            results.append({
                'config': config,
                'position': ee_pos,
                'rotation': ee_rot,
                'jacobian': jacobian
            })
            
            # Wait a bit for visualization
            if self.viewer:
                time.sleep(1.0)
        
        return results
    
    def animate_trajectory(self, waypoints: List[List[float]], duration: float = 5.0):
        """
        Animate a trajectory through multiple waypoints
        
        Args:
            waypoints: List of joint configurations
            duration: Total animation duration in seconds
        """
        print(f"\n=== Animating Trajectory ({len(waypoints)} waypoints) ===")
        
        if not self.viewer:
            print("Starting viewer for animation...")
            self.start_viewer()
            time.sleep(2)  # Wait for viewer to initialize
        
        n_steps = int(duration * 100)  # 100 Hz animation
        
        for step in range(n_steps):
            t = step / n_steps
            
            # Find current segment
            segment_length = 1.0 / (len(waypoints) - 1)
            segment_idx = min(int(t / segment_length), len(waypoints) - 2)
            local_t = (t - segment_idx * segment_length) / segment_length
            
            # Interpolate between waypoints
            current_config = []
            for j in range(len(waypoints[0])):
                start_angle = waypoints[segment_idx][j]
                end_angle = waypoints[segment_idx + 1][j]
                current_angle = start_angle + local_t * (end_angle - start_angle)
                current_config.append(current_angle)
            
            # Set joint angles
            self.set_joint_angles(current_config)
            
            # Get current end-effector position for trajectory visualization
            ee_pos, _ = self.get_end_effector_pose()
            
            time.sleep(0.01)  # 100 Hz
        
        print("Animation complete!")
    
    def workspace_analysis(self, n_samples: int = 1000) -> np.ndarray:
        """
        Analyze reachable workspace by sampling joint configurations
        
        Args:
            n_samples: Number of random configurations to sample
            
        Returns:
            Array of reachable end-effector positions
        """
        print(f"Analyzing workspace with {n_samples} samples...")
        
        # Define joint limits from model
        joint_limits = []
        for i in self.joint_ids:
            joint_range = self.model.jnt_range[i]
            joint_limits.append((joint_range[0], joint_range[1]))
        
        positions = []
        
        for _ in range(n_samples):
            # Sample random joint configuration
            config = []
            for (min_angle, max_angle) in joint_limits:
                angle = np.random.uniform(min_angle, max_angle)
                config.append(angle)
            
            # Compute forward kinematics
            try:
                self.set_joint_angles(config)
                ee_pos, _ = self.get_end_effector_pose()
                positions.append(ee_pos.copy())
            except:
                continue
        
        return np.array(positions)
    
    def visualize_workspace(self, n_samples: int = 1000):
        """Visualize robot workspace"""
        workspace_points = self.workspace_analysis(n_samples)
        
        # Create 3D plot
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot workspace points
        scatter = ax.scatter(workspace_points[:, 0], workspace_points[:, 1], workspace_points[:, 2], 
                           c=workspace_points[:, 2], cmap='viridis', alpha=0.6, s=2)
        
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.set_title('SO-ARM100 Workspace Analysis (MuJoCo)')
        
        # Add colorbar
        plt.colorbar(scatter, ax=ax, label='Height (m)')
        
        # Make axes equal
        max_range = np.array([workspace_points[:, 0].ptp(),
                             workspace_points[:, 1].ptp(),
                             workspace_points[:, 2].ptp()]).max() / 2.0
        
        mid_x = workspace_points[:, 0].mean()
        mid_y = workspace_points[:, 1].mean()
        mid_z = workspace_points[:, 2].mean()
        
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)
        
        plt.tight_layout()
        plt.show()
        
        print(f"Workspace analysis complete:")
        print(f"  - {workspace_points.shape[0]} reachable points")
        print(f"  - X range: [{workspace_points[:, 0].min():.3f}, {workspace_points[:, 0].max():.3f}] m")
        print(f"  - Y range: [{workspace_points[:, 1].min():.3f}, {workspace_points[:, 1].max():.3f}] m")
        print(f"  - Z range: [{workspace_points[:, 2].min():.3f}, {workspace_points[:, 2].max():.3f}] m")
    
    def __del__(self):
        """Cleanup"""
        self.stop_viewer()


def main():
    """Main demonstration function"""
    print("SO-ARM100 MuJoCo Forward Kinematics Simulation")
    print("=" * 50)
    
    # Initialize simulation
    sim = MuJoCoForwardKinematics()
    
    # Start viewer
    print("Starting MuJoCo viewer...")
    sim.start_viewer()
    
    # Test configurations
    test_configs = [
        [0, 0, 0, 0, 0, 0],                    # Home position
        [0.5, 0.3, -0.5, 0.2, 0.1, 1],       # Random pose 1
        [-0.5, -0.3, 0.5, -0.2, -0.1, -1],    # Random pose 2
        [1.0, 0.8, -1.2, 0.5, 0.3, 0],       # Extended pose
        [0, 1.0, -1.5, 0, 0.5, 1],           # High reach
    ]
    
    # Test forward kinematics
    results = sim.test_forward_kinematics(test_configs)
    
    # Animate trajectory
    trajectory_waypoints = [
        [0, 0, 0, 0, 0, 0],
        [0.5, 0.5, -0.5, 0, 0, 0.1],
        [1.0, 0.8, -1.2, 0.5, 0.3, -0.2],
        [-0.5, 0.5, -0.8, -0.5, -0.3, 0.5],
        [0, 0, 0, 0, 0, 0],
    ]
    
    input("Press Enter to start trajectory animation...")
    sim.animate_trajectory(trajectory_waypoints, duration=8.0)
    
    # Workspace analysis
    input("Press Enter to analyze workspace...")
    sim.visualize_workspace(n_samples=2000)
    
    input("Press Enter to exit...")
    sim.stop_viewer()


if __name__ == "__main__":
    main()