"""
Forward Kinematics implementation for SO-ARM100
"""

import numpy as np
from typing import List, Tuple, Optional
import pybullet as p


class ForwardKinematics:
    """
    Forward kinematics solver for SO-ARM100 robotic arm
    """
    
    def __init__(self, urdf_path: str):
        """
        Initialize the forward kinematics solver
        
        Args:
            urdf_path: Path to the URDF file of SO-ARM100
        """
        self.urdf_path = urdf_path
        self.robot_id = None
        self.joint_indices = []
        self.link_indices = []
        self.dh_params = None  # Will be set based on robot configuration
        
        self._load_robot_model()
        self._setup_dh_parameters()
    
    def _load_robot_model(self):
        """Load robot model from URDF"""
        # Connect to PyBullet (in DIRECT mode for kinematics only)
        p.connect(p.DIRECT)
        self.robot_id = p.loadURDF(self.urdf_path)
        
        # Get joint information
        num_joints = p.getNumJoints(self.robot_id)
        for i in range(num_joints):
            joint_info = p.getJointInfo(self.robot_id, i)
            if joint_info[2] != p.JOINT_FIXED:  # Only consider movable joints
                self.joint_indices.append(i)
        
        print(f"Loaded robot with {len(self.joint_indices)} active joints")
    
    def _setup_dh_parameters(self):
        """
        Set up Denavit-Hartenberg parameters for SO-ARM100
        DH parameters: [a, alpha, d, theta_offset]
        
        Note: These need to be measured/extracted from the actual robot
        This is a template - replace with actual values
        """
        # Template DH parameters - REPLACE WITH ACTUAL VALUES
        self.dh_params = np.array([
            [0,      np.pi/2,  0.1,    0],      # Base to joint 1
            [0.15,   0,        0,      0],      # Joint 1 to joint 2  
            [0.12,   0,        0,      0],      # Joint 2 to joint 3
            [0,      np.pi/2,  0.1,    0],      # Joint 3 to joint 4
            [0,      0,        0.08,   0],      # Joint 4 to end-effector
        ])
    
    def dh_transform(self, a: float, alpha: float, d: float, theta: float) -> np.ndarray:
        """
        Compute transformation matrix using DH parameters
        
        Args:
            a: Link length
            alpha: Link twist
            d: Link offset  
            theta: Joint angle
            
        Returns:
            4x4 transformation matrix
        """
        ct = np.cos(theta)
        st = np.sin(theta)
        ca = np.cos(alpha)
        sa = np.sin(alpha)
        
        T = np.array([
            [ct,    -st*ca,  st*sa,   a*ct],
            [st,     ct*ca, -ct*sa,   a*st],
            [0,      sa,     ca,      d],
            [0,      0,      0,       1]
        ])
        
        return T
    
    def forward_kinematics(self, joint_angles: List[float]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute forward kinematics for given joint angles
        
        Args:
            joint_angles: List of joint angles in radians
            
        Returns:
            Tuple of (position, rotation_matrix) of end-effector
        """
        if len(joint_angles) != len(self.dh_params):
            raise ValueError(f"Expected {len(self.dh_params)} joint angles, got {len(joint_angles)}")
        
        # Initialize with identity matrix
        T_total = np.eye(4)
        
        # Multiply transformation matrices
        for i, (a, alpha, d, theta_offset) in enumerate(self.dh_params):
            theta = joint_angles[i] + theta_offset
            T_i = self.dh_transform(a, alpha, d, theta)
            T_total = T_total @ T_i
        
        # Extract position and rotation
        position = T_total[:3, 3]
        rotation_matrix = T_total[:3, :3]
        
        return position, rotation_matrix
    
    def get_joint_positions(self, joint_angles: List[float]) -> List[np.ndarray]:
        """
        Get positions of all joints for visualization
        
        Args:
            joint_angles: List of joint angles in radians
            
        Returns:
            List of 3D positions for each joint
        """
        positions = [np.array([0, 0, 0])]  # Base position
        T_cumulative = np.eye(4)
        
        for i, (a, alpha, d, theta_offset) in enumerate(self.dh_params):
            theta = joint_angles[i] + theta_offset
            T_i = self.dh_transform(a, alpha, d, theta)
            T_cumulative = T_cumulative @ T_i
            positions.append(T_cumulative[:3, 3])
        
        return positions
    
    def get_jacobian(self, joint_angles: List[float]) -> np.ndarray:
        """
        Compute geometric Jacobian matrix
        
        Args:
            joint_angles: Current joint angles
            
        Returns:
            6xN Jacobian matrix (linear and angular velocities)
        """
        n_joints = len(joint_angles)
        jacobian = np.zeros((6, n_joints))
        
        # Get all joint positions and orientations
        joint_positions = self.get_joint_positions(joint_angles)
        end_effector_pos = joint_positions[-1]
        
        # Compute Jacobian columns
        T_cumulative = np.eye(4)
        for i in range(n_joints):
            # Z-axis of current frame (rotation axis)
            z_axis = T_cumulative[:3, 2]
            
            # Position from current joint to end-effector
            r = end_effector_pos - joint_positions[i]
            
            # Linear velocity component
            jacobian[:3, i] = np.cross(z_axis, r)
            
            # Angular velocity component  
            jacobian[3:, i] = z_axis
            
            # Update cumulative transformation
            if i < len(self.dh_params):
                a, alpha, d, theta_offset = self.dh_params[i]
                theta = joint_angles[i] + theta_offset
                T_i = self.dh_transform(a, alpha, d, theta)
                T_cumulative = T_cumulative @ T_i
        
        return jacobian
    
    def workspace_analysis(self, n_samples: int = 1000) -> np.ndarray:
        """
        Analyze reachable workspace by sampling joint space
        
        Args:
            n_samples: Number of random configurations to sample
            
        Returns:
            Array of reachable end-effector positions
        """
        # Define joint limits (replace with actual limits)
        joint_limits = [
            (-np.pi, np.pi),      # Joint 1
            (-np.pi/2, np.pi/2),  # Joint 2
            (-np.pi/2, np.pi/2),  # Joint 3
            (-np.pi, np.pi),      # Joint 4
            (-np.pi, np.pi),      # Joint 5
        ]
        
        positions = []
        for _ in range(n_samples):
            # Sample random joint configuration
            joint_angles = []
            for (min_angle, max_angle) in joint_limits:
                angle = np.random.uniform(min_angle, max_angle)
                joint_angles.append(angle)
            
            # Compute forward kinematics
            try:
                pos, _ = self.forward_kinematics(joint_angles)
                positions.append(pos)
            except:
                continue
        
        return np.array(positions)
    
    def __del__(self):
        """Cleanup PyBullet connection"""
        try:
            p.disconnect()
        except:
            pass


# Example usage
if __name__ == "__main__":
    # Initialize forward kinematics
    fk = ForwardKinematics("so100.urdf")
    
    # Test with sample joint angles
    joint_angles = [0, 0.5, -0.3, 0.2, 0]
    
    # Compute forward kinematics
    position, rotation = fk.forward_kinematics(joint_angles)
    
    print(f"End-effector position: {position}")
    print(f"End-effector orientation:\n{rotation}")
    
    # Compute Jacobian
    jacobian = fk.get_jacobian(joint_angles)
    print(f"Jacobian shape: {jacobian.shape}")
    
    # Analyze workspace
    workspace = fk.workspace_analysis(100)
    print(f"Workspace analysis: {workspace.shape[0]} reachable points")