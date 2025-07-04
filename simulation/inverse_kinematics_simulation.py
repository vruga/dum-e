import mujoco
import mujoco.viewer
import numpy as np
import math
import time
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional
from scipy.optimize import minimize, differential_evolution
import threading
import queue


class MuJoCoInverseKinematics:
    def __init__(self, mjcf_path: Optional[str] = "scene.xml"):
        self.model = mujoco.MjModel.from_xml_path(mjcf_path)
        self.data = mujoco.MjData(self.model)

        self.joint_names = [
            "Rotation", "Pitch", "Elbow", "Wrist_Pitch", "Wrist_Roll", "Jaw"
        ]
        self.joint_ids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
                          for name in self.joint_names]
        
        # Get joint limits
        self.joint_limits = [self.model.jnt_range[i] for i in self.joint_ids]
        
        self.end_effector_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, 'Moving_Jaw')

        if self.end_effector_id == -1:
            self.end_effector_id = self.model.nbody - 1

        print("Loaded model with joints:", self.joint_names)
        print("End-effector body ID:", self.end_effector_id)
        
        # Pre-compute workspace points for validation
        self.workspace_points = self._generate_workspace_points(n_samples=5000)
        print(f"Generated {len(self.workspace_points)} workspace points")
        
        # Current target and solution
        self.current_target = None
        self.current_solution = None
        self.target_reached = False
        
        # For real-time visualization
        self.update_queue = queue.Queue()

    def _generate_workspace_points(self, n_samples: int = 5000) -> np.ndarray:
        """Pre-generate workspace points for validation"""
        points = []
        
        # Use Latin Hypercube Sampling for better coverage
        from scipy.stats import qmc
        
        # Generate samples using quasi-random sequence
        sampler = qmc.LatinHypercube(d=len(self.joint_limits))
        samples = sampler.random(n=n_samples)
        
        # Scale to joint limits
        for i, sample in enumerate(samples):
            config = []
            for j, (low, high) in enumerate(self.joint_limits):
                angle = low + sample[j] * (high - low)
                config.append(angle)
            
            self.set_joint_angles(config)
            pos, _ = self.get_end_effector_pose()
            if pos[2] < 0.01:
                continue
            if self.has_self_collision():
                continue

            if np.linalg.norm(pos[:2]) < 0.05:  # avoid base collisions
                continue
            points.append(pos)
            
            # Print progress
            if i % 1000 == 0:
                print(f"Generated {i}/{n_samples} workspace points...")
        
        return np.array(points)

    def find_nearest_reachable_point(self, target_pos: np.ndarray) -> Tuple[np.ndarray, int]:
        """Find the nearest reachable point to the target"""
        distances = np.linalg.norm(self.workspace_points - target_pos, axis=1)
        nearest_idx = np.argmin(distances)
        return self.workspace_points[nearest_idx], nearest_idx
    
    def has_self_collision(self) -> bool:
        mujoco.mj_forward(self.model, self.data)  # Update contacts
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            geom1 = contact.geom1
            geom2 = contact.geom2

        # Skip ground contacts (geom 0 is often the floor)
            if 0 in (geom1, geom2):
                continue

        # If both geoms are part of the robot and not the ground, it's a self-collision
            if self.model.geom_bodyid[geom1] != 0 and self.model.geom_bodyid[geom2] != 0:
                return True
        return False


    def set_joint_angles(self, joint_angles: List[float]):
        """Set joint angles and update simulation"""
        for name, angle in zip(self.joint_names, joint_angles):
            j_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            qpos_id = self.model.jnt_qposadr[j_id]
            self.data.qpos[qpos_id] = angle
        mujoco.mj_forward(self.model, self.data)

    def get_end_effector_pose(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get end-effector position and orientation"""
        pos = self.data.body(self.end_effector_id).xpos.copy()
        rot = self.data.body(self.end_effector_id).xmat.copy().reshape(3, 3)
        return pos, rot

    def is_point_in_workspace(self, target_pos: np.ndarray, tolerance: float = 0.1) -> bool:
        """Check if a target position is reachable within workspace"""
        distances = np.linalg.norm(self.workspace_points - target_pos, axis=1)
        return np.min(distances) < tolerance

    def objective_function(self, joint_angles: np.ndarray, target_pos: np.ndarray) -> float:
        """Objective function for IK optimization"""
        # Check joint limits
        for i, (angle, (low, high)) in enumerate(zip(joint_angles, self.joint_limits)):
            if not (low <= angle <= high):
                return 1e6  # Large penalty for out-of-bounds
        # Check for self-collision
        if self.has_self_collision():
            return 1e6  # Big penalty

        # Set joint angles and compute forward kinematics
        self.set_joint_angles(joint_angles.tolist())
        current_pos, _ = self.get_end_effector_pose()
        
        # Position error (primary objective)
        pos_error = np.linalg.norm(current_pos - target_pos)
        
        # Add small penalty for joint limits to encourage centered solutions
        joint_penalty = 0.001 * sum([((angle - (low + high)/2) / (high - low))**2 
                                   for angle, (low, high) in zip(joint_angles[:-1], self.joint_limits[:-1])])
        
        # Penalty for large joint angles (avoid singularities)
        singularity_penalty = 0.001 * sum([abs(angle)**2 for angle in joint_angles[:-1]])
        
        return pos_error + joint_penalty + singularity_penalty

    def solve_ik(self, target_pos: np.ndarray, initial_guess: Optional[np.ndarray] = None) -> Tuple[np.ndarray, bool]:
        """Solve inverse kinematics for target position"""
        if not self.is_point_in_workspace(target_pos):
            print(f"Warning: Target position {target_pos} may not be reachable")
        
        # Initial guess - use current configuration or random
        if initial_guess is None:
            initial_guess = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.4])  # Better starting position
        
        # Bounds for optimization (excluding gripper for now)
        bounds = [(low, high) for (low, high) in self.joint_limits[:-1]]
        bounds.append((0.0, 0.8))  # Gripper constraint
        
        # Try multiple optimization approaches
        best_solution = None
        best_error = float('inf')
        
        # Method 1: Multiple random starts with L-BFGS-B
        for attempt in range(10):  # More attempts
            if attempt == 0:
                start_config = initial_guess[:-1]  # Use initial guess first
            else:
                # Random start but bias towards center of joint ranges
                start_config = []
                for low, high in self.joint_limits[:-1]:
                    center = (low + high) / 2
                    range_size = high - low
                    # Sample from center ± 50% of range
                    rand_val = np.random.uniform(center - 0.5*range_size, center + 0.5*range_size)
                    start_config.append(np.clip(rand_val, low, high))
                start_config = np.array(start_config)
            
            try:
                result = minimize(
                    lambda x: self.objective_function(np.append(x, 0.4), target_pos),
                    start_config,
                    method='L-BFGS-B',
                    bounds=bounds[:-1],
                    options={'maxiter': 1000, 'ftol': 1e-9}
                )
                
                if result.fun < best_error:
                    best_error = result.fun
                    best_solution = np.append(result.x, 0.4)
                    
                    # Early exit if solution is good enough
                    if best_error < 0.001:
                        break
                        
            except Exception as e:
                continue
        
        # Method 2: Differential Evolution (global optimization)
        if best_error > 0.01:
            try:
                result = differential_evolution(
                    lambda x: self.objective_function(np.append(x, 0.4), target_pos),
                    bounds[:-1],
                    maxiter=500,
                    seed=None,  # Different seed each time
                    atol=1e-6,
                    popsize=15
                )
                
                if result.fun < best_error:
                    best_error = result.fun
                    best_solution = np.append(result.x, 0.4)
            except Exception as e:
                pass
        
        # Method 3: Jacobian-based approach (if we have a good starting point)
        if best_solution is not None and best_error < 0.1:
            try:
                refined_solution = self.refine_solution_jacobian(best_solution, target_pos)
                if refined_solution is not None:
                    test_error = self.objective_function(refined_solution, target_pos)
                    if test_error < best_error:
                        best_error = test_error
                        best_solution = refined_solution
            except:
                pass
        
        success = best_solution is not None and best_error < 0.02
        
        if success:
            print(f"✓ IK converged with error: {best_error:.6f}")
        else:
            print(f"✗ IK failed, best error: {best_error:.6f}")
            
        return best_solution if best_solution is not None else initial_guess, success

    def refine_solution_jacobian(self, solution: np.ndarray, target_pos: np.ndarray, max_iterations: int = 10) -> Optional[np.ndarray]:
        """Refine IK solution using Jacobian-based method"""
        current_config = solution.copy()
        
        for iteration in range(max_iterations):
            # Set current configuration
            self.set_joint_angles(current_config.tolist())
            current_pos, _ = self.get_end_effector_pose()
            
            # Check if we're close enough
            error = np.linalg.norm(current_pos - target_pos)
            if error < 1e-6:
                return current_config
            
            # Compute numerical Jacobian
            jacobian = self.compute_jacobian(current_config)
            
            # Compute position error
            pos_error = target_pos - current_pos
            
            # Solve for joint velocities using damped least squares
            damping = 0.01
            J_damped = jacobian.T @ jacobian + damping * np.eye(len(current_config) - 1)  # Exclude gripper
            
            try:
                delta_q = np.linalg.solve(J_damped, jacobian.T @ pos_error)
                
                # Update joint angles (excluding gripper)
                step_size = 0.1  # Conservative step size
                new_config = current_config.copy()
                new_config[:-1] += step_size * delta_q
                
                # Clip to joint limits
                for i, (low, high) in enumerate(self.joint_limits[:-1]):
                    new_config[i] = np.clip(new_config[i], low, high)
                
                current_config = new_config
                
            except np.linalg.LinAlgError:
                break
        
        return current_config

    def compute_jacobian(self, config: np.ndarray, eps: float = 1e-6) -> np.ndarray:
        """Compute numerical Jacobian matrix"""
        self.set_joint_angles(config.tolist())
        pos0, _ = self.get_end_effector_pose()
        
        jacobian = np.zeros((3, len(config) - 1))  # 3D position, exclude gripper
        
        for i in range(len(config) - 1):  # Exclude gripper
            # Positive perturbation
            config_plus = config.copy()
            config_plus[i] += eps
            
            # Check bounds
            low, high = self.joint_limits[i]
            if config_plus[i] > high:
                config_plus[i] = config[i] - eps
                
            self.set_joint_angles(config_plus.tolist())
            pos_plus, _ = self.get_end_effector_pose()
            
            # Compute finite difference
            jacobian[:, i] = (pos_plus - pos0) / eps
        
        # Restore original configuration
        self.set_joint_angles(config.tolist())
        
        return jacobian
        """Generate smooth trajectory between two configurations"""
        trajectory = []
        for i in range(steps):
            t = i / (steps - 1)
            config = start_config + t * (end_config - start_config)
            trajectory.append(config)
        return np.array(trajectory)

    def interactive_ik_loop(self):
        """Interactive IK with real-time MuJoCo visualization"""
        print("\n=== Interactive Inverse Kinematics ===")
        print("Click points in 3D space or enter coordinates manually")
        
        with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
            # Initialize with home position
            home_config = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.4])
            self.set_joint_angles(home_config.tolist())
            current_config = home_config.copy()
            
            while viewer.is_running():
                try:
                    print("\nOptions:")
                    print("1. Enter target coordinates")
                    print("2. Random workspace point")
                    print("3. Demo trajectory")
                    print("4. Go to home position")
                    print("q. Quit")
                    
                    choice = input("Choose: ").strip()
                    
                    if choice == 'q':
                        break
                    elif choice == '1':
                        # Manual coordinate input
                        x = float(input("Target X: "))
                        y = float(input("Target Y: "))
                        z = float(input("Target Z: "))
                        target_pos = np.array([x, y, z])
                    elif choice == '2':
                        # Random workspace point
                        idx = np.random.randint(0, len(self.workspace_points))
                        target_pos = self.workspace_points[idx]
                        print(f"Random target: {target_pos}")
                    elif choice == '3':
                        # Demo trajectory
                        self.demo_trajectory(viewer, current_config)
                        continue
                    elif choice == '4':
                        # Go home
                        target_pos = None
                        target_config = home_config
                    else:
                        print("Invalid choice")
                        continue
                    
                    if target_pos is not None:
                        # Check if point is reachable
                        if not self.is_point_in_workspace(target_pos):
                            nearest_point, _ = self.find_nearest_reachable_point(target_pos)
                            print(f"Target unreachable! Nearest reachable point: {nearest_point}")
                            use_nearest = input("Use nearest reachable point? (y/n): ").strip().lower()
                            if use_nearest == 'y':
                                target_pos = nearest_point
                            else:
                                continue
                        
                        print(f"Solving IK for target: {target_pos}")
                        target_config, success = self.solve_ik(target_pos, current_config)
                        
                        if success:
                            print("✓ IK solution found!")
                            self.set_joint_angles(target_config.tolist())
                            actual_pos, _ = self.get_end_effector_pose()
                            error = np.linalg.norm(actual_pos - target_pos)
                            print(f"Position error: {error:.6f}")
                        else:
                            print("✗ IK solution failed, using best attempt")
                    else:
                        print("Going to home position...")
                        target_config = home_config
                    
                    print("Executing trajectory...")
                    trajectory = self.interpolate_trajectory(current_config, target_config, steps=100)

                    for config in trajectory:
                        self.set_joint_angles(config.tolist())
                        
                        # Simulation step
                        mujoco.mj_step(self.model, self.data)
                        viewer.sync()
                        time.sleep(0.02)  # 50 Hz
                        
                        if not viewer.is_running():
                            break
                    
                    current_config = target_config
                    
                    if target_pos is not None:
                        final_pos, _ = self.get_end_effector_pose()
                        print(f"Final position: {final_pos}")
                        print(f"Target position: {target_pos}")
                        print(f"Final error: {np.linalg.norm(final_pos - target_pos):.6f}")
                
                except KeyboardInterrupt:
                    break
                except Exception as e:
                    print(f"Error: {e}")
                    continue

    def demo_trajectory(self, viewer, start_config: np.ndarray):
        """Demo trajectory through multiple points"""
        print("Running demo trajectory...")
        
        # Get actual workspace bounds for realistic demo points
        x_min, x_max = self.workspace_points[:, 0].min(), self.workspace_points[:, 0].max()
        y_min, y_max = self.workspace_points[:, 1].min(), self.workspace_points[:, 1].max()
        z_min, z_max = self.workspace_points[:, 2].min(), self.workspace_points[:, 2].max()
        
        print(f"Workspace bounds: X[{x_min:.3f}, {x_max:.3f}], Y[{y_min:.3f}, {y_max:.3f}], Z[{z_min:.3f}, {z_max:.3f}]")
        
        # Define demo points within actual workspace
        demo_points = [
            np.array([0.8 * x_max, 0.0, 0.5 * (z_min + z_max)]),
            np.array([0.5 * x_max, 0.8 * y_max, 0.7 * z_max]),
            np.array([0.0, 0.9 * y_max, 0.3 * z_max]),
            np.array([0.5 * x_min, 0.8 * y_max, 0.7 * z_max]),
            np.array([0.8 * x_min, 0.0, 0.5 * (z_min + z_max)]),
            np.array([0.0, 0.0, 0.9 * z_max])
        ]
        
        # Validate demo points and replace unreachable ones
        valid_demo_points = []
        for i, point in enumerate(demo_points):
            if self.is_point_in_workspace(point, tolerance=0.05):
                valid_demo_points.append(point)
                print(f"✓ Demo point {i+1}: {point}")
            else:
                # Replace with a random valid point
                idx = np.random.randint(0, len(self.workspace_points))
                valid_point = self.workspace_points[idx]
                valid_demo_points.append(valid_point)
                print(f"✗ Demo point {i+1} replaced with: {valid_point}")
        
        demo_current_config = start_config.copy()
        
        for i, target_pos in enumerate(valid_demo_points):
            print(f"\nMoving to point {i+1}/{len(valid_demo_points)}: {target_pos}")
            
            # Solve IK
            target_config, success = self.solve_ik(target_pos, demo_current_config)
            
            # Execute trajectory
            trajectory = self.interpolate_trajectory(demo_current_config, target_config, steps=80)
            
            
            for config in trajectory:
                self.set_joint_angles(config.tolist())
                mujoco.mj_step(self.model, self.data)
                viewer.sync()
                time.sleep(0.025)  # 40 Hz
                
                if not viewer.is_running():
                    return
            
            demo_current_config = target_config
            #trajectory = self.interpolate_trajectory(current_config, target_config, steps=100)
            # Verify final position
            final_pos, _ = self.get_end_effector_pose()
            error = np.linalg.norm(final_pos - target_pos)
            print(f"Final error: {error:.6f}")
            
            time.sleep(0.5)  # Pause at each point

    def interpolate_trajectory(self, start_config: np.ndarray, end_config: np.ndarray, steps: int = 50) -> np.ndarray:
        trajectory = []
        for i in range(steps):
            t = i / (steps - 1)
            config = start_config + t * (end_config - start_config)
            trajectory.append(config)
        return np.array(trajectory)

    def workspace_analysis(self, n_samples: int = 2000):
        """Analyze and visualize workspace"""
        print("\nAnalyzing workspace...")
        
        fig = plt.figure(figsize=(12, 5))
        
        # 3D workspace plot
        ax1 = fig.add_subplot(121, projection='3d')
        sc = ax1.scatter(self.workspace_points[:, 0], self.workspace_points[:, 1], 
                        self.workspace_points[:, 2], c=self.workspace_points[:, 2], 
                        cmap='viridis', s=1, alpha=0.6)
        ax1.set_xlabel("X (m)")
        ax1.set_ylabel("Y (m)")
        ax1.set_zlabel("Z (m)")
        ax1.set_title("SO-ARM100 Workspace")
        plt.colorbar(sc, ax=ax1, label="Z height")
        
        # Workspace statistics
        ax2 = fig.add_subplot(122)
        ranges = {
            'X': (self.workspace_points[:, 0].min(), self.workspace_points[:, 0].max()),
            'Y': (self.workspace_points[:, 1].min(), self.workspace_points[:, 1].max()),
            'Z': (self.workspace_points[:, 2].min(), self.workspace_points[:, 2].max())
        }
        
        axes = list(ranges.keys())
        mins = [ranges[ax][0] for ax in axes]
        maxs = [ranges[ax][1] for ax in axes]
        
        x_pos = np.arange(len(axes))
        width = 0.35
        
        ax2.bar(x_pos - width/2, mins, width, label='Min', alpha=0.8)
        ax2.bar(x_pos + width/2, maxs, width, label='Max', alpha=0.8)
        ax2.set_xlabel('Axis')
        ax2.set_ylabel('Position (m)')
        ax2.set_title('Workspace Ranges')
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(axes)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # Print statistics
        print("\nWorkspace Statistics:")
        for axis in axes:
            print(f"{axis}: [{ranges[axis][0]:.3f}, {ranges[axis][1]:.3f}] m")
        
        volume = np.prod([ranges[ax][1] - ranges[ax][0] for ax in axes])
        print(f"Approximate workspace volume: {volume:.6f} m³")


def main():
    print("SO-ARM100 MuJoCo Inverse Kinematics Simulator")
    print("=" * 50)
    
    try:
        sim = MuJoCoInverseKinematics()
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Make sure 'scene.xml' is in the current directory")
        return
    
    while True:
        print("\nOptions:")
        print("1. Interactive Inverse Kinematics")
        print("2. Workspace Analysis")
        print("3. Exit")
        
        choice = input("Choose: ").strip()
        
        if choice == "1":
            sim.interactive_ik_loop()
        elif choice == "2":
            sim.workspace_analysis()
        elif choice == "3":
            print("Goodbye!")
            break
        else:
            print("Invalid choice. Please try again.")


if __name__ == "__main__":
    main()