
# SO-100 ARM MuJoCo Kinematics Simulation

A comprehensive robotics simulation project implementing forward and inverse kinematics for the SO-100 robotic arm using MuJoCo physics engine and modern Python tooling.

## 🚀 Features

- **SO-100 ARM Model**: Complete MuJoCo XML model definition
- **Forward Kinematics**: Calculate end-effector pose from joint angles
- **Inverse Kinematics**: Solve for joint angles given desired end-effector pose
- **Real-time Visualization**: Interactive MuJoCo simulation environment
- **Modern Python Stack**: Built with `uv` for fast dependency management

## 📋 Prerequisites

- Python 3.9+
- `uv` package manager ([installation guide](https://docs.astral.sh/uv/getting-started/installation/))
- MuJoCo (will be installed via dependencies)

## 🛠️ Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/so100-arm-mujoco.git
   cd so100-arm-mujoco
   ```

2. **Install dependencies using uv**
   ```bash
   uv sync
   ```

3. **Activate the virtual environment**
   ```bash
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

## 🎯 Quick Start

### Basic Simulation
```bash
# Run the basic arm simulation
uv run python simulation/simulate.py

# Run forward kinematics demo
uv run python simulation/forward_kinematics.py

# Run inverse kinematics demo
uv run python simulation/inverse_kinematics.py
```



## 🔧 Usage Examples

### Forward Kinematics
```python
from src.arm_model import SO100Arm
import numpy as np

# Initialize the arm
arm = SO100Arm()

# Define joint angles (in radians)
joint_angles = np.array([0.5, -0.3, 0.8, 0.2, -0.1, 0.0])

# Calculate end-effector pose
end_effector_pose = arm.forward_kinematics(joint_angles)
print(f"End-effector position: {end_effector_pose[:3]}")
print(f"End-effector orientation: {end_effector_pose[3:]}")
```

### Inverse Kinematics
```python
# Define target pose [x, y, z, rx, ry, rz]
target_pose = np.array([0.3, 0.2, 0.4, 0.0, 0.0, 0.0])

# Solve for joint angles
joint_solution = arm.inverse_kinematics(target_pose)
print(f"Joint angles: {joint_solution}")
```




## 🔬 Technical Details

### SO-100 ARM Specifications
- **DOF**: 6-axis articulated arm
- **Reach**: 850mm maximum reach
- **Payload**: 5kg maximum payload
- **Joints**: All revolute joints with position/velocity/torque control

### Kinematics Implementation
- **Forward**: Standard DH parameter method
- **Inverse**: Jacobian-based iterative solver with singularity handling
- **Workspace**: Comprehensive reachability analysis included

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 Dependencies

Core dependencies managed by `uv`:
- `mujoco>=3.0.0` - Physics simulation engine
- `numpy>=1.24.0` - Numerical computations
- `scipy>=1.10.0` - Scientific computing (for IK solver)
- `matplotlib>=3.7.0` - Plotting and visualization
- `pytest>=7.0.0` - Testing framework
- `black>=23.0.0` - Code formatting
- `ruff>=0.1.0` - Fast Python linter

## 🐛 Troubleshooting

### Common Issues

**MuJoCo not rendering**
```bash
# Install additional dependencies for headless systems
sudo apt-get install libglfw3-dev libgl1-mesa-dev libglu1-mesa-dev
```

**Permission denied errors**
```bash
# Ensure proper permissions
chmod +x scripts/*.sh
```

**uv sync fails**
```bash
# Clear cache and retry
uv cache clean
uv sync --refresh
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🔗 References

- [MuJoCo Documentation](https://mujoco.readthedocs.io/)
- [SO-100 ARM Technical Specifications](https://example.com/so100-specs)
- [Modern Robotics: Mechanics, Planning, and Control](http://modernrobotics.org/)


---

**Happy Simulating! 🤖**
