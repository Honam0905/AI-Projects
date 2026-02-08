# Tennis PettingZoo Demo

A demonstration project showcasing multi-agent reinforcement learning using PettingZoo's Tennis environment. This self-contained notebook implements and trains agents to play tennis in a competitive setting.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
- [Environment Details](#environment-details)
- [Training](#training)
- [Results](#results)
- [License](#license)

## Overview

This project demonstrates the implementation of multi-agent reinforcement learning algorithms using the PettingZoo library's Tennis environment. The notebook is fully self-contained with all dependencies, configurations, and implementations included within the single file.

## Features

- 🎾 Tennis environment simulation using PettingZoo
- 🤖 Multi-agent reinforcement learning implementation
- 📦 Self-contained notebook with dependency installation
- 📊 Training visualization and metrics
- 🔄 Interactive demonstration of agent gameplay
- 📈 Performance analysis and evaluation

## Prerequisites

- Python 3.8+
- Jupyter Notebook or JupyterLab

**Note**: All required packages are installed automatically within the notebook itself.

## Installation

1. Clone the repository or download the notebook:
```bash
git clone <repository-url>
cd petting-zoo_demo
```

2. Launch Jupyter Notebook:
```bash
jupyter notebook tennis_petting_zoo.ipynb
```

## Usage

Simply open `tennis_petting_zoo.ipynb` in Jupyter and run the cells sequentially:

1. **Dependency Installation**: The first cells install all required packages
2. **Environment Setup**: Initialize the Tennis environment
3. **Agent Configuration**: Set up agent parameters and architecture
4. **Training Loop**: Train the agents through multiple episodes
5. **Visualization**: View training metrics and gameplay
6. **Evaluation**: Test trained agents and analyze performance

The notebook handles everything from installation to evaluation in a single workflow.

## Environment Details

The Tennis environment is a competitive two-player game where:
- **Agents**: 2 players (left and right)
- **Observation Space**: Visual observations of the game state
- **Action Space**: Discrete actions (move up, down, stay)
- **Reward Structure**: +1 for scoring, -1 for opponent scoring
- **Episode Length**: Variable, ends when one player reaches winning score

### Action Space
- 0: Move up
- 1: Stay
- 2: Move down

### Observation Space
- RGB pixel observations
- Shape: [210, 160, 3]

## Training

The notebook implements training using **Proximal Policy Optimization (PPO)** from Stable-Baselines3.

### Algorithm Configuration

- **Algorithm**: PPO (Proximal Policy Optimization)
- **Policy**: CnnPolicy (Convolutional Neural Network)
- **Total Training Steps**: 3,000,000 environment steps
- **Number of Parallel Environments**: 32
- **Device**: CUDA (if available) or CPU

### Hyperparameters

- **Learning Rate**: 2.5e-4
- **Rollout Steps (n_steps)**: 128
- **Batch Size**: 1024
- **Optimization Epochs (n_epochs)**: 4
- **Discount Factor (γ)**: 0.99
- **GAE Lambda**: 0.95
- **Clip Range**: 0.1
- **Entropy Coefficient**: 0.01
- **Value Function Coefficient**: 0.5
- **Max Gradient Norm**: 0.5

### Environment Preprocessing

The environment uses standard Atari preprocessing pipeline:
- Max observation (2 frames) - reduces flickering
- Frame skip (4) - action repeat for efficiency
- Color reduction to grayscale
- Resize to 84x84 pixels
- Frame stacking (4 frames) - temporal information
- Reward clipping [-1, 1]
- Observation normalization [0, 1]
- Agent indicator for parameter sharing
- Black death wrapper for multi-agent safety

### Callbacks

- **EvalCallback**: Evaluates model every 250,000 steps on 10 episodes
- **CheckpointCallback**: Saves model checkpoints every 1,000,000 steps

### Training Outputs

- Best model: `./tennis_sb3_logs/best/best_model.zip`
- Final model: `./tennis_sb3_logs/final_ppo_tennis.zip`
- Checkpoints: `./tennis_sb3_logs/checkpoints/`
- TensorBoard logs: `./tennis_sb3_logs/`
- Training monitor: `./tennis_sb3_logs/train_monitor.csv`
- Evaluation monitor: `./tennis_sb3_logs/eval_monitor.csv`

All configurations can be modified directly in the notebook cells.

## Results

Training results include:
- Agent win rates over time
- Average episode rewards (rollout/ep_rew_mean)
- Evaluation mean rewards
- PPO training metrics (value loss, policy gradient loss, entropy loss, approx KL)
- Real-time gameplay demonstrations saved as MP4 video

## Demo video


https://github.com/user-attachments/assets/6f958123-a5ad-43b5-9521-cdbed48ecfc8



## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [PettingZoo](https://pettingzoo.farama.org/) - Multi-agent RL environments
- [Gymnasium](https://gymnasium.farama.org/) - RL environment standard
- [Stable-Baselines3](https://stable-baselines3.readthedocs.io/) - RL algorithms implementation
- [SuperSuit](https://github.com/Farama-Foundation/SuperSuit) - Wrappers for multi-agent environments

## References

- PettingZoo Documentation: https://pettingzoo.farama.org/
- Tennis Environment: https://pettingzoo.farama.org/environments/atari/tennis/
- Stable-Baselines3 PPO: https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html

## Contact

For questions or issues, please open an issue on the GitHub repository.

---

**Note**: This is a self-contained demonstration project for educational purposes. All dependencies are installed automatically within the notebook.
