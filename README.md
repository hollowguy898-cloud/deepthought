# Deep Thought

**Adaptive Sparse Cognitive Network for Reinforcement Learning**

Deep Thought is a novel RL architecture that combines sparse neural networks, mixture-of-experts routing, synaptic pruning, neuroplasticity, and fast adaptive memory systems into a single, self-optimizing cognitive system.

## Core Philosophy

Instead of running all parameters on every computation, Deep Thought:

- **Activates only relevant subnetworks** per task
- **Temporarily suppresses unused regions** (dormant neurons)
- **Permanently prunes weak neurons/connections** 
- **Grows new neurons when needed** (neurogenesis)
- **Learns task-specific pathways** dynamically
- **Maintains multi-timescale memory** (working, episodic, semantic, procedural)

## Architecture Overview

Deep Thought consists of 5 adaptive layers:

1. **Perception & Encoding** - Compresses observations into factorized latent space
2. **Context Inference** - Determines "what kind of problem is this?"
3. **Sparse Cognitive Graph** - Expert bank with dynamic routing
4. **Fast Adaptation Layer** - Meta-learning + fast weights
5. **Memory & World Model** - Experience + prediction

### Key Components

- **Sparse Router** - Top-k expert selection with load balancing
- **Expert Bank** - Specialized MLP modules (128 experts, 4 active per step)
- **Feature Validation Engine (FVE)** - Validates features before integration
- **Feature → Expert Compiler (FEC)** - Converts validated features into experts
- **Temporal Cognition & Planning Layer (TCPL)** - Orchestrates experts over time
- **Self-Regression Prevention (SRP)** - Prevents performance collapse
- **Multi-Scale Memory** - Working, episodic, semantic, procedural layers

## Installation

```bash
pip install -e .
```

For Atari environments:
```bash
pip install -e ".[atari]"
```

For MuJoCo environments:
```bash
pip install -e ".[mujoco]"
```

## Quick Start

```python
from deep_thought import DeepThoughtAgent
from deep_thought.config import DeepThoughtConfig

# Create configuration
config = DeepThoughtConfig(
    latent_dim=1024,
    num_experts=128,
    active_experts=4,
    use_world_model=True,
    use_memory=True,
)

# Create agent
agent = DeepThoughtAgent(config)

# Train
agent.train(env, total_steps=10_000_000)
```

## Key Innovations

### 1. Conditional Intelligence
The model becomes specialized dynamically based on context and task requirements.

### 2. Lower Compute
Massive efficiency gain through sparse activation (only ~3% of parameters active per step).

### 3. Faster Learning
Localized adaptation via fast weights and Hebbian learning rules.

### 4. Reduced Catastrophic Forgetting
Unused skills preserved in dormant state, recoverable when needed.

### 5. Self-Optimization
Architecture evolves itself through pruning, growth, and feature validation.

## Training Pipeline

1. **Representation Warmup** - Train encoder + world model only (5-20M frames)
2. **Sparse RL Activation** - Enable experts + router with PPO/IMPALA
3. **Controlled Specialization** - Add compute penalty, reduce active experts
4. **Pruning Begins** - Remove low-utility experts after stable learning
5. **Growth + Adaptation** - Spawn new experts when capacity insufficient

## Metrics

We optimize for **Capability Density**:
```
Capability Density = Reward / Active Parameters
```

This metric rewards parameter efficiency, not just raw performance.

## Project Structure

```
deep_thought/
├── architecture/          # Core architecture components
│   ├── encoder.py         # Observation encoder
│   ├── router.py          # Sparse routing system
│   ├── experts.py         # Expert bank
│   ├── memory/            # Memory systems
│   ├── world_model.py     # Latent dynamics
│   └── planning/          # Temporal coordination
├── learning/              # Learning systems
│   ├── meta_learning.py   # Meta-learning layer
│   ├── fast_weights.py    # Fast weight adaptation
│   └── feature_validation.py  # FVE
├── optimization/          # Training infrastructure
│   ├── ppo.py             # PPO implementation
│   ├── losses.py          # Loss functions
│   └── schedulers.py      # Learning rate schedules
├── stability/             # Stability systems
│   ├── srs.py             # Self-Regression Prevention
│   └── monitoring.py      # Performance monitoring
├── config/                # Configuration
├── utils/                 # Utilities
├── examples/              # Usage examples
└── tests/                 # Tests
```

## Citation

If you use Deep Thought in your research, please cite:

```bibtex
@misc{deepthought2024,
  title={Deep Thought: Adaptive Sparse Cognitive Networks for Reinforcement Learning},
  author={Deep Thought Contributors},
  year={2024},
  howpublished={\url{https://github.com/hollowguy898/deepthough}}
}
```

## License

MLP License - see LICENSE file for details.

## Acknowledgments

Inspired by:
- Human brain architecture (sparse activation, neuroplasticity)
- Mixture-of-Experts (MoE) systems
- Synaptic pruning in biological neural networks
- Meta-learning and fast adaptation research

## Contributing

Contributions welcome! Please see CONTRIBUTING.md for guidelines.

## Disclaimer

This is research code. Training sparse adaptive systems is fragile and requires careful tuning. Expect debugging.
