"""
Training script for Deep Thought.
"""

import torch
import torch.optim as optim
import gymnasium as gym
from tqdm import tqdm
import os
from typing import Optional
import numpy as np

from deep_thought.agent import DeepThoughtAgent
from deep_thought.config import DeepThoughtConfig
from deep_thought.optimization.ppo import PPOTrainer
from deep_thought.optimization.schedulers import CosineAnnealingWarmupScheduler
from deep_thought.utils.logger import setup_logger
from deep_thought.utils.metrics import MetricsTracker
from deep_thought.utils.checkpoint import save_checkpoint, load_checkpoint


def _get_env_info(env_id: str):
    """Get observation dim, action dim, and action space type from env."""
    with gym.make(env_id) as env:
        # Observation space
        if isinstance(env.observation_space, gym.spaces.Box):
            if len(env.observation_space.shape) == 1:
                observation_dim = env.observation_space.shape[0]
            else:
                # Image observation - flatten
                observation_dim = int(np.prod(env.observation_space.shape))
        elif isinstance(env.observation_space, gym.spaces.Dict):
            # Take first key
            first_key = list(env.observation_space.spaces.keys())[0]
            observation_dim = int(np.prod(env.observation_space[first_key].shape))
        else:
            observation_dim = 1
        
        # Action space
        if isinstance(env.action_space, gym.spaces.Discrete):
            action_dim = env.action_space.n
            num_actions = action_dim
            action_space_type = "discrete"
        elif isinstance(env.action_space, gym.spaces.Box):
            action_dim = env.action_space.shape[0]
            num_actions = action_dim * 2  # mean + log_std for continuous
            action_space_type = "continuous"
        elif isinstance(env.action_space, gym.spaces.MultiDiscrete):
            action_dim = env.action_space.nvec.sum()
            num_actions = action_dim
            action_space_type = "discrete"
        else:
            action_dim = env.action_space.n
            num_actions = action_dim
            action_space_type = "discrete"
    
    return observation_dim, action_dim, num_actions, action_space_type


def train(
    config: DeepThoughtConfig,
    env_id: str = "CartPole-v1",
    total_steps: int = 10_000_000,
    resume_from: Optional[str] = None
):
    """
    Train Deep Thought agent.
    
    Args:
        config: Deep Thought configuration
        env_id: Gym environment ID
        total_steps: Total training steps
        resume_from: Path to checkpoint to resume from
    """
    # Setup
    logger = setup_logger("deep_thought", config.log_dir)
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    
    # GPU VERIFICATION: Log detailed device info so we can confirm GPU usage
    logger.info(f"Training on device: {device}")
    if device.type == "cuda":
        logger.info(f"  CUDA device: {torch.cuda.get_device_name(0)}")
        logger.info(f"  CUDA memory: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")
    else:
        logger.warning("WARNING: Training on CPU! Set config.device='cuda' and ensure CUDA is available.")
    logger.info(f"Environment: {env_id}")
    
    # Create environment to get info
    env = gym.make(env_id)
    observation_dim, action_dim, num_actions, action_space_type = _get_env_info(env_id)
    
    logger.info(f"Observation dim: {observation_dim}, Action dim: {action_dim}, "
                f"Action space: {action_space_type}")
    
    # Update config
    config.observation_dim = observation_dim
    config.action_dim = action_dim
    config.num_actions = num_actions
    config.action_space = action_space_type
    
    # Create agent
    agent = DeepThoughtAgent(config).to(device)
    
    # GPU VERIFICATION: Confirm model is on GPU
    model_device = next(agent.parameters()).device
    logger.info(f"Agent created with {sum(p.numel() for p in agent.parameters())} parameters")
    logger.info(f"Agent device: {model_device}")
    if model_device.type != device.type:
        logger.warning(f"WARNING: Agent is on {model_device} but expected {device}!")
    
    # Verify all submodules are on the correct device
    for name, param in agent.named_parameters():
        if param.device.type != device.type:
            logger.warning(f"  Parameter {name} on {param.device} (expected {device})")
            break
    else:
        logger.info(f"All {sum(1 for _ in agent.parameters())} parameter tensors on {device}")
    
    # Create optimizer
    optimizer = optim.Adam(
        agent.parameters(),
        lr=config.training.learning_rate
    )
    
    # Create scheduler
    scheduler = CosineAnnealingWarmupScheduler(
        optimizer,
        warmup_steps=10000,
        total_steps=total_steps
    )
    
    # Create PPO trainer
    ppo_trainer = PPOTrainer(
        config.training, agent,
        action_space=config.action_space,
        action_dim=config.action_dim
    )
    
    # Create metrics tracker
    metrics_tracker = MetricsTracker()
    
    # Resume from checkpoint
    start_step = 0
    if resume_from is not None and os.path.exists(resume_from):
        logger.info(f"Resuming from {resume_from}")
        checkpoint = load_checkpoint(resume_from, agent, optimizer, str(device))
        start_step = checkpoint["step"]
    
    # Training loop
    logger.info("Starting training...")
    
    observation, _ = env.reset()
    observation = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)
    agent.reset(1)
    
    episode_reward = 0.0
    episode_length = 0
    
    for step in tqdm(range(start_step, total_steps), desc="Training"):
        # Collect rollout
        rollout_reward, rollout_length, episode_done, last_observation = ppo_trainer.collect_rollout(
            env, observation, agent.h_t, agent.m_t, device
        )
        
        episode_reward += rollout_reward
        episode_length += rollout_length
        
        # Compute bootstrap value for GAE when episode is not done
        if not episode_done and len(ppo_trainer.buffer) > 0:
            with torch.no_grad():
                bootstrap_latent, _ = agent.encoder(last_observation)
                bootstrap_value = agent.critic_head(bootstrap_latent).item()
        else:
            bootstrap_value = 0.0
        
        # Update
        metrics = ppo_trainer.update(optimizer, bootstrap_value=bootstrap_value)
        scheduler.step()
        
        # Update agent systems (now governed by timescale controller)
        # Fix 2: All operations respect time-scale separation via governor
        agent.prune_experts()      # Governor checks if SLOW timescale allows
        agent.grow_experts()       # Governor checks if SLOW timescale allows
        agent.consolidate_memory() # Governor checks if MEDIUM timescale allows
        agent.validate_features()  # Governor checks if SLOW timescale allows

        # Update SRP
        agent.update_srp(rollout_reward, metrics.get("total_loss", 0.0))
        
        # Track metrics
        metrics_tracker.update(metrics)
        
        # Handle episode reset
        if episode_done:
            metrics_tracker.add_episode(episode_reward, episode_length)
            episode_reward = 0.0
            episode_length = 0
            observation, _ = env.reset()
            observation = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)
            agent.reset(1)
        else:
            # Use the actual last observation from the rollout (after the last env.step)
            observation = last_observation.detach()
        
        # Logging
        if step % config.log_interval == 0:
            episode_stats = metrics_tracker.get_episode_stats()
            agent_stats = agent.get_stats()
            total_loss = metrics_tracker.get_metric("total_loss")
            
            log_msg = (
                f"Step {step} | "
                f"Reward: {episode_stats['mean_reward']:.2f} | "
                f"Length: {episode_stats['mean_length']:.1f} | "
                f"Experts: {agent_stats['num_experts']} | "
                f"Active: {agent_stats['active_experts']}"
            )
            if total_loss is not None:
                log_msg += f" | Loss: {total_loss:.4f}"
            logger.info(log_msg)
        
        # Evaluation
        if step % config.eval_interval == 0 and step > 0:
            eval_reward = evaluate(agent, env_id, device, num_episodes=10)
            logger.info(f"Evaluation reward: {eval_reward:.2f}")
        
        # Checkpoint
        if step % config.save_interval == 0 and step > 0:
            checkpoint_path = os.path.join(
                config.log_dir,
                f"checkpoint_{step}.pt"
            )
            save_checkpoint(
                agent,
                optimizer,
                step,
                checkpoint_path,
                extra_data={"metrics": metrics_tracker.get_all_metrics()}
            )
            logger.info(f"Saved checkpoint to {checkpoint_path}")
    
    logger.info("Training complete!")
    
    # Final save
    os.makedirs(config.log_dir, exist_ok=True)
    final_path = os.path.join(config.log_dir, "final_checkpoint.pt")
    save_checkpoint(
        agent,
        optimizer,
        total_steps,
        final_path,
        extra_data={"metrics": metrics_tracker.get_all_metrics()}
    )


def evaluate(
    agent: DeepThoughtAgent,
    env_id: str,
    device: torch.device,
    num_episodes: int = 10
) -> float:
    """
    Evaluate agent.
    
    Args:
        agent: Agent to evaluate
        env_id: Environment ID (create fresh env to avoid state issues)
        device: Device
        num_episodes: Number of episodes
        
    Returns:
        mean_reward: Mean episode reward
    """
    agent.eval()
    
    env = gym.make(env_id)
    total_rewards = []
    
    for _ in range(num_episodes):
        observation, _ = env.reset()
        observation = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)
        agent.reset(1)
        
        episode_reward = 0.0
        done = False
        
        while not done:
            with torch.no_grad():
                action, _, _ = agent.act(observation, deterministic=True)
            
            action_np = action.cpu().numpy()
            if action_np.ndim == 0:
                action_np = action_np.item()
            else:
                action_np = action_np[0]
            
            observation, reward, done, truncated, _ = env.step(action_np)
            done = done or truncated
            observation = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)
            
            episode_reward += reward
        
        total_rewards.append(episode_reward)
    
    env.close()
    agent.train()
    
    return sum(total_rewards) / len(total_rewards)


if __name__ == "__main__":
    # Default configuration
    config = DeepThoughtConfig()
    
    # Override for quick test
    config.observation_dim = 4
    config.action_dim = 2
    config.num_actions = 2
    config.action_space = "discrete"
    config.encoder.latent_dim = 256
    config.encoder.hidden_dim = 512
    config.router.num_experts = 32
    config.router.active_experts = 2
    config.training.batch_size = 64
    config.training.rollout_length = 128
    config.log_dir = "./logs"
    
    train(config, env_id="CartPole-v1", total_steps=100000)
