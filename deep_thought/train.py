"""
Training script for Deep Thought.
"""

import torch
import torch.optim as optim
import gymnasium as gym
from tqdm import tqdm
import os

from deep_thought.agent import DeepThoughtAgent
from deep_thought.config import DeepThoughtConfig
from deep_thought.optimization.ppo import PPOTrainer
from deep_thought.optimization.schedulers import CosineAnnealingWarmupScheduler
from deep_thought.utils.logger import setup_logger
from deep_thought.utils.metrics import MetricsTracker
from deep_thought.utils.checkpoint import save_checkpoint, load_checkpoint


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
    
    logger.info(f"Training on device: {device}")
    logger.info(f"Environment: {env_id}")
    
    # Create environment
    env = gym.make(env_id)
    observation_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    # Update config
    config.observation_dim = observation_dim
    config.action_dim = action_dim
    config.num_actions = action_dim
    config.action_space = "discrete"
    
    # Create agent
    agent = DeepThoughtAgent(config).to(device)
    
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
    ppo_trainer = PPOTrainer(config.training, agent)
    
    # Create metrics tracker
    metrics_tracker = MetricsTracker()
    
    # Resume from checkpoint
    start_step = 0
    if resume_from is not None and os.path.exists(resume_from):
        logger.info(f"Resuming from {resume_from}")
        checkpoint = load_checkpoint(resume_from, agent, optimizer, device)
        start_step = checkpoint["step"]
    
    # Training loop
    logger.info("Starting training...")
    
    observation, _ = env.reset()
    observation = torch.tensor(observation).unsqueeze(0).float().to(device)
    agent.reset(1)
    
    episode_reward = 0.0
    episode_length = 0
    
    for step in tqdm(range(start_step, total_steps), desc="Training"):
        # Collect rollout
        rollout_reward, rollout_length = ppo_trainer.collect_rollout(
            env, observation, agent.h_t, agent.m_t
        )
        
        episode_reward += rollout_reward
        episode_length += rollout_length
        
        # Update
        metrics = ppo_trainer.update(optimizer)
        scheduler.step()
        
        # Update agent systems
        if step % config.training.prune_interval == 0:
            agent.prune_experts()
        
        if step % config.training.growth_interval == 0:
            agent.grow_experts()
        
        if step % 10000 == 0:
            agent.consolidate_memory()
            agent.validate_features()
        
        # Update SRP
        agent.update_srp(rollout_reward, metrics.get("total_loss", 0.0))
        
        # Track metrics
        metrics_tracker.update(metrics)
        
        # Reset if episode done
        done = rollout_length < config.training.rollout_length
        if done:
            metrics_tracker.add_episode(episode_reward, episode_length)
            episode_reward = 0.0
            episode_length = 0
            observation, _ = env.reset()
            observation = torch.tensor(observation).unsqueeze(0).float().to(device)
            agent.reset(1)
        else:
            # Get last observation from buffer
            if len(ppo_trainer.buffer) > 0:
                observation = ppo_trainer.buffer.observations[-1].to(device)
        
        # Logging
        if step % config.log_interval == 0:
            episode_stats = metrics_tracker.get_episode_stats()
            agent_stats = agent.get_stats()
            
            logger.info(
                f"Step {step} | "
                f"Reward: {episode_stats['mean_reward']:.2f} | "
                f"Length: {episode_stats['mean_length']:.1f} | "
                f"Experts: {agent_stats['num_experts']} | "
                f"Active: {agent_stats['active_experts']} | "
                f"Loss: {metrics_tracker.get_metric('total_loss'):.4f}"
            )
        
        # Evaluation
        if step % config.eval_interval == 0 and step > 0:
            eval_reward = evaluate(agent, env, device, num_episodes=10)
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
    env: gym.Env,
    device: torch.device,
    num_episodes: int = 10
) -> float:
    """
    Evaluate agent.
    
    Args:
        agent: Agent to evaluate
        env: Environment
        device: Device
        num_episodes: Number of episodes
        
    Returns:
        mean_reward: Mean episode reward
    """
    agent.eval()
    
    total_rewards = []
    
    for _ in range(num_episodes):
        observation, _ = env.reset()
        observation = torch.tensor(observation).unsqueeze(0).float().to(device)
        agent.reset(1)
        
        episode_reward = 0.0
        done = False
        
        while not done:
            with torch.no_grad():
                action, _, _ = agent.act(observation, deterministic=True)
            
            observation, reward, done, truncated, _ = env.step(
                action.cpu().numpy()[0]
            )
            done = done or truncated
            observation = torch.tensor(observation).unsqueeze(0).float().to(device)
            
            episode_reward += reward
        
        total_rewards.append(episode_reward)
    
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
