"""
Minimal training test - strips away all advanced subsystems to test
if the core RL pipeline (encoder -> router -> experts -> policy) works.
"""
import sys
sys.path.insert(0, "/tmp/deepthough")

import torch
import torch.nn as nn
import numpy as np
import gymnasium as gym
from collections import deque
import time

from deep_thought.config import DeepThoughtConfig
from deep_thought.agent import DeepThoughtAgent

def run_minimal_training():
    device = torch.device("cpu")
    print("="*70)
    print("  MINIMAL Deep Thought Training Test")
    print("="*70)
    
    env = gym.make("CartPole-v1")
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n
    
    # Create MINIMAL config - disable ALL advanced subsystems
    config = DeepThoughtConfig()
    config.observation_dim = obs_dim
    config.action_dim = act_dim
    config.num_actions = act_dim
    config.action_space = "discrete"
    config.device = "cpu"
    
    # Tiny model
    config.encoder.latent_dim = 32
    config.encoder.hidden_dim = 64
    config.router.num_experts = 4
    config.router.active_experts = 2
    config.expert.hidden_dim = 64
    config.expert.num_layers = 1
    config.memory.episodic_capacity = 50
    config.memory.semantic_capacity = 20
    config.memory.episodic_key_dim = 16
    config.memory.episodic_value_dim = 32
    config.memory.semantic_dim = 16
    config.world_model.latent_dim = 32
    config.world_model.hidden_dim = 64
    
    # Training params - optimized for CartPole
    config.training.batch_size = 64
    config.training.rollout_length = 128
    config.training.learning_rate = 3e-4
    config.training.ppo_epochs = 4
    config.training.gamma = 0.99
    config.training.gae_lambda = 0.95
    config.training.clip_eps = 0.2
    config.training.value_loss_coef = 0.25
    config.training.entropy_coef = 0.05
    config.training.target_kl = 0.05
    
    # DISABLE ALL advanced subsystems
    config.planning.use_tcpl = False
    config.opponent_modeling.use_opponent_modeling = False
    config.hierarchical.use_hierarchy = False
    config.compute_economy.use_compute_market = False
    config.shadow_evolution.use_shadow_evolution = False
    config.meta_learning_rules.use_meta_optimizer = False
    config.mechanic_discovery.use_mde = False
    config.autonomous_specialization.use_autonomous_specialization = False
    config.stability_in_the_dark.use_stability_in_the_dark = False
    config.curiosity.use_curiosity = False
    config.attention_maps.use_attention_maps = False
    config.subgoal.use_subgoals = False
    config.meta_learning.use_meta_learning = False
    config.feature_validation.use_fve = False
    config.srp.use_srp = False
    config.governance.use_governor = False
    config.meta_loop.use_meta_loop = False
    config.formal_verification.use_formal_verification = False
    config.dynamic_hyperparams.use_dynamic_hyperparams = False
    config.reasoning.use_reasoning = False
    config.expert_compiler.use_fec = False
    
    # Create agent
    agent = DeepThoughtAgent(config).to(device)
    total_params = sum(p.numel() for p in agent.parameters())
    print(f"  Parameters: {total_params:,}")
    print(f"  latent_dim={config.encoder.latent_dim}, experts={config.router.num_experts}")
    
    optimizer = torch.optim.Adam(agent.parameters(), lr=config.training.learning_rate)
    
    # Training loop - manual PPO
    episode_rewards = deque(maxlen=100)
    all_rewards = []
    step_count = 0
    
    print(f"\n  {'Step':>6} | {'Reward':>8} | {'Avg100':>8} | {'Loss':>8} | {'Entropy':>8} | {'Episodes':>8}")
    print(f"  {'-'*60}")
    
    start_time = time.time()
    
    for iteration in range(50):  # 50 iterations of 128-step rollouts
        # Collect rollout
        obs, _ = env.reset()
        obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        agent.reset(1)
        
        obs_list = []
        actions_list = []
        log_probs_list = []
        values_list = []
        rewards_list = []
        dones_list = []
        
        episode_reward = 0.0
        episodes_in_rollout = 0
        
        for t in range(128):
            with torch.no_grad():
                outputs = agent.forward(obs_tensor, training=False)
                policy_logits = outputs["policy_logits"]
                value = outputs["value"].item()
                
                probs = torch.softmax(policy_logits, dim=-1)
                dist = torch.distributions.Categorical(probs)
                action = dist.sample()
                log_prob = dist.log_prob(action).item()
            
            # Step env
            action_np = action.item()
            next_obs, reward, done, truncated, _ = env.step(action_np)
            done = done or truncated
            
            obs_list.append(obs_tensor.squeeze(0))
            actions_list.append(action)
            log_probs_list.append(log_prob)
            values_list.append(value)
            rewards_list.append(reward)
            dones_list.append(done)
            
            episode_reward += reward
            step_count += 1
            
            if done:
                all_rewards.append(episode_reward)
                episode_rewards.append(episode_reward)
                episode_reward = 0.0
                episodes_in_rollout += 1
                obs, _ = env.reset()
                obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                agent.reset(1)
            else:
                obs_tensor = torch.tensor(next_obs, dtype=torch.float32, device=device).unsqueeze(0)
        
        # Compute GAE
        T = len(rewards_list)
        advantages = []
        returns_gae = []
        gae = 0.0
        
        for t in reversed(range(T)):
            if t == T - 1:
                next_value = 0.0
                next_non_terminal = 0.0
            else:
                next_value = values_list[t + 1]
                next_non_terminal = 1.0 - dones_list[t + 1]
            
            delta = rewards_list[t] + 0.99 * next_value * next_non_terminal - values_list[t]
            gae = delta + 0.99 * 0.95 * next_non_terminal * gae
            advantages.insert(0, gae)
            returns_gae.insert(0, gae + values_list[t])
        
        # Convert to tensors
        obs_batch = torch.stack(obs_list).to(device)
        actions_batch = torch.stack(actions_list).to(device)
        old_log_probs = torch.tensor(log_probs_list, dtype=torch.float32, device=device).detach()
        advantages_tensor = torch.tensor(advantages, dtype=torch.float32, device=device)
        returns_tensor = torch.tensor(returns_gae, dtype=torch.float32, device=device)
        
        # Normalize
        if len(advantages_tensor) > 1:
            advantages_tensor = (advantages_tensor - advantages_tensor.mean()) / (advantages_tensor.std() + 1e-8)
        if len(returns_tensor) > 1:
            returns_tensor = (returns_tensor - returns_tensor.mean()) / (returns_tensor.std() + 1e-8)
        
        # PPO update
        for epoch in range(4):
            # Forward pass through full agent
            agent.train()
            agent.reset(obs_batch.size(0))
            outputs = agent.forward(obs_batch, training=True)
            
            policy_logits = outputs["policy_logits"]
            value = outputs["value"].squeeze(-1)
            
            # Compute new log probs
            probs = torch.softmax(policy_logits, dim=-1)
            dist = torch.distributions.Categorical(probs)
            new_log_probs = dist.log_prob(actions_batch)
            entropy = dist.entropy().mean()
            
            # PPO clipped loss
            ratio = torch.exp(new_log_probs - old_log_probs)
            ratio = torch.clamp(ratio, 0.1, 10.0)  # Stability clamp
            surr1 = ratio * advantages_tensor
            surr2 = torch.clamp(ratio, 1 - 0.2, 1 + 0.2) * advantages_tensor
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # Value loss with clipping
            value_pred_clipped = value + torch.clamp(value - value.detach(), -0.2, 0.2)
            value_loss = torch.max(
                nn.functional.mse_loss(value_pred_clipped, returns_tensor),
                nn.functional.mse_loss(value, returns_tensor)
            )
            
            # Entropy bonus
            entropy_loss = -0.05 * entropy
            
            total_loss = policy_loss + 0.25 * value_loss + entropy_loss
            
            optimizer.zero_grad()
            total_loss.backward()
            nn.utils.clip_grad_norm_(agent.parameters(), 0.5)
            optimizer.step()
        
        # Logging
        if iteration % 5 == 0 or iteration == 49:
            avg100 = np.mean(episode_rewards) if episode_rewards else 0.0
            print(f"  {iteration*128:>6} | {all_rewards[-1] if all_rewards else 0:>8.1f} | {avg100:>8.2f} | {total_loss.item():>8.4f} | {entropy.item():>8.4f} | {len(all_rewards):>8}")
    
    elapsed = time.time() - start_time
    
    print(f"\n  Training time: {elapsed:.1f}s")
    print(f"  Total episodes: {len(all_rewards)}")
    
    if all_rewards:
        print(f"  First 10 rewards: {[f'{r:.0f}' for r in all_rewards[:10]]}")
        print(f"  Last 10 rewards:  {[f'{r:.0f}' for r in all_rewards[-10:]]}")
        print(f"  Mean reward: {np.mean(all_rewards):.2f}")
        print(f"  Last 100 mean: {np.mean(all_rewards[-100:]):.2f}")
        
        first10 = np.mean(all_rewards[:10])
        last10 = np.mean(all_rewards[-10:])
        print(f"  First 10 avg: {first10:.2f}")
        print(f"  Last 10 avg:  {last10:.2f}")
        print(f"  Improvement: {last10 - first10:+.2f}")
        print(f"  Status: {'LEARNING ✓' if last10 > first10 + 5 else 'NOT LEARNING ✗'}")
    
    env.close()

if __name__ == "__main__":
    run_minimal_training()
