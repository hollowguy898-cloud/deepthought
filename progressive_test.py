"""Progressive test - add subsystems one at a time to find what breaks learning."""
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

def run_test(name, config_mods, steps=6400):
    device = torch.device("cpu")
    env = gym.make("CartPole-v1")
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n
    
    config = DeepThoughtConfig()
    config.observation_dim = obs_dim
    config.action_dim = act_dim
    config.num_actions = act_dim
    config.action_space = "discrete"
    config.device = "cpu"
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
    config.training.batch_size = 64
    config.training.rollout_length = 128
    config.training.learning_rate = 3e-4
    config.training.ppo_epochs = 4
    config.training.value_loss_coef = 0.25
    config.training.entropy_coef = 0.05
    config.training.target_kl = 0.05
    
    # Disable all by default
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
    
    # Apply mods
    for key, val in config_mods.items():
        parts = key.split('.')
        obj = config
        for p in parts[:-1]:
            obj = getattr(obj, p)
        setattr(obj, parts[-1], val)
    
    agent = DeepThoughtAgent(config).to(device)
    optimizer = torch.optim.Adam(agent.parameters(), lr=config.training.learning_rate)
    
    episode_rewards = deque(maxlen=100)
    all_rewards = []
    
    iterations = steps // 128
    for iteration in range(iterations):
        obs, _ = env.reset()
        obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        agent.reset(1)
        
        obs_list, actions_list, log_probs_list, values_list = [], [], [], []
        rewards_list, dones_list = [], []
        episode_reward = 0.0
        
        for t in range(128):
            with torch.no_grad():
                outputs = agent.forward(obs_tensor, training=False)
                policy_logits = outputs["policy_logits"]
                value = outputs["value"].item()
                probs = torch.softmax(policy_logits, dim=-1)
                dist = torch.distributions.Categorical(probs)
                action = dist.sample()
                log_prob = dist.log_prob(action).item()
            
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
            
            if done:
                all_rewards.append(episode_reward)
                episode_rewards.append(episode_reward)
                episode_reward = 0.0
                obs, _ = env.reset()
                obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                agent.reset(1)
            else:
                obs_tensor = torch.tensor(next_obs, dtype=torch.float32, device=device).unsqueeze(0)
        
        # GAE
        T = len(rewards_list)
        advantages, returns_gae = [], []
        gae = 0.0
        for t in reversed(range(T)):
            next_val = values_list[t + 1] if t < T - 1 else 0.0
            next_nt = (1.0 - dones_list[t + 1]) if t < T - 1 else 0.0
            delta = rewards_list[t] + 0.99 * next_val * next_nt - values_list[t]
            gae = delta + 0.99 * 0.95 * next_nt * gae
            advantages.insert(0, gae)
            returns_gae.insert(0, gae + values_list[t])
        
        obs_batch = torch.stack(obs_list).to(device)
        actions_batch = torch.stack(actions_list).to(device)
        old_log_probs = torch.tensor(log_probs_list, dtype=torch.float32, device=device).detach()
        adv_tensor = torch.tensor(advantages, dtype=torch.float32, device=device)
        ret_tensor = torch.tensor(returns_gae, dtype=torch.float32, device=device)
        
        if len(adv_tensor) > 1:
            adv_tensor = (adv_tensor - adv_tensor.mean()) / (adv_tensor.std() + 1e-8)
        if len(ret_tensor) > 1:
            ret_tensor = (ret_tensor - ret_tensor.mean()) / (ret_tensor.std() + 1e-8)
        
        # PPO
        for epoch in range(4):
            agent.train()
            agent.reset(obs_batch.size(0))
            outputs = agent.forward(obs_batch, training=True)
            policy_logits = outputs["policy_logits"]
            value = outputs["value"].squeeze(-1)
            
            probs = torch.softmax(policy_logits, dim=-1)
            dist = torch.distributions.Categorical(probs)
            new_log_probs = dist.log_prob(actions_batch)
            entropy = dist.entropy().mean()
            
            ratio = torch.exp(torch.clamp(new_log_probs - old_log_probs, -10, 10))
            surr1 = ratio * adv_tensor
            surr2 = torch.clamp(ratio, 0.8, 1.2) * adv_tensor
            policy_loss = -torch.min(surr1, surr2).mean()
            
            value_pred_clipped = value + torch.clamp(value - value.detach(), -0.2, 0.2)
            value_loss = torch.max(
                nn.functional.mse_loss(value_pred_clipped, ret_tensor),
                nn.functional.mse_loss(value, ret_tensor)
            )
            
            total_loss = policy_loss + 0.25 * value_loss - 0.05 * entropy
            
            optimizer.zero_grad()
            total_loss.backward()
            nn.utils.clip_grad_norm_(agent.parameters(), 0.5)
            optimizer.step()
    
    env.close()
    
    first10 = np.mean(all_rewards[:10]) if len(all_rewards) >= 10 else np.mean(all_rewards)
    last10 = np.mean(all_rewards[-10:]) if len(all_rewards) >= 10 else np.mean(all_rewards)
    avg_last100 = np.mean(all_rewards[-100:]) if all_rewards else 0.0
    
    return first10, last10, avg_last100, len(all_rewards)

# Test configurations
tests = [
    ("Baseline (no extras)", {}),
    ("+ World model", {"world_model.use_world_model": True}),
    ("+ Meta-learning", {"meta_learning.use_meta_learning": True, "meta_learning.context_dim": 32}),
    ("+ Curiosity", {"curiosity.use_curiosity": True, "curiosity.state_embedding_dim": 16}),
    ("+ Attention maps", {"attention_maps.use_attention_maps": True, "attention_maps.num_heads": 4}),
    ("+ Reasoning engine", {"reasoning.use_reasoning": True, "reasoning.num_reasoning_steps": 2, "reasoning.num_counterfactual_actions": 2}),
    ("+ Governor", {"governance.use_governor": True}),
    ("+ SRP", {"srp.use_srp": True}),
    ("+ Expert compiler", {"expert_compiler.use_fec": True}),
    ("+ Subgoals", {"subgoal.use_subgoals": True, "subgoal.goal_embedding_dim": 32}),
    ("+ Meta-loop", {"meta_loop.use_meta_loop": True}),
    ("+ Dynamic hyperparams", {"dynamic_hyperparams.use_dynamic_hyperparams": True}),
    ("All together", {
        "world_model.use_world_model": True,
        "meta_learning.use_meta_learning": True, "meta_learning.context_dim": 32,
        "curiosity.use_curiosity": True, "curiosity.state_embedding_dim": 16,
        "attention_maps.use_attention_maps": True, "attention_maps.num_heads": 4,
        "reasoning.use_reasoning": True, "reasoning.num_reasoning_steps": 2, "reasoning.num_counterfactual_actions": 2,
        "governance.use_governor": True,
        "srp.use_srp": True,
        "expert_compiler.use_fec": True,
        "subgoal.use_subgoals": True, "subgoal.goal_embedding_dim": 32,
        "meta_loop.use_meta_loop": True,
        "dynamic_hyperparams.use_dynamic_hyperparams": True,
    }),
]

print(f"\n{'Name':<30} | {'First10':>8} | {'Last10':>8} | {'AvgLast100':>10} | {'Improve':>8} | {'Status':>8}")
print("-" * 90)

for name, mods in tests:
    try:
        first10, last10, avg_last100, n_eps = run_test(name, mods, steps=6400)
        improvement = last10 - first10
        status = "✓" if improvement > 0 else "✗"
        print(f"{name:<30} | {first10:>8.1f} | {last10:>8.1f} | {avg_last100:>10.1f} | {improvement:>+8.1f} | {status:>8}")
    except Exception as e:
        print(f"{name:<30} | ERROR: {str(e)[:50]}")
