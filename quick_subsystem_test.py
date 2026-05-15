"""Quick test of specific subsystems to find what breaks learning."""
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

def quick_train(name, config_mods, steps=3200):
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
    for attr in ['planning', 'opponent_modeling', 'hierarchical', 'compute_economy',
                 'shadow_evolution', 'meta_learning_rules', 'mechanic_discovery',
                 'autonomous_specialization', 'stability_in_the_dark', 'curiosity',
                 'attention_maps', 'subgoal', 'meta_learning', 'feature_validation',
                 'srp', 'governance', 'meta_loop', 'formal_verification',
                 'dynamic_hyperparams', 'reasoning', 'expert_compiler']:
        subconfig = getattr(config, attr)
        for field_name in dir(subconfig):
            if field_name.startswith('use_'):
                setattr(subconfig, field_name, False)
    
    for key, val in config_mods.items():
        parts = key.split('.')
        obj = config
        for p in parts[:-1]:
            obj = getattr(obj, p)
        setattr(obj, parts[-1], val)
    
    agent = DeepThoughtAgent(config).to(device)
    optimizer = torch.optim.Adam(agent.parameters(), lr=config.training.learning_rate)
    
    all_rewards = []
    episode_reward = 0.0
    obs, _ = env.reset()
    obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
    agent.reset(1)
    
    for step in range(steps):
        with torch.no_grad():
            outputs = agent.forward(obs_tensor, training=False)
            policy_logits = outputs["policy_logits"]
            probs = torch.softmax(policy_logits, dim=-1)
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()
        
        next_obs, reward, done, truncated, _ = env.step(action.item())
        done = done or truncated
        episode_reward += reward
        
        if done:
            all_rewards.append(episode_reward)
            episode_reward = 0.0
            obs, _ = env.reset()
            obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            agent.reset(1)
        else:
            obs_tensor = torch.tensor(next_obs, dtype=torch.float32, device=device).unsqueeze(0)
    
    env.close()
    
    if len(all_rewards) < 5:
        return 0, 0, 0
    
    first5 = np.mean(all_rewards[:5])
    last5 = np.mean(all_rewards[-5:])
    return first5, last5, last5 - first5

tests = [
    ("Baseline", {}),
    ("+WorldModel", {"world_model.use_world_model": True}),
    ("+Reasoning", {"reasoning.use_reasoning": True, "reasoning.num_reasoning_steps": 2, "reasoning.num_counterfactual_actions": 2}),
    ("+Governor+SRP", {"governance.use_governor": True, "srp.use_srp": True}),
    ("+Curiosity", {"curiosity.use_curiosity": True, "curiosity.state_embedding_dim": 16}),
    ("+Attention", {"attention_maps.use_attention_maps": True, "attention_maps.num_heads": 4}),
    ("+MetaLearn", {"meta_learning.use_meta_learning": True, "meta_learning.context_dim": 32}),
]

print(f"\n{'Name':<20} | {'First5':>8} | {'Last5':>8} | {'Delta':>8} | Status")
print("-" * 60)

for name, mods in tests:
    try:
        first5, last5, delta = quick_train(name, mods, steps=3200)
        status = "✓" if delta > 0 else "✗"
        print(f"{name:<20} | {first5:>8.1f} | {last5:>8.1f} | {delta:>+8.1f} | {status}")
    except Exception as e:
        print(f"{name:<20} | ERROR: {str(e)[:40]}")
