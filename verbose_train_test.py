"""
Verbose training test for Deep Thought.
"""

import sys
import os
import torch
import torch.nn as nn
import numpy as np
import gymnasium as gym
from collections import defaultdict
import traceback
import time

sys.path.insert(0, "/tmp/deepthough")

from deep_thought.config import DeepThoughtConfig
from deep_thought.agent import DeepThoughtAgent
from deep_thought.optimization.ppo import PPOTrainer
from deep_thought.optimization.schedulers import CosineAnnealingWarmupScheduler


def print_header(msg):
    print(f"\n{'='*70}")
    print(f"  {msg}")
    print(f"{'='*70}")

def print_subheader(msg):
    print(f"\n--- {msg} ---")

def check_device_consistency(agent, device):
    issues = []
    for name, param in agent.named_parameters():
        if param.device.type != device.type:
            issues.append(f"  PARAM: {name} on {param.device} (expected {device})")
    for name, buf in agent.named_buffers():
        if buf.device.type != device.type:
            issues.append(f"  BUFFER: {name} on {buf.device} (expected {device})")
    if issues:
        print(f"  FOUND {len(issues)} DEVICE ISSUES:")
        for i in issues[:20]:
            print(i)
    else:
        print(f"  All params/buffers on {device} ✓")
    return len(issues)

def check_gradient_flow(agent):
    grad_info = defaultdict(list)
    no_grad_params = []
    for name, param in agent.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            module = name.split('.')[0]
            grad_info[module].append((name, grad_norm))
        elif param.requires_grad:
            no_grad_params.append(name)
    
    print("  Modules with gradients:")
    for module, params in sorted(grad_info.items()):
        total_norm = sum(n for _, n in params)
        count = len(params)
        print(f"    {module}: {count} params, grad_norm={total_norm:.6f}")
    
    if no_grad_params:
        print(f"\n  WARNING: {len(no_grad_params)} params NO gradient:")
        for n in no_grad_params[:10]:
            print(f"    - {n}")
    else:
        print(f"\n  All trainable params received gradients ✓")
    return len(no_grad_params)

def run_verbose_training(steps=500, env_id="CartPole-v1"):
    print_header("Deep Thought Verbose Training Test")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")
    print(f"  PyTorch: {torch.__version__}")
    
    env = gym.make(env_id)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n
    print(f"  Environment: {env_id} (obs={obs_dim}, act={act_dim})")
    
    print_subheader("Configuration")
    config = DeepThoughtConfig()
    config.observation_dim = obs_dim
    config.action_dim = act_dim
    config.num_actions = act_dim
    config.action_space = "discrete"
    config.device = str(device)
    
    # Scale down for fast CPU testing
    config.encoder.latent_dim = 64
    config.encoder.hidden_dim = 128
    config.router.num_experts = 8
    config.router.active_experts = 2
    config.expert.hidden_dim = 128
    config.expert.num_layers = 1
    
    config.memory.episodic_capacity = 100
    config.memory.semantic_capacity = 50
    config.memory.episodic_key_dim = 32
    config.memory.episodic_value_dim = 64
    config.memory.semantic_dim = 32
    
    config.world_model.latent_dim = 64
    config.world_model.hidden_dim = 128
    
    config.training.batch_size = 32
    config.training.rollout_length = 32
    config.training.learning_rate = 3e-4
    config.training.ppo_epochs = 2
    
    # Disable heavy subsystems for speed
    config.planning.use_tcpl = False
    config.opponent_modeling.use_opponent_modeling = False
    config.hierarchical.use_hierarchy = False
    config.compute_economy.use_compute_market = False
    config.shadow_evolution.use_shadow_evolution = False
    config.meta_learning_rules.use_meta_optimizer = False
    config.mechanic_discovery.use_mde = False
    config.autonomous_specialization.use_autonomous_specialization = False
    config.stability_in_the_dark.use_stability_in_the_dark = False
    
    # Keep these enabled
    config.curiosity.use_curiosity = True
    config.curiosity.state_embedding_dim = 32
    config.attention_maps.use_attention_maps = True
    config.attention_maps.num_heads = 4
    config.subgoal.use_subgoals = True
    config.subgoal.goal_embedding_dim = 32
    config.meta_learning.use_meta_learning = True
    config.meta_learning.context_dim = 32
    config.feature_validation.use_fve = True
    config.srp.use_srp = True
    config.governance.use_governor = True
    config.meta_loop.use_meta_loop = True
    config.formal_verification.use_formal_verification = True
    config.dynamic_hyperparams.use_dynamic_hyperparams = True
    config.reasoning.use_reasoning = True
    config.reasoning.num_reasoning_steps = 2
    config.reasoning.num_counterfactual_actions = 2
    config.expert_compiler.use_fec = True
    
    print(f"  latent_dim={config.encoder.latent_dim}, experts={config.router.num_experts}")
    
    # Create agent
    print_subheader("Creating Agent")
    try:
        agent = DeepThoughtAgent(config).to(device)
        total_params = sum(p.numel() for p in agent.parameters())
        print(f"  Total parameters: {total_params:,}")
        print(f"  Agent device: {next(agent.parameters()).device}")
    except Exception as e:
        print(f"  AGENT CREATION FAILED: {e}")
        traceback.print_exc()
        return
    
    # Device check
    device_issues = check_device_consistency(agent, device)
    
    # Forward pass test
    print_subheader("Forward Pass Test")
    try:
        agent.train()
        agent.reset(1)
        obs = torch.randn(1, obs_dim, device=device)
        outputs = agent.forward(obs, training=True)
        print(f"  Output keys: {sorted(outputs.keys())}")
        
        # Second pass with action (tests world model path)
        action = torch.tensor([1], device=device)
        outputs2 = agent.forward(obs, action=action, reward=1.0, done=False, training=True)
        print(f"  World model present: {'world_model' in outputs2}")
        
        policy_logits = outputs["policy_logits"]
        probs = torch.softmax(policy_logits, dim=-1)
        print(f"  Policy probs: {probs.detach().cpu().numpy()}")
        print(f"  Value: {outputs['value'].item():.4f}")
        print(f"  Forward pass OK ✓")
    except Exception as e:
        print(f"  Forward pass FAILED: {e}")
        traceback.print_exc()
        return
    
    # Memory system test
    print_subheader("Memory System Test")
    try:
        latent = torch.randn(1, config.encoder.latent_dim, device=device)
        obs_flat = torch.randn(obs_dim, device=device)
        action_flat = torch.tensor(1.0, device=device)
        
        for i in range(5):
            agent.memory.episodic.write(
                latent=latent, observation=obs_flat, action=action_flat,
                reward=float(i)/4, done=(i==4),
                prediction_error=float(i)*0.1, novelty=float(i)*0.2
            )
        
        ep_size = agent.memory.episodic.get_size()
        print(f"  Episodic entries: {ep_size}")
        
        if ep_size > 0:
            query = torch.randn(1, config.encoder.latent_dim, device=device)
            read_result, entries = agent.memory.episodic.read(query, k=3)
            print(f"  Episodic read: shape={read_result.shape}, device={read_result.device}")
            assert read_result.device.type == device.type
            print(f"  Episodic read device OK ✓")
        
        for i in range(3):
            agent.memory.semantic.write(latent, obs_flat.unsqueeze(0) if obs_flat.dim() == 1 else obs_flat)
        
        sem_size = agent.memory.semantic.get_size()
        print(f"  Semantic concepts: {sem_size}")
        if sem_size > 0:
            query = torch.randn(1, config.encoder.latent_dim, device=device)
            sem_read, concepts = agent.memory.semantic.read(query, k=2)
            print(f"  Semantic read: shape={sem_read.shape}, device={sem_read.device}")
            assert sem_read.device.type == device.type
            print(f"  Semantic read device OK ✓")
    except Exception as e:
        print(f"  Memory test FAILED: {e}")
        traceback.print_exc()
    
    # Gradient flow test
    print_subheader("Gradient Flow Test")
    try:
        optimizer = torch.optim.Adam(agent.parameters(), lr=config.training.learning_rate)
        obs_test = torch.randn(2, obs_dim, device=device)
        agent.train()
        agent.reset(2)
        out = agent.forward(obs_test, training=True)
        fake_loss = out["policy_logits"].sum() + out["value"].sum()
        optimizer.zero_grad()
        fake_loss.backward()
        no_grad_count = check_gradient_flow(agent)
        
        critical = ['encoder', 'router', 'expert_bank', 'policy_head', 'critic_head']
        for mod_name in critical:
            module = getattr(agent, mod_name, None)
            if module is not None:
                has_grad = any(p.grad is not None and p.grad.norm() > 0 for p in module.parameters())
                print(f"  {mod_name}: {'✓' if has_grad else '✗ NO GRAD'}")
    except Exception as e:
        print(f"  Gradient test FAILED: {e}")
        traceback.print_exc()
        no_grad_count = 999
    
    # ================================================================
    # TRAINING LOOP
    # ================================================================
    print_header("Training Loop")
    
    ppo_trainer = PPOTrainer(
        config.training, agent,
        action_space=config.action_space,
        action_dim=config.action_dim
    )
    scheduler = CosineAnnealingWarmupScheduler(optimizer, warmup_steps=50, total_steps=steps)
    
    episode_rewards = []
    episode_lengths = []
    loss_history = []
    
    observation, _ = env.reset()
    observation = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)
    agent.reset(1)
    
    episode_reward = 0.0
    episode_length = 0
    start_time = time.time()
    
    print(f"\n  Running {steps} steps on {device}...\n")
    print(f"  {'Step':>6} | {'Reward':>8} | {'AvgRwd':>8} | {'Loss':>10} | {'PolLoss':>10} | {'ValLoss':>10} | {'Entr':>8} | {'KL':>8} | {'Eps':>5} | {'Exps':>4}")
    print(f"  {'-'*90}")
    
    for step in range(steps):
        rollout_reward, rollout_length, episode_done, last_obs = ppo_trainer.collect_rollout(
            env, observation, agent.h_t, agent.m_t, device
        )
        
        episode_reward += rollout_reward
        episode_length += rollout_length
        
        if not episode_done and len(ppo_trainer.buffer) > 0:
            with torch.no_grad():
                bootstrap_latent, _ = agent.encoder(last_obs)
                bootstrap_value = agent.critic_head(bootstrap_latent).item()
        else:
            bootstrap_value = 0.0
        
        metrics = ppo_trainer.update(optimizer, bootstrap_value=bootstrap_value)
        scheduler.step()
        
        total_loss_val = metrics.get("total_loss", 0.0)
        policy_loss_val = metrics.get("policy_loss", 0.0)
        value_loss_val = metrics.get("value_loss", 0.0)
        entropy_val = metrics.get("epoch_0_entropy", 0.0)
        kl_val = metrics.get("epoch_0_kl", 0.0)
        loss_history.append(total_loss_val)
        
        agent.prune_experts()
        agent.grow_experts()
        agent.consolidate_memory()
        agent.validate_features()
        agent.update_srp(rollout_reward, total_loss_val)
        
        if episode_done:
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            episode_reward = 0.0
            episode_length = 0
            observation, _ = env.reset()
            observation = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)
            agent.reset(1)
        else:
            observation = last_obs.detach()
        
        if step % 10 == 0 or step == steps - 1:
            avg_reward = np.mean(episode_rewards[-10:]) if episode_rewards else 0.0
            n_episodes = len(episode_rewards)
            n_experts = agent.get_stats()['num_experts']
            print(f"  {step:>6} | {rollout_reward:>8.2f} | {avg_reward:>8.2f} | {total_loss_val:>10.4f} | {policy_loss_val:>10.4f} | {value_loss_val:>10.4f} | {entropy_val:>8.4f} | {kl_val:>8.4f} | {n_episodes:>5} | {n_experts:>4}")
    
    elapsed = time.time() - start_time
    
    # ================================================================
    # RESULTS
    # ================================================================
    print_header("Training Results")
    
    print(f"  Steps: {steps}, Time: {elapsed:.1f}s ({steps/elapsed:.1f} steps/sec)")
    print(f"  Episodes: {len(episode_rewards)}")
    
    if episode_rewards:
        print(f"  First 5 rewards: {[f'{r:.1f}' for r in episode_rewards[:5]]}")
        print(f"  Last 5 rewards:  {[f'{r:.1f}' for r in episode_rewards[-5:]]}")
        print(f"  Mean reward: {np.mean(episode_rewards):.2f}")
        print(f"  Max reward:  {np.max(episode_rewards):.2f}")
        
        if len(episode_rewards) >= 4:
            first_q = np.mean(episode_rewards[:len(episode_rewards)//4])
            last_q = np.mean(episode_rewards[-(len(episode_rewards)//4):])
            improvement = last_q - first_q
            print(f"  First quarter avg: {first_q:.2f}")
            print(f"  Last quarter avg:  {last_q:.2f}")
            print(f"  Improvement: {improvement:+.2f} ({'LEARNING ✓' if improvement > 0 else 'NOT LEARNING ✗'})")
    
    if loss_history:
        print(f"  First 5 losses: {[f'{l:.4f}' for l in loss_history[:5]]}")
        print(f"  Last 5 losses:  {[f'{l:.4f}' for l in loss_history[-5:]]}")
    
    # Final checks
    print_subheader("Final Checks")
    device_issues_final = check_device_consistency(agent, device)
    
    agent.train()
    agent.reset(1)
    obs_test = torch.randn(1, obs_dim, device=device)
    out = agent.forward(obs_test, training=True)
    fake_loss = out["policy_logits"].sum() + out["value"].sum()
    optimizer.zero_grad()
    fake_loss.backward()
    check_gradient_flow(agent)
    
    print(f"\n  Episodic entries: {agent.memory.episodic.get_size()}")
    print(f"  Semantic concepts: {agent.memory.semantic.get_size()}")
    
    stats = agent.get_stats()
    for k, v in sorted(stats.items()):
        print(f"  {k}: {v}")
    
    print_header("FINAL ASSESSMENT")
    issues = []
    if device_issues > 0 or device_issues_final > 0:
        issues.append(f"Device issues: {device_issues + device_issues_final}")
    if episode_rewards and len(episode_rewards) >= 4:
        first_q = np.mean(episode_rewards[:len(episode_rewards)//4])
        last_q = np.mean(episode_rewards[-(len(episode_rewards)//4):])
        if last_q <= first_q:
            issues.append("No reward improvement")
    if loss_history and all(l == 0.0 for l in loss_history):
        issues.append("Loss always zero")
    
    if not issues:
        print("\n  ✓ ALL CHECKS PASSED")
        print("  Note: GPU-specific issues only verifiable on CUDA machine.")
    else:
        print("\n  ✗ ISSUES FOUND:")
        for i in issues:
            print(f"    - {i}")
    
    env.close()

if __name__ == "__main__":
    run_verbose_training(steps=500, env_id="CartPole-v1")
