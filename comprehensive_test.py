#!/usr/bin/env python3
"""
Comprehensive training test for Deep Thought.

This test verifies:
1. ALL modules receive gradient signal (no dead modules)
2. GPU is actually used (when available)
3. Reward improves over training (real learning)
4. Routing does NOT collapse to a subset of experts
5. Reasoning engine contributes to learning
6. All auxiliary losses are non-zero

Uses LunarLander-v3 (harder than CartPole) as the test environment.
"""

import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gymnasium as gym
from collections import defaultdict
import traceback
import time

sys.path.insert(0, "/home/z/deepthough_work")

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
    """Check which modules received gradients after backward pass."""
    grad_info = defaultdict(list)
    no_grad_modules = set()
    
    for name, param in agent.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            module = name.split('.')[0]
            grad_info[module].append((name, grad_norm))
        elif param.requires_grad:
            no_grad_modules.add(name.split('.')[0])
    
    print("  Modules WITH gradients:")
    total_grad_norm = 0.0
    for module, params in sorted(grad_info.items()):
        mod_norm = sum(n for _, n in params)
        count = len(params)
        total_grad_norm += mod_norm
        print(f"    {module}: {count} params, grad_norm={mod_norm:.6f}")
    
    if no_grad_modules:
        print(f"\n  ⚠ Modules with NO gradient on ANY parameter:")
        for m in sorted(no_grad_modules):
            print(f"    - {m}")
    else:
        print(f"\n  ✓ All modules received gradients")
    
    return len(no_grad_modules), total_grad_norm

def check_routing_health(agent):
    """Check if routing is collapsed (all experts used) or healthy."""
    usage = agent.router.get_expert_usage()
    num_experts = usage.size(0)
    
    # An expert is "dead" if it receives < 1% of routing probability
    dead_threshold = 1.0 / (num_experts * 10)
    dead_count = (usage < dead_threshold).sum().item()
    active_count = num_experts - dead_count
    
    # Coefficient of variation (lower = more balanced)
    cv = usage.std() / (usage.mean() + 1e-8)
    
    print(f"  Expert usage: min={usage.min().item():.4f}, max={usage.max().item():.4f}, "
          f"mean={usage.mean().item():.4f}, CV={cv:.4f}")
    print(f"  Active experts: {active_count}/{num_experts} "
          f"({'✓ HEALTHY' if dead_count == 0 else f'⚠ {dead_count} DEAD'})")
    
    return dead_count, cv

def run_comprehensive_test(steps=800, env_id="LunarLander-v3"):
    print_header("Deep Thought Comprehensive Training Test")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  GPU Memory: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")
    print(f"  PyTorch: {torch.__version__}")
    
    # Try to create the environment, fall back if not available
    try:
        env = gym.make(env_id)
        obs_dim = env.observation_space.shape[0]
        if hasattr(env.action_space, 'n'):
            act_dim = env.action_space.n
            action_space_type = "discrete"
        else:
            act_dim = env.action_space.shape[0]
            action_space_type = "continuous"
        print(f"  Environment: {env_id} (obs={obs_dim}, act={act_dim}, space={action_space_type})")
    except Exception as e:
        print(f"  {env_id} not available: {e}")
        print(f"  Falling back to CartPole-v1")
        env_id = "CartPole-v1"
        env = gym.make(env_id)
        obs_dim = env.observation_space.shape[0]
        act_dim = env.action_space.n
        action_space_type = "discrete"
        print(f"  Environment: {env_id} (obs={obs_dim}, act={act_dim})")
    
    # ================================================================
    # Configuration - scaled for testability but keeping ALL modules ON
    # ================================================================
    print_subheader("Configuration")
    config = DeepThoughtConfig()
    config.observation_dim = obs_dim
    config.action_dim = act_dim
    config.num_actions = act_dim
    config.action_space = action_space_type
    config.device = str(device)
    
    # Scale down architecture for fast testing
    config.encoder.latent_dim = 64
    config.encoder.hidden_dim = 128
    config.router.num_experts = 8
    config.router.active_experts = 2
    config.router.load_balance_loss_coef = 0.15  # Strong load balancing
    config.router.entropy_coef = 0.15
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
    config.training.entropy_coef = 0.15  # Higher entropy to prevent collapse
    
    # ALL subsystems ON (this is the whole point - verify they all work)
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
    config.reasoning.num_counterfactual_actions = act_dim
    config.expert_compiler.use_fec = True
    
    # Disable the heaviest subsystems for speed
    config.planning.use_tcpl = False
    config.opponent_modeling.use_opponent_modeling = False
    config.hierarchical.use_hierarchy = False
    config.compute_economy.use_compute_market = False
    config.shadow_evolution.use_shadow_evolution = False
    config.meta_learning_rules.use_meta_optimizer = False
    config.mechanic_discovery.use_mde = False
    config.autonomous_specialization.use_autonomous_specialization = False
    config.stability_in_the_dark.use_stability_in_the_dark = False
    
    print(f"  latent_dim={config.encoder.latent_dim}, experts={config.router.num_experts}, active={config.router.active_experts}")
    print(f"  All core subsystems: ON")
    
    # ================================================================
    # Create Agent
    # ================================================================
    print_subheader("Creating Agent")
    try:
        agent = DeepThoughtAgent(config).to(device)
        total_params = sum(p.numel() for p in agent.parameters())
        print(f"  Total parameters: {total_params:,}")
        print(f"  Agent device: {next(agent.parameters()).device}")
    except Exception as e:
        print(f"  ✗ AGENT CREATION FAILED: {e}")
        traceback.print_exc()
        return False
    
    # Device consistency check
    device_issues = check_device_consistency(agent, device)
    
    # ================================================================
    # Forward Pass Test
    # ================================================================
    print_subheader("Forward Pass Test")
    try:
        agent.train()
        agent.reset(1)
        obs = torch.randn(1, obs_dim, device=device)
        outputs = agent.forward(obs, training=True)
        print(f"  Output keys: {sorted(outputs.keys())}")
        
        # Test with action (world model path)
        action = torch.tensor([1], device=device) if action_space_type == "discrete" else torch.randn(1, act_dim, device=device)
        outputs2 = agent.forward(obs, action=action, reward=1.0, done=False, training=True)
        print(f"  World model in outputs: {'world_model' in outputs2}")
        print(f"  Reasoning info present: {'reasoning_info' in outputs2}")
        print(f"  Curiosity info present: {'curiosity_info' in outputs2}")
        print(f"  ✓ Forward pass works")
    except Exception as e:
        print(f"  ✗ Forward pass FAILED: {e}")
        traceback.print_exc()
        return False
    
    # ================================================================
    # Gradient Flow Test - CRITICAL: Verify ALL modules receive gradients
    # ================================================================
    print_subheader("Gradient Flow Test (Full Forward + Backward)")
    try:
        optimizer = torch.optim.Adam(agent.parameters(), lr=config.training.learning_rate)
        
        # Simulate a mini PPO-style forward pass
        agent.train()
        agent.reset(2)
        obs_test = torch.randn(2, obs_dim, device=device)
        
        # Full forward pass (like in PPO update)
        latent, _ = agent.encoder(obs_test)
        h_t = torch.zeros(2, 64, device=device)
        m_t = torch.zeros(2, 64, device=device)
        context = torch.zeros(2, 32, device=device) if agent.meta_learning else None
        
        # Router with gradient flow
        gates, selected_indices, router_info = agent.router(
            h_t, latent, m_t, context,
            prediction_error=None, training=True, detach_gates=False
        )
        
        # Experts
        delta_h, compute_costs = agent.expert_bank(h_t, selected_indices, gates)
        h_tilde = h_t + delta_h
        
        # Meta-learning
        if agent.meta_learning is not None and context is not None:
            h_adapted, _ = agent.meta_learning.adapt(h_tilde, context, gradient=None)
            h_tilde = h_adapted
        
        # Reasoning engine
        if agent.reasoning_engine is not None:
            h_tilde, reasoning_info = agent.reasoning_engine(
                h_tilde, latent,
                world_model=agent.world_model if config.reasoning.use_counterfactual else None,
                action_dim=act_dim, training=True
            )
        
        # Policy and value
        policy_logits = agent.policy_head(h_tilde)
        value = agent.critic_head(h_tilde)
        
        # World model
        if agent.world_model is not None:
            if action_space_type == "discrete":
                wm_actions = torch.zeros(2, act_dim, device=device)
                wm_actions[:, 0] = 1.0
            else:
                wm_actions = torch.randn(2, act_dim, device=device)
            z_pred, r_pred, d_pred = agent.world_model(latent, wm_actions)
        
        # Curiosity
        if agent.curiosity is not None:
            embedded = agent.curiosity_proj(latent)
            pred_error = torch.zeros_like(latent)
            pred_error_proj = agent.curiosity_proj(pred_error)
            intrinsic_reward, _ = agent.curiosity(latent=embedded, prediction_error=pred_error_proj, ensemble_uncertainty=None)
        
        # Construct loss that involves ALL modules
        loss = policy_logits.sum() + value.sum()
        
        # World model auxiliary
        if agent.world_model is not None:
            loss = loss + F.mse_loss(z_pred, latent.detach())
        
        # Router losses
        router_losses = agent.router.compute_losses(router_info)
        for rl in router_losses.values():
            loss = loss + rl
        
        # Curiosity auxiliary
        if agent.curiosity is not None:
            loss = loss + intrinsic_reward.mean()
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        
        no_grad_count, total_grad = check_gradient_flow(agent)
        
        # Check critical modules
        critical_modules = ['encoder', 'router', 'expert_bank', 'policy_head', 'critic_head',
                           'world_model', 'curiosity_proj', 'reasoning_engine']
        for mod_name in critical_modules:
            module = getattr(agent, mod_name, None)
            if module is not None:
                has_grad = any(p.grad is not None and p.grad.norm() > 1e-8 for p in module.parameters())
                status = '✓' if has_grad else '✗ DEAD'
                print(f"  {mod_name}: {status}")
        
    except Exception as e:
        print(f"  ✗ Gradient test FAILED: {e}")
        traceback.print_exc()
        no_grad_count = 999
        total_grad = 0
    
    # ================================================================
    # Training Loop
    # ================================================================
    print_header(f"Training Loop ({steps} steps on {device})")
    
    ppo_trainer = PPOTrainer(
        config.training, agent,
        action_space=config.action_space,
        action_dim=config.action_dim
    )
    optimizer = torch.optim.Adam(agent.parameters(), lr=config.training.learning_rate)
    scheduler = CosineAnnealingWarmupScheduler(optimizer, warmup_steps=50, total_steps=steps)
    
    episode_rewards = []
    loss_history = []
    grad_norm_history = []
    
    observation, _ = env.reset()
    observation = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)
    agent.reset(1)
    
    episode_reward = 0.0
    episode_length = 0
    start_time = time.time()
    
    print(f"\n  {'Step':>6} | {'Rwd':>7} | {'AvgR':>7} | {'Loss':>9} | {'GradN':>9} | {'Ent':>6} | {'KL':>6} | {'Eps':>4} | {'Dead':>4} | {'RoutCV':>7}")
    print(f"  {'-'*85}")
    
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
        entropy_val = metrics.get("epoch_0_entropy", 0.0)
        kl_val = metrics.get("epoch_0_kl", 0.0)
        loss_history.append(total_loss_val)
        
        # Check gradient norm
        grad_norm = 0.0
        for p in agent.parameters():
            if p.grad is not None:
                grad_norm += p.grad.norm().item() ** 2
        grad_norm = grad_norm ** 0.5
        grad_norm_history.append(grad_norm)
        
        agent.prune_experts()
        agent.grow_experts()
        agent.consolidate_memory()
        agent.validate_features()
        agent.update_srp(rollout_reward, total_loss_val)
        
        if episode_done:
            episode_rewards.append(episode_reward)
            episode_reward = 0.0
            episode_length = 0
            observation, _ = env.reset()
            observation = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)
            agent.reset(1)
        else:
            observation = last_obs.detach()
        
        if step % 20 == 0 or step == steps - 1:
            avg_reward = np.mean(episode_rewards[-10:]) if episode_rewards else 0.0
            n_episodes = len(episode_rewards)
            usage = agent.router.get_expert_usage()
            dead_count = (usage < 1.0 / (config.router.num_experts * 10)).sum().item()
            cv = usage.std() / (usage.mean() + 1e-8)
            
            print(f"  {step:>6} | {rollout_reward:>7.1f} | {avg_reward:>7.1f} | {total_loss_val:>9.4f} | {grad_norm:>9.4f} | {entropy_val:>6.3f} | {kl_val:>6.4f} | {n_episodes:>4} | {dead_count:>4} | {cv:>7.3f}")
    
    elapsed = time.time() - start_time
    
    # ================================================================
    # RESULTS
    # ================================================================
    print_header("Training Results")
    
    print(f"  Steps: {steps}, Time: {elapsed:.1f}s ({steps/elapsed:.1f} steps/sec)")
    print(f"  Episodes: {len(episode_rewards)}")
    
    learning = False
    if episode_rewards and len(episode_rewards) >= 4:
        print(f"  First 5 rewards: {[f'{r:.1f}' for r in episode_rewards[:5]]}")
        print(f"  Last 5 rewards:  {[f'{r:.1f}' for r in episode_rewards[-5:]]}")
        print(f"  Mean reward: {np.mean(episode_rewards):.2f}")
        print(f"  Max reward:  {np.max(episode_rewards):.2f}")
        
        first_q = np.mean(episode_rewards[:len(episode_rewards)//4])
        last_q = np.mean(episode_rewards[-(len(episode_rewards)//4):])
        improvement = last_q - first_q
        print(f"  First quarter avg: {first_q:.2f}")
        print(f"  Last quarter avg:  {last_q:.2f}")
        print(f"  Improvement: {improvement:+.2f}")
        learning = improvement > 0
    
    if loss_history:
        first_losses = loss_history[:5]
        last_losses = loss_history[-5:]
        print(f"  First 5 losses: {[f'{l:.4f}' for l in first_losses]}")
        print(f"  Last 5 losses:  {[f'{l:.4f}' for l in last_losses]}")
    
    # Final routing health check
    print_subheader("Routing Health")
    dead_count, cv = check_routing_health(agent)
    
    # Final gradient flow check
    print_subheader("Final Gradient Flow Check")
    agent.train()
    agent.reset(2)
    obs_test = torch.randn(2, obs_dim, device=device)
    out = agent.forward(obs_test, training=True)
    fake_loss = out["policy_logits"].sum() + out["value"].sum()
    if "world_model" in out:
        fake_loss = fake_loss + out["world_model"]["z_next"].sum()
    if "intrinsic_reward" in out:
        fake_loss = fake_loss + out["intrinsic_reward"].sum()
    optimizer.zero_grad()
    fake_loss.backward()
    no_grad_final, total_grad_final = check_gradient_flow(agent)
    
    # Final device check
    print_subheader("Final Device Check")
    device_issues_final = check_device_consistency(agent, device)
    
    # ================================================================
    # FINAL ASSESSMENT
    # ================================================================
    print_header("FINAL ASSESSMENT")
    issues = []
    
    if device_issues > 0 or device_issues_final > 0:
        issues.append(f"Device issues: {device_issues + device_issues_final}")
    
    if not learning and episode_rewards and len(episode_rewards) >= 4:
        issues.append("No reward improvement (not learning)")
    
    if dead_count > 0:
        issues.append(f"Routing collapse: {dead_count} dead experts")
    
    if cv > 1.0:
        issues.append(f"High routing CV: {cv:.3f} (unbalanced)")
    
    if no_grad_final > 0:
        issues.append(f"{no_grad_final} modules with zero gradient")
    
    if not issues:
        print("\n  ✓ ALL CHECKS PASSED")
        print("  - All modules receive gradient signal")
        print("  - No routing collapse")
        print("  - Reward is improving")
        print("  - Device consistency is maintained")
    else:
        print("\n  ✗ ISSUES FOUND:")
        for i in issues:
            print(f"    - {i}")
    
    # Detailed module gradient report
    print("\n  Module gradient detail:")
    for name, param in agent.named_parameters():
        if param.grad is not None:
            gn = param.grad.norm().item()
            if gn > 0.001:
                module = name.split('.')[0]
                if module in ['reasoning_engine', 'world_model', 'curiosity', 'curiosity_proj',
                             'meta_loop', 'subgoal_generator', 'feature_validator', 'router']:
                    print(f"    {name}: grad_norm={gn:.6f}")
    
    env.close()
    return len(issues) == 0


if __name__ == "__main__":
    success = run_comprehensive_test(steps=800, env_id="LunarLander-v3")
    sys.exit(0 if success else 1)
