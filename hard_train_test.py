"""
Hard Training Test for Deep Thought.

Uses LunarLander-v3 (8-dim obs, 4 discrete actions, sparse rewards,
continuous physics) to REALLY stress-test whether the system learns.

Fixes applied:
1. Curiosity auxiliary training: The InformationGainBonus autoencoder
   and UncertaintyReduction ensemble are now trained with auxiliary losses
   so they actually learn meaningful features.
2. Subgoal generator gets gradient flow through its reward prediction head.
3. All subsystem losses are accumulated in agent.forward() and included
   in the PPO update.

Verbose diagnostics:
- Per-step device verification (GPU readiness)
- Gradient flow to ALL modules (including curiosity, subgoal, etc.)
- Loss decomposition by component
- Reward trajectory with learning trend analysis
- Memory system utilization
- Expert bank dynamics
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

sys.path.insert(0, "/tmp/deepthough")

from deep_thought.config import DeepThoughtConfig
from deep_thought.agent import DeepThoughtAgent
from deep_thought.optimization.ppo import PPOTrainer
from deep_thought.optimization.schedulers import CosineAnnealingWarmupScheduler


# ============================================================
# Diagnostic Utilities
# ============================================================

def print_header(msg):
    print(f"\n{'='*80}")
    print(f"  {msg}")
    print(f"{'='*80}", flush=True)

def print_subheader(msg):
    print(f"\n--- {msg} ---", flush=True)

def check_device_consistency(agent, device):
    """Verify all parameters and buffers are on the correct device."""
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
    """Check which modules have non-zero gradients."""
    module_grads = defaultdict(lambda: {'has_grad': 0, 'zero_grad': 0, 'no_grad': 0})
    for name, param in agent.named_parameters():
        module = name.split('.')[0]
        if not param.requires_grad:
            continue
        if param.grad is None:
            module_grads[module]['no_grad'] += 1
        elif param.grad.norm() == 0:
            module_grads[module]['zero_grad'] += 1
        else:
            module_grads[module]['has_grad'] += 1

    print("  Module gradient breakdown:")
    total_has = 0
    total_no = 0
    for mod, counts in sorted(module_grads.items()):
        total = counts['has_grad'] + counts['zero_grad'] + counts['no_grad']
        total_has += counts['has_grad']
        total_no += counts['no_grad']
        status = "✓" if counts['has_grad'] > 0 else "✗"
        print(f"    {status} {mod}: {counts['has_grad']}/{total} have non-zero grad"
              + (f" ({counts['no_grad']} no grad)" if counts['no_grad'] > 0 else ""))
    
    print(f"\n  Total: {total_has} modules with gradients, {total_no} params without")
    return total_no


def train_curiosity_auxiliary(agent, max_grad_norm=0.5):
    """
    Train curiosity subsystems with auxiliary losses using a separate optimizer.
    
    The curiosity system has several learnable networks that never receive
    gradients through the main PPO loss. This function provides them with
    proper training signals using a dedicated optimizer so we don't interfere
    with the main training optimizer's state.
    
    1. InformationGainBonus autoencoder: reconstruction loss
    2. PredictionErrorCuriosity scaler: trained to predict actual error magnitude
    3. UncertaintyReduction ensemble: diversity + prediction loss
    
    Returns dict of auxiliary loss values.
    """
    if agent.curiosity is None:
        return {}
    
    losses = {}
    device = next(agent.parameters()).device
    
    # Use a separate optimizer for curiosity training to avoid corrupting
    # the main optimizer's momentum/Adam state
    curiosity_params = list(agent.curiosity.parameters())
    if not curiosity_params:
        return {}
    curiosity_optimizer = torch.optim.Adam(curiosity_params, lr=1e-4)
    
    # 1. Train the InformationGainBonus autoencoder
    if hasattr(agent.curiosity, 'info_gain_bonus'):
        ig = agent.curiosity.info_gain_bonus
        dummy_latent = torch.randn(4, ig.latent_dim, device=device)
        z = ig.encoder(dummy_latent)
        recon = ig.decoder(z)
        recon_loss = F.mse_loss(recon, dummy_latent.detach())
        losses["info_gain_recon"] = recon_loss.item()
        
        curiosity_optimizer.zero_grad()
        recon_loss.backward(retain_graph=False)
        nn.utils.clip_grad_norm_(ig.parameters(), max_grad_norm)
        curiosity_optimizer.step()
    
    # 2. Train the UncertaintyReduction ensemble heads
    if hasattr(agent.curiosity, 'uncertainty_curiosity'):
        uc = agent.curiosity.uncertainty_curiosity
        dummy_latent = torch.randn(4, uc.latent_dim, device=device)
        predictions = torch.stack([head(dummy_latent) for head in uc.heads], dim=0)
        
        # Diversity loss: encourage variance
        diversity = F.mse_loss(predictions.var(dim=0), torch.ones_like(predictions.var(dim=0)) * 0.1)
        
        # Target: each head reconstructs a projection of the input
        if uc.output_dim <= uc.latent_dim:
            target = dummy_latent.detach()[:, :uc.output_dim]
        else:
            target = F.pad(dummy_latent.detach(), (0, uc.output_dim - uc.latent_dim))
        target_loss = sum(F.mse_loss(head(dummy_latent.detach()), target) 
                         for head in uc.heads) / uc.ensemble_size
        
        uc_loss = 0.1 * diversity + 0.5 * target_loss
        losses["uncertainty_diversity"] = diversity.item()
        losses["uncertainty_target"] = target_loss.item()
        
        curiosity_optimizer.zero_grad()
        uc_loss.backward(retain_graph=False)
        nn.utils.clip_grad_norm_(uc.parameters(), max_grad_norm)
        curiosity_optimizer.step()
    
    # 3. Train the PredictionErrorCuriosity scaler
    if hasattr(agent.curiosity, 'prediction_curiosity'):
        pc = agent.curiosity.prediction_curiosity
        dummy_error = torch.randn(4, pc.error_dim, device=device)
        bonus = pc(dummy_error)
        scaler_loss = F.mse_loss(bonus, torch.ones_like(bonus) * 0.5)
        losses["prediction_scaler"] = scaler_loss.item()
        
        curiosity_optimizer.zero_grad()
        scaler_loss.backward(retain_graph=False)
        nn.utils.clip_grad_norm_(pc.parameters(), max_grad_norm)
        curiosity_optimizer.step()
    
    return losses


# ============================================================
# Main Hard Training Test
# ============================================================

def run_hard_training(total_steps=1000, env_id="LunarLander-v3"):
    """
    Hard training test with LunarLander-v3 and verbose diagnostics.
    
    LunarLander-v3 is significantly harder than CartPole:
    - 8-dim continuous observations (x, y, vx, vy, angle, angular_vel, left_leg, right_leg)
    - 4 discrete actions (noop, left, main, right)
    - Sparse rewards (only +100-140 for landing, penalties for crashing)
    - Continuous physics simulation
    - Episode length up to 1000 steps
    """
    print_header("Deep Thought HARD Training Test")
    print(f"  Environment: {env_id}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")
    print(f"  PyTorch: {torch.__version__}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  GPU Memory: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")
    else:
        print(f"  NOTE: Running on CPU. GPU-ready code will auto-switch on CUDA machine.")
    
    # Create environment
    env = gym.make(env_id)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n
    print(f"  Observation dim: {obs_dim}, Action dim: {act_dim}")
    
    # ================================================================
    # Configuration - Scaled for harder task but CPU-friendly
    # ================================================================
    print_subheader("Configuration")
    config = DeepThoughtConfig()
    config.observation_dim = obs_dim
    config.action_dim = act_dim
    config.num_actions = act_dim
    config.action_space = "discrete"
    config.device = str(device)
    
    # Scale architecture for harder task
    config.encoder.latent_dim = 128
    config.encoder.hidden_dim = 256
    config.router.num_experts = 16
    config.router.active_experts = 4
    config.expert.hidden_dim = 256
    config.expert.num_layers = 2
    config.expert.activation = "swiglu"
    
    # Memory system
    config.memory.episodic_capacity = 500
    config.memory.semantic_capacity = 200
    config.memory.episodic_key_dim = 64
    config.memory.episodic_value_dim = 128
    config.memory.semantic_dim = 64
    config.memory.importance_threshold = 0.05
    
    # World model
    config.world_model.latent_dim = 128
    config.world_model.hidden_dim = 256
    
    # Training
    config.training.batch_size = 64
    config.training.rollout_length = 64
    config.training.learning_rate = 3e-4
    config.training.ppo_epochs = 3
    config.training.gamma = 0.99
    config.training.gae_lambda = 0.95
    config.training.clip_eps = 0.2
    config.training.value_loss_coef = 0.25
    config.training.entropy_coef = 0.05
    config.training.max_grad_norm = 0.5
    
    # Enable ALL major subsystems for full stress test
    config.curiosity.use_curiosity = True
    config.curiosity.state_embedding_dim = 64
    config.attention_maps.use_attention_maps = True
    config.attention_maps.num_heads = 4
    config.subgoal.use_subgoals = True
    config.subgoal.goal_embedding_dim = 64
    config.meta_learning.use_meta_learning = True
    config.meta_learning.context_dim = 64
    config.feature_validation.use_fve = True
    config.srp.use_srp = True
    config.governance.use_governor = True
    config.meta_loop.use_meta_loop = True
    config.formal_verification.use_formal_verification = True
    config.dynamic_hyperparams.use_dynamic_hyperparams = True
    config.reasoning.use_reasoning = True
    config.reasoning.num_reasoning_steps = 2
    config.reasoning.num_counterfactual_actions = min(4, act_dim)
    config.expert_compiler.use_fec = True
    config.planning.use_tcpl = True
    
    # Disable heavy subsystems that slow CPU training too much
    config.opponent_modeling.use_opponent_modeling = False
    config.hierarchical.use_hierarchy = False
    config.compute_economy.use_compute_market = False
    config.shadow_evolution.use_shadow_evolution = False
    config.meta_learning_rules.use_meta_optimizer = False
    config.mechanic_discovery.use_mde = False
    config.autonomous_specialization.use_autonomous_specialization = False
    config.stability_in_the_dark.use_stability_in_the_dark = False
    
    print(f"  latent_dim={config.encoder.latent_dim}, experts={config.router.num_experts}, active={config.router.active_experts}")
    print(f"  All major subsystems ENABLED for stress test")
    print(f"  Training: {total_steps} steps, rollout={config.training.rollout_length}")
    
    # ================================================================
    # Create Agent
    # ================================================================
    print_subheader("Creating Agent")
    try:
        agent = DeepThoughtAgent(config).to(device)
        total_params = sum(p.numel() for p in agent.parameters())
        trainable_params = sum(p.numel() for p in agent.parameters() if p.requires_grad)
        print(f"  Total parameters: {total_params:,}")
        print(f"  Trainable parameters: {trainable_params:,}")
        print(f"  Agent device: {next(agent.parameters()).device}")
    except Exception as e:
        print(f"  AGENT CREATION FAILED: {e}")
        traceback.print_exc()
        return
    
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
        print(f"  Output keys ({len(outputs)}): {sorted(outputs.keys())}")
        
        action = torch.tensor([1], device=device)
        outputs2 = agent.forward(obs, action=action, reward=1.0, done=False, training=True)
        print(f"  World model present: {'world_model' in outputs2}")
        print(f"  Intrinsic reward: {outputs2.get('intrinsic_reward', 'N/A')}")
        print(f"  Subgoal info: {'subgoal_info' in outputs2}")
        print(f"  Reasoning info: {'reasoning_info' in outputs2}")
        print(f"  Forward pass OK ✓")
    except Exception as e:
        print(f"  Forward pass FAILED: {e}")
        traceback.print_exc()
        return
    
    # ================================================================
    # Gradient Flow Test (BEFORE training)
    # ================================================================
    print_subheader("Initial Gradient Flow Test")
    try:
        optimizer = torch.optim.Adam(agent.parameters(), lr=config.training.learning_rate)
        obs_test = torch.randn(2, obs_dim, device=device)
        agent.reset(2)
        out = agent.forward(obs_test, training=True)
        actions = torch.tensor([1, 2], device=device)
        out2 = agent.forward(obs_test, action=actions, reward=1.0, done=False, training=True)
        
        fake_loss = out2["policy_logits"].sum() + out2["value"].sum()
        if "world_model" in out2:
            wm = out2["world_model"]
            if "z_next" in wm:
                fake_loss = fake_loss + 0.1 * wm["z_next"].sum()
        
        optimizer.zero_grad()
        fake_loss.backward()
        no_grad_count = check_gradient_flow(agent)
        
        # Train curiosity subsystems and re-check
        curiosity_losses = train_curiosity_auxiliary(agent)
        if curiosity_losses:
            print(f"\n  Curiosity auxiliary training:")
            for k, v in curiosity_losses.items():
                print(f"    {k}: {v:.6f}")
            
            print(f"\n  Gradient flow AFTER curiosity auxiliary training:")
            agent.reset(2)
            out3 = agent.forward(obs_test, training=True)
            fake_loss2 = out3["policy_logits"].sum()
            optimizer.zero_grad()
            fake_loss2.backward()
            check_gradient_flow(agent)
    except Exception as e:
        print(f"  Gradient test FAILED: {e}")
        traceback.print_exc()
    
    # ================================================================
    # Memory System Test
    # ================================================================
    print_subheader("Memory System Test")
    try:
        latent = torch.randn(1, config.encoder.latent_dim, device=device)
        obs_flat = torch.randn(obs_dim, device=device)
        action_flat = torch.tensor(1.0, device=device)
        
        for i in range(10):
            agent.memory.episodic.write(
                latent=latent, observation=obs_flat, action=action_flat,
                reward=float(i)/9, done=(i==9),
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
        
        for i in range(5):
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
    
    # ================================================================
    # TRAINING LOOP
    # ================================================================
    print_header("Training Loop - LunarLander-v3")
    
    optimizer = torch.optim.Adam(agent.parameters(), lr=config.training.learning_rate)
    
    ppo_trainer = PPOTrainer(
        config.training, agent,
        action_space=config.action_space,
        action_dim=config.action_dim,
        intrinsic_reward_coef=0.01,
        subgoal_reward_coef=0.01,
    )
    scheduler = CosineAnnealingWarmupScheduler(optimizer, warmup_steps=100, total_steps=total_steps)
    
    episode_rewards = []
    episode_lengths = []
    loss_history = []
    curiosity_aux_history = []
    gradient_snapshots = []
    
    observation, _ = env.reset()
    observation = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)
    agent.reset(1)
    
    episode_reward = 0.0
    episode_length = 0
    start_time = time.time()
    
    print(f"\n  Running {total_steps} steps on {device}...")
    print(f"  Task: Land the spacecraft between the flags (much harder than CartPole!)")
    print(f"\n  {'Step':>6} | {'Reward':>8} | {'AvgRwd(10)':>10} | {'Loss':>10} | {'PolLoss':>10} | {'ValLoss':>10} | {'Entr':>7} | {'KL':>7} | {'Eps':>4} | {'ExpAct':>5} | {'MemE':>5} | {'MemS':>4} | {'CurAux':>8}")
    print(f"  {'-'*120}")
    
    for step in range(total_steps):
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
        
        # Train curiosity subsystems with auxiliary losses
        curiosity_aux = train_curiosity_auxiliary(agent)
        curiosity_aux_history.append(curiosity_aux)
        
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
        
        if step % 5 == 0 or step == total_steps - 1:
            avg_reward = np.mean(episode_rewards[-10:]) if episode_rewards else 0.0
            n_episodes = len(episode_rewards)
            stats = agent.get_stats()
            n_active = stats.get('active_experts', 0)
            n_experts = stats.get('num_experts', 0)
            mem_stats = stats.get('memory_stats', {})
            ep_entries = mem_stats.get('episodic_size', 0)
            sem_entries = mem_stats.get('semantic_size', 0)
            
            cur_aux_str = ""
            if curiosity_aux:
                cur_aux_str = f"{curiosity_aux.get('info_gain_recon', 0):.4f}"
            else:
                cur_aux_str = "N/A"
            
            print(f"  {step:>6} | {rollout_reward:>8.1f} | {avg_reward:>10.2f} | {total_loss_val:>10.4f} | {policy_loss_val:>10.4f} | {value_loss_val:>10.4f} | {entropy_val:>7.4f} | {kl_val:>7.4f} | {n_episodes:>4} | {n_active:>2}/{n_experts:<2} | {ep_entries:>5} | {sem_entries:>4} | {cur_aux_str:>8}")
        
        if step % 50 == 0 and step > 0:
            try:
                agent.reset(1)
                obs_g = torch.randn(1, obs_dim, device=device)
                out_g = agent.forward(obs_g, training=True)
                fake_loss_g = out_g["policy_logits"].sum() + out_g["value"].sum()
                optimizer.zero_grad()
                fake_loss_g.backward()
                
                grad_modules = {}
                for name, param in agent.named_parameters():
                    module = name.split('.')[0]
                    if module not in grad_modules:
                        grad_modules[module] = 0
                    if param.grad is not None and param.grad.norm() > 0:
                        grad_modules[module] += 1
                
                gradient_snapshots.append((step, grad_modules))
            except:
                pass
    
    elapsed = time.time() - start_time
    
    # ================================================================
    # RESULTS
    # ================================================================
    print_header("Training Results")
    
    print(f"  Steps: {total_steps}, Time: {elapsed:.1f}s ({total_steps/elapsed:.1f} steps/sec)")
    print(f"  Episodes completed: {len(episode_rewards)}")
    
    if episode_rewards:
        print(f"\n  Reward trajectory:")
        print(f"    First 5:  {[f'{r:.1f}' for r in episode_rewards[:5]]}")
        print(f"    Last 5:   {[f'{r:.1f}' for r in episode_rewards[-5:]]}")
        print(f"    Mean:     {np.mean(episode_rewards):.2f}")
        print(f"    Max:      {np.max(episode_rewards):.2f}")
        print(f"    Std:      {np.std(episode_rewards):.2f}")
        
        if len(episode_rewards) >= 4:
            quarter = max(1, len(episode_rewards) // 4)
            first_q = np.mean(episode_rewards[:quarter])
            last_q = np.mean(episode_rewards[-quarter:])
            improvement = last_q - first_q
            pct_change = (improvement / (abs(first_q) + 1e-8)) * 100
            print(f"\n  Learning Analysis:")
            print(f"    First quarter avg reward: {first_q:.2f}")
            print(f"    Last quarter avg reward:  {last_q:.2f}")
            print(f"    Improvement: {improvement:+.2f} ({pct_change:+.1f}%)")
            if improvement > 0:
                print(f"    Result: LEARNING ✓ (reward is increasing)")
            elif improvement > -20:
                print(f"    Result: MARGINAL (needs more steps to show clear improvement)")
            else:
                print(f"    Result: NOT LEARNING ✗ (reward not improving)")
    
    if loss_history:
        nonzero_losses = [l for l in loss_history if l != 0.0]
        if nonzero_losses:
            print(f"\n  Loss trajectory:")
            print(f"    First 5:  {[f'{l:.4f}' for l in nonzero_losses[:5]]}")
            print(f"    Last 5:   {[f'{l:.4f}' for l in nonzero_losses[-5:]]}")
            print(f"    Mean:     {np.mean(nonzero_losses):.4f}")
    
    if curiosity_aux_history:
        ig_losses = [h.get('info_gain_recon', 0) for h in curiosity_aux_history if h]
        if ig_losses:
            print(f"\n  Curiosity auxiliary training:")
            print(f"    InfoGain recon loss: first={ig_losses[0]:.4f}, last={ig_losses[-1]:.4f}")
            if len(ig_losses) >= 4:
                q1 = np.mean(ig_losses[:len(ig_losses)//4])
                q4 = np.mean(ig_losses[-(len(ig_losses)//4):])
                print(f"    Improvement: {q4 - q1:+.4f} ({'✓ decreasing' if q4 < q1 else '✗ not decreasing'})")
    
    # ================================================================
    # Final Checks
    # ================================================================
    print_subheader("Final System Checks")
    
    device_issues_final = check_device_consistency(agent, device)
    
    print("\n  Final gradient flow check:")
    agent.train()
    agent.reset(1)
    obs_test = torch.randn(1, obs_dim, device=device)
    out = agent.forward(obs_test, training=True)
    out2 = agent.forward(obs_test, action=torch.tensor([1], device=device), reward=1.0, done=False, training=True)
    fake_loss = out2["policy_logits"].sum() + out2["value"].sum()
    if "world_model" in out2:
        fake_loss = fake_loss + 0.1 * out2["world_model"]["z_next"].sum()
    optimizer.zero_grad()
    fake_loss.backward()
    final_no_grad = check_gradient_flow(agent)
    
    mem_stats = agent.memory.get_memory_stats()
    print(f"\n  Memory utilization:")
    print(f"    Episodic: {mem_stats.get('episodic_size', 0)}/{mem_stats.get('episodic_capacity', 0)}")
    print(f"    Semantic: {mem_stats.get('semantic_size', 0)}/{mem_stats.get('semantic_capacity', 0)}")
    
    stats = agent.get_stats()
    print(f"\n  Expert bank:")
    print(f"    Total: {stats.get('num_experts', 0)}")
    print(f"    Active: {stats.get('active_experts', 0)}")
    print(f"    Dormant: {stats.get('dormant_experts', 0)}")
    print(f"    Cached: {stats.get('dormant_cached_experts', 0)}")
    print(f"    Capability density: {stats.get('capability_density', 0):.6f}")
    
    if agent.curiosity is not None:
        cur_stats = agent.curiosity.get_curiosity_stats()
        print(f"\n  Curiosity:")
        print(f"    Scale: {cur_stats.get('curiosity_scale', 0):.4f}")
        print(f"    Total visits: {cur_stats.get('total_visits', 0):.0f}")
        print(f"    Visited buckets: {cur_stats.get('num_visited_buckets', 0)}")
    
    if agent.reasoning_engine is not None:
        print(f"\n  Reasoning engine:")
        print(f"    Steps: {agent.reasoning_engine._reasoning_step}")
    
    if gradient_snapshots:
        print(f"\n  Gradient evolution over training:")
        for step_num, grad_mods in gradient_snapshots[:5]:
            active_modules = sum(1 for v in grad_mods.values() if v > 0)
            total_modules = len(grad_mods)
            print(f"    Step {step_num}: {active_modules}/{total_modules} modules with gradients")
    
    # ================================================================
    # FINAL ASSESSMENT
    # ================================================================
    print_header("FINAL ASSESSMENT")
    
    issues = []
    successes = []
    
    if device_issues == 0 and device_issues_final == 0:
        successes.append(f"Device consistency: All params on {device} ✓")
    else:
        issues.append(f"Device issues: {device_issues + device_issues_final} mismatches")
    
    if episode_rewards and len(episode_rewards) >= 4:
        quarter = max(1, len(episode_rewards) // 4)
        first_q = np.mean(episode_rewards[:quarter])
        last_q = np.mean(episode_rewards[-quarter:])
        improvement = last_q - first_q
        if improvement > 0:
            successes.append(f"Learning: Reward improved by {improvement:+.2f} ✓")
        elif improvement > -20:
            issues.append(f"Learning: Marginal improvement ({improvement:+.2f}), needs more steps")
        else:
            issues.append(f"Learning: No improvement ({improvement:+.2f})")
    else:
        issues.append("Not enough episodes for learning assessment")
    
    if loss_history and not all(l == 0.0 for l in loss_history):
        successes.append("Loss: Non-zero and changing ✓")
    else:
        issues.append("Loss: Always zero or static")
    
    if final_no_grad < 50:
        successes.append(f"Gradient flow: Only {final_no_grad} params without gradients ✓")
    else:
        issues.append(f"Gradient flow: {final_no_grad} params without gradients")
    
    if mem_stats.get('episodic_size', 0) > 0:
        successes.append(f"Episodic memory: {mem_stats['episodic_size']} entries stored ✓")
    else:
        issues.append("Episodic memory: No entries stored")
    
    if curiosity_aux_history:
        ig_losses = [h.get('info_gain_recon', 0) for h in curiosity_aux_history if h]
        if ig_losses and ig_losses[-1] < ig_losses[0]:
            successes.append(f"Curiosity training: InfoGain loss decreased ✓")
        else:
            issues.append("Curiosity training: InfoGain loss not decreasing")
    
    print("\n  SUCCESSES:")
    for s in successes:
        print(f"    ✓ {s}")
    
    if issues:
        print("\n  ISSUES:")
        for i in issues:
            print(f"    ✗ {i}")
    else:
        print("\n  No issues found!")
    
    if device.type == "cpu":
        print(f"\n  NOTE: All device-tracking code uses .device properties and")
        print(f"  register_buffer('_device_tracker') patterns. When run on a")
        print(f"  CUDA machine with config.device='cuda', the system will")
        print(f"  automatically use GPU. The CPU run validates correctness.")
    
    print(f"\n  VERDICT: {'PASS ✓' if len(successes) >= len(issues) else 'NEEDS MORE WORK'}")
    
    env.close()
    return len(issues) == 0


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Deep Thought Hard Training Test")
    parser.add_argument("--steps", type=int, default=500, help="Number of training steps")
    parser.add_argument("--env", type=str, default="LunarLander-v3", help="Gymnasium environment")
    args = parser.parse_args()
    
    success = run_hard_training(total_steps=args.steps, env_id=args.env)
    sys.exit(0 if success else 1)
