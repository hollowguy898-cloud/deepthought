"""
Command-line interface for Deep Thought.
"""

import argparse
import sys

from deep_thought.train import train
from deep_thought.config import DeepThoughtConfig


def main():
    parser = argparse.ArgumentParser(description="Deep Thought RL Training")
    
    parser.add_argument(
        "--env",
        type=str,
        default="CartPole-v1",
        help="Gym environment ID"
    )
    
    parser.add_argument(
        "--steps",
        type=int,
        default=10_000_000,
        help="Total training steps"
    )
    
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config YAML file"
    )
    
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from"
    )
    
    parser.add_argument(
        "--log-dir",
        type=str,
        default="./logs",
        help="Logging directory"
    )
    
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use"
    )
    
    args = parser.parse_args()
    
    # Load or create config
    if args.config is not None:
        config = DeepThoughtConfig.from_yaml(args.config)
    else:
        config = DeepThoughtConfig()
    
    # Override with CLI args
    config.log_dir = args.log_dir
    config.device = args.device
    
    # Train
    train(
        config,
        env_id=args.env,
        total_steps=args.steps,
        resume_from=args.resume
    )


if __name__ == "__main__":
    main()
