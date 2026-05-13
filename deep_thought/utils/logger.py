"""
Logging utilities for Deep Thought.
"""

import logging
import sys
from typing import Optional
from pathlib import Path


def setup_logger(
    name: str = "deep_thought",
    log_dir: Optional[str] = None,
    level: int = logging.INFO
) -> logging.Logger:
    """
    Setup logger with file and console handlers.
    
    Args:
        name: Logger name
        log_dir: Directory for log files
        level: Logging level
        
    Returns:
        logger: Configured logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Remove existing handlers
    logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # File handler
    if log_dir is not None:
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_path / f"{name}.log")
        file_handler.setLevel(level)
        file_formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    
    return logger


class Logger:
    """Simple logger wrapper for convenience."""
    
    def __init__(self, name: str = "deep_thought", log_dir: Optional[str] = None):
        self.logger = setup_logger(name, log_dir)
    
    def info(self, msg: str):
        self.logger.info(msg)
    
    def debug(self, msg: str):
        self.logger.debug(msg)
    
    def warning(self, msg: str):
        self.logger.warning(msg)
    
    def error(self, msg: str):
        self.logger.error(msg)
