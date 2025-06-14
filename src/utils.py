import torch
import numpy as np
import random
import os
from typing import List, Dict, Any
import json
import logging

def set_seed(seed: int):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def setup_logging(log_dir: str = "logs"):
    """Setup logging configuration."""
    os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(log_dir, 'training.log')),
            logging.StreamHandler()
        ]
    )

def save_json(data: List[Dict[str, Any]], filepath: str):
    """Save data to JSON file."""
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)

def load_json(filepath: str) -> List[Dict[str, Any]]:
    """Load data from JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)

def compute_metrics(predictions: List[str], ground_truths: List[str]) -> Dict[str, float]:
    """Compute evaluation metrics."""
    exact_matches = sum(1 for p, g in zip(predictions, ground_truths) if p.strip() == g.strip())
    accuracy = exact_matches / len(predictions) if predictions else 0.0
    
    return {
        "accuracy": accuracy,
        "exact_matches": exact_matches,
        "total_samples": len(predictions)
    }

def format_step(step: str) -> str:
    """Format a reasoning step for better readability."""
    if step.startswith(("1.", "2.", "3.", "4.", "5.", "6.", "7.", "8.", "9.")):
        return step
    return f"• {step}"

def create_checkpoint_dir(base_dir: str, iteration: int) -> str:
    """Create and return checkpoint directory path."""
    checkpoint_dir = os.path.join(base_dir, f"iteration_{iteration}")
    os.makedirs(checkpoint_dir, exist_ok=True)
    return checkpoint_dir 