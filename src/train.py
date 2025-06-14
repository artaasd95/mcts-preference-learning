import torch
from models.qwen_model import QwenModel
from mcts.mcts import MCTS
from training.dpo_trainer import DPOTrainer
from typing import List, Tuple
import json
import os
from tqdm import tqdm

def load_dataset(dataset_path: str) -> List[str]:
    """Load prompts from dataset."""
    with open(dataset_path, 'r') as f:
        data = json.load(f)
    return [item['prompt'] for item in data]

def main():
    # Initialize model
    model = QwenModel(
        model_name="Qwen/Qwen-1_5B",
        load_in_8bit=True
    )
    
    # Initialize MCTS
    mcts = MCTS(
        model=model,
        c_puct=1.0,
        num_simulations=5,
        max_depth=4
    )
    
    # Initialize DPO trainer
    trainer = DPOTrainer(
        model=model.model,
        tokenizer=model.tokenizer,
        beta=0.1,
        batch_size=32,
        learning_rate=1e-5,
        max_epochs=3
    )
    
    # Load dataset
    dataset_path = "data/train.json"  # Update with your dataset path
    prompts = load_dataset(dataset_path)
    
    # Training loop
    num_iterations = 10
    for iteration in range(num_iterations):
        print(f"\nIteration {iteration + 1}/{num_iterations}")
        
        # Collect preferences using MCTS
        all_preferences = []
        for prompt in tqdm(prompts, desc="Collecting preferences"):
            preference_pairs = mcts.search(prompt)
            for winner, loser in preference_pairs:
                all_preferences.append((prompt, winner, loser))
        
        # Train using DPO
        trainer.train(all_preferences)
        
        # Save checkpoint
        checkpoint_dir = f"checkpoints/iteration_{iteration + 1}"
        os.makedirs(checkpoint_dir, exist_ok=True)
        torch.save(model.model.state_dict(), f"{checkpoint_dir}/model.pt")

if __name__ == "__main__":
    main() 