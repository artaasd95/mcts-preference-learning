import torch
from torch.utils.data import Dataset, DataLoader
from transformers import Trainer, TrainingArguments
from typing import List, Tuple, Dict
import numpy as np

class PreferenceDataset(Dataset):
    def __init__(self, preferences: List[Tuple[str, str, str]]):
        self.preferences = preferences
    
    def __len__(self):
        return len(self.preferences)
    
    def __getitem__(self, idx):
        prompt, winner, loser = self.preferences[idx]
        return {
            "prompt": prompt,
            "winner": winner,
            "loser": loser
        }

class DPOTrainer:
    def __init__(
        self,
        model,
        tokenizer,
        beta: float = 0.1,
        batch_size: int = 32,
        learning_rate: float = 1e-5,
        max_epochs: int = 3
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.beta = beta
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        
        # Initialize reference model (frozen copy of current model)
        self.ref_model = type(model)(model.config)
        self.ref_model.load_state_dict(model.state_dict())
        self.ref_model.eval()
    
    def compute_dpo_loss(
        self,
        prompt: str,
        winner: str,
        loser: str,
        alpha: float = 0.0
    ) -> torch.Tensor:
        """Compute DPO loss for a single preference pair."""
        # Get logits for winner and loser
        winner_logits = self.model(prompt + winner)
        loser_logits = self.model(prompt + loser)
        
        # Get reference model logits
        with torch.no_grad():
            ref_winner_logits = self.ref_model(prompt + winner)
            ref_loser_logits = self.ref_model(prompt + loser)
        
        # Compute preference difference
        h = (
            winner_logits - ref_winner_logits
            - (loser_logits - ref_loser_logits)
        )
        
        # Compute DPO loss
        loss = -(
            (1 - alpha) * torch.log(torch.sigmoid(self.beta * h))
            + alpha * torch.log(torch.sigmoid(-self.beta * h))
        )
        
        return loss
    
    def train_step(
        self,
        batch: Dict[str, str],
        alpha: float = 0.0
    ) -> torch.Tensor:
        """Perform a single training step."""
        self.model.train()
        
        total_loss = 0
        for prompt, winner, loser in zip(
            batch["prompt"],
            batch["winner"],
            batch["loser"]
        ):
            loss = self.compute_dpo_loss(prompt, winner, loser, alpha)
            total_loss += loss
        
        return total_loss / len(batch["prompt"])
    
    def train(
        self,
        preferences: List[Tuple[str, str, str]],
        validation_split: float = 0.1
    ):
        """Train the model using DPO."""
        # Split into train and validation
        np.random.shuffle(preferences)
        split_idx = int(len(preferences) * (1 - validation_split))
        train_prefs = preferences[:split_idx]
        val_prefs = preferences[split_idx:]
        
        # Create datasets
        train_dataset = PreferenceDataset(train_prefs)
        val_dataset = PreferenceDataset(val_prefs)
        
        # Create dataloaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size
        )
        
        # Initialize optimizer
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate
        )
        
        # Training loop
        best_val_loss = float('inf')
        for epoch in range(self.max_epochs):
            # Training
            self.model.train()
            train_loss = 0
            for batch in train_loader:
                loss = self.train_step(batch)
                train_loss += loss.item()
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            # Validation
            self.model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch in val_loader:
                    loss = self.train_step(batch)
                    val_loss += loss.item()
            
            # Print progress
            print(f"Epoch {epoch + 1}/{self.max_epochs}")
            print(f"Train Loss: {train_loss / len(train_loader):.4f}")
            print(f"Val Loss: {val_loss / len(val_loader):.4f}")
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                # Save model checkpoint
                torch.save(self.model.state_dict(), "best_model.pt") 