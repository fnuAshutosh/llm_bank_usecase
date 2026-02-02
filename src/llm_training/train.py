"""
Training Script for Banking LLM

Train a transformer from scratch on banking conversations
"""

import json
import time
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset

from .tokenizer import SimpleTokenizer, create_banking_tokenizer
from .transformer import BankingLLM, create_small_model


class BankingDataset(Dataset):
    """Dataset for banking conversations"""
    
    def __init__(
        self,
        texts: List[str],
        tokenizer: SimpleTokenizer,
        max_length: int = 512
    ):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Pre-tokenize all texts
        self.encoded_texts = [
            tokenizer.encode(text, add_special_tokens=True)
            for text in texts
        ]
    
    def __len__(self) -> int:
        return len(self.encoded_texts)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # Get encoded text
        token_ids = self.encoded_texts[idx]
        
        # Truncate if too long
        if len(token_ids) > self.max_length:
            token_ids = token_ids[:self.max_length]
        
        # For language modeling: input is tokens[:-1], target is tokens[1:]
        input_ids = token_ids[:-1]
        target_ids = token_ids[1:]
        
        # Pad to max_length - 1 (since we shifted)
        pad_length = self.max_length - 1 - len(input_ids)
        if pad_length > 0:
            input_ids = input_ids + [self.tokenizer.pad_id] * pad_length
            target_ids = target_ids + [-1] * pad_length  # -1 will be ignored in loss
        
        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'target_ids': torch.tensor(target_ids, dtype=torch.long),
        }


class LLMTrainer:
    """Trainer for banking LLM"""
    
    def __init__(
        self,
        model: BankingLLM,
        tokenizer: SimpleTokenizer,
        train_dataset: Dataset,
        val_dataset: Dataset,
        device: str = 'cpu',
        lr: float = 3e-4,
        batch_size: int = 8,
        accumulation_steps: int = 4,
    ):
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.device = device
        
        # Dataloaders
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,  # Set to 0 for Codespaces
        )
        
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
        )
        
        # Optimizer (AdamW like GPT-3)
        self.optimizer = AdamW(
            model.parameters(),
            lr=lr,
            betas=(0.9, 0.95),
            weight_decay=0.1,
        )
        
        # Learning rate scheduler
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=len(self.train_loader) * 10,  # 10 epochs
            eta_min=lr * 0.1,
        )
        
        self.accumulation_steps = accumulation_steps
        
        # Tracking
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
    
    def train_epoch(self, epoch: int) -> float:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        num_batches = len(self.train_loader)
        
        self.optimizer.zero_grad()
        
        for i, batch in enumerate(self.train_loader):
            input_ids = batch['input_ids'].to(self.device)
            target_ids = batch['target_ids'].to(self.device)
            
            # Forward pass
            logits, loss = self.model(input_ids, target_ids)
            
            # Normalize loss for accumulation
            loss = loss / self.accumulation_steps
            
            # Backward pass
            loss.backward()
            
            # Update weights every accumulation_steps
            if (i + 1) % self.accumulation_steps == 0:
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
            
            total_loss += loss.item() * self.accumulation_steps
            
            # Print progress
            if (i + 1) % 10 == 0:
                avg_loss = total_loss / (i + 1)
                print(f"Epoch {epoch} | Batch {i+1}/{num_batches} | Loss: {avg_loss:.4f}")
        
        return total_loss / num_batches
    
    @torch.no_grad()
    def evaluate(self) -> float:
        """Evaluate on validation set"""
        self.model.eval()
        total_loss = 0
        num_batches = len(self.val_loader)
        
        for batch in self.val_loader:
            input_ids = batch['input_ids'].to(self.device)
            target_ids = batch['target_ids'].to(self.device)
            
            # Forward pass
            logits, loss = self.model(input_ids, target_ids)
            total_loss += loss.item()
        
        return total_loss / num_batches
    
    def train(self, num_epochs: int, save_dir: str = 'models'):
        """Full training loop"""
        save_path = Path(save_dir)
        save_path.mkdir(exist_ok=True)
        
        print("=" * 80)
        print(f"Training Banking LLM for {num_epochs} epochs")
        print(f"Device: {self.device}")
        print(f"Model parameters: {self.model.count_parameters()['total_millions']:.2f}M")
        print("=" * 80)
        
        for epoch in range(1, num_epochs + 1):
            start_time = time.time()
            
            # Train
            train_loss = self.train_epoch(epoch)
            self.train_losses.append(train_loss)
            
            # Validate
            val_loss = self.evaluate()
            self.val_losses.append(val_loss)
            
            epoch_time = time.time() - start_time
            
            print(f"\nEpoch {epoch} Summary:")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val Loss: {val_loss:.4f}")
            print(f"  Time: {epoch_time:.2f}s")
            print("-" * 80)
            
            # Save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_checkpoint(save_path / 'best_model.pt')
                print(f"  ✓ New best model saved (val_loss: {val_loss:.4f})")
            
            # Save checkpoint every 5 epochs
            if epoch % 5 == 0:
                self.save_checkpoint(save_path / f'checkpoint_epoch{epoch}.pt')
            
            print()
        
        print("Training complete!")
        print(f"Best validation loss: {self.best_val_loss:.4f}")
    
    def save_checkpoint(self, filepath: Path):
        """Save model checkpoint"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'best_val_loss': self.best_val_loss,
        }, filepath)
    
    def load_checkpoint(self, filepath: Path):
        """Load model checkpoint"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.train_losses = checkpoint['train_losses']
        self.val_losses = checkpoint['val_losses']
        self.best_val_loss = checkpoint['best_val_loss']


def load_banking_data(filepath: str = None) -> List[str]:
    """
    Load banking conversation data
    
    For now, returns sample data. Replace with real data loading.
    """
    if filepath and Path(filepath).exists():
        with open(filepath, 'r') as f:
            return [line.strip() for line in f if line.strip()]
    
    # Sample banking conversations
    sample_data = [
        "Customer: What is my current account balance? Agent: Your checking account balance is $2,450.32.",
        "Customer: I need to transfer $500 to savings. Agent: Transfer initiated. Your new checking balance is $1,950.32.",
        "Customer: How much interest do I earn? Agent: Your savings account earns 2.5% APY.",
        "Customer: Can you help with a declined transaction? Agent: Let me check. The transaction was declined due to insufficient funds.",
        "Customer: What are your ATM fees? Agent: ATM withdrawals at our network are free. Out-of-network fees are $3.",
        "Customer: I want to open a new account. Agent: Great! We offer checking, savings, and money market accounts.",
        "Customer: How do I dispute a charge? Agent: You can file a dispute through our mobile app or by calling customer service.",
        "Customer: What's my credit card limit? Agent: Your current credit limit is $10,000 with $7,200 available.",
        "Customer: Can I get a loan? Agent: Yes, we offer personal loans, auto loans, and mortgages. What are you looking for?",
        "Customer: I lost my debit card. Agent: I'll deactivate it immediately and order a replacement. It will arrive in 5-7 days.",
    ] * 100  # Repeat for more training data
    
    return sample_data


def main():
    """Main training script"""
    
    # Device
    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Load data
    print("Loading data...")
    all_texts = load_banking_data()
    
    # Split train/val
    split_idx = int(len(all_texts) * 0.9)
    train_texts = all_texts[:split_idx]
    val_texts = all_texts[split_idx:]
    
    print(f"Train samples: {len(train_texts)}")
    print(f"Val samples: {len(val_texts)}")
    
    # Create tokenizer
    print("\nBuilding tokenizer...")
    tokenizer = create_banking_tokenizer(all_texts, vocab_size=5000)
    
    # Save tokenizer
    tokenizer.save('models/tokenizer.json')
    
    # Create datasets
    print("Creating datasets...")
    train_dataset = BankingDataset(train_texts, tokenizer, max_length=256)
    val_dataset = BankingDataset(val_texts, tokenizer, max_length=256)
    
    # Create model
    print("Initializing model...")
    model = create_small_model(tokenizer.vocab_size)
    print(f"Model size: {model.count_parameters()['total_millions']:.2f}M parameters")
    
    # Create trainer
    trainer = LLMTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        device=device,
        lr=3e-4,
        batch_size=4,  # Small batch for Codespaces
        accumulation_steps=8,  # Effective batch size = 32
    )
    
    # Train
    trainer.train(num_epochs=10, save_dir='models')
    
    print("\n" + "=" * 80)
    print("Testing generation...")
    print("=" * 80)
    
    # Test generation
    model.eval()
    test_prompts = [
        "Customer: What is my balance?",
        "Customer: I need help with",
        "Customer: How do I transfer",
    ]
    
    for prompt in test_prompts:
        print(f"\nPrompt: {prompt}")
        
        # Encode
        input_ids = torch.tensor([tokenizer.encode(prompt, add_special_tokens=True)]).to(device)
        
        # Generate
        generated = model.generate(
            input_ids,
            max_new_tokens=30,
            temperature=0.8,
            top_k=50,
        )
        
        # Decode
        generated_text = tokenizer.decode(generated[0].tolist())
        print(f"Generated: {generated_text}")


if __name__ == "__main__":
    main()
