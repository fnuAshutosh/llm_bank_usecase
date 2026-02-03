"""
Complete Training Pipeline for Banking LLM with LoRA

Trains TinyLlama 1.1B on Banking77 dataset using:
- QLoRA for parameter-efficient fine-tuning
- Real banking conversation data
- Mixed precision training
- Validation and checkpointing
"""

import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from datasets import DatasetDict, load_dataset
from peft import LoraConfig, TaskType, get_peft_model
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

logger = logging.getLogger(__name__)

# Suppress transformers warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"


class BankingIntentDataset(Dataset):
    """Banking intent classification dataset"""
    
    def __init__(self, 
                 data: List[Dict],
                 tokenizer,
                 max_length: int = 512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Banking-specific system prompt
        self.system_prompt = """You are a professional banking assistant. 
Respond concisely and accurately to customer banking inquiries.
Maintain professional tone and ensure security compliance."""
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Format as instruction-following
        instruction = item.get("text", item.get("input", ""))
        response = item.get("output", "")
        
        # Create formatted prompt
        formatted_prompt = f"""<s>[INST] <<SYS>>
{self.system_prompt}
<</SYS>>

{instruction} [/INST] {response} </s>"""
        
        # Tokenize
        encoded = self.tokenizer(
            formatted_prompt,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        return {
            "input_ids": encoded["input_ids"].squeeze(),
            "attention_mask": encoded["attention_mask"].squeeze(),
        }


class BankingLLMTrainer:
    """Complete trainer for Banking LLM with LoRA"""
    
    def __init__(self,
                 model_name: str = "TinyLlama/TinyLlama-1.1B",
                 data_path: Optional[str] = None,
                 output_dir: str = "models/banking_llm",
                 device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        """
        Initialize trainer
        
        Args:
            model_name: Base model from Hugging Face
            data_path: Path to training data (JSON)
            output_dir: Output directory for checkpoints
            device: Device to train on
        """
        self.model_name = model_name
        self.data_path = data_path or "data/finetuning/train.json"
        self.output_dir = output_dir
        self.device = device
        
        # Create output directory
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.tokenizer = None
        self.model = None
        self.train_dataset = None
        self.val_dataset = None
        self.trainer = None
        
        logger.info(f"BankingLLMTrainer initialized - Model: {model_name}, Device: {device}")
    
    def load_data(self) -> Tuple[List[Dict], List[Dict]]:
        """Load training data from JSON"""
        logger.info(f"Loading data from {self.data_path}...")
        
        with open(self.data_path, 'r') as f:
            data = json.load(f)
        
        logger.info(f"Loaded {len(data)} training examples")
        
        # Split into train/val (80/20)
        np.random.shuffle(data)
        split_idx = int(0.8 * len(data))
        
        train_data = data[:split_idx]
        val_data = data[split_idx:]
        
        logger.info(f"Train: {len(train_data)}, Val: {len(val_data)}")
        
        return train_data, val_data
    
    def setup_model_and_tokenizer(self):
        """Initialize tokenizer and model with QLoRA"""
        logger.info(f"Loading tokenizer from {self.model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        # Set pad token if not set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        logger.info(f"Loading model {self.model_name}...")
        
        # 4-bit quantization config for efficient training
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        
        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=bnb_config,
            device_map=self.device,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2" if torch.cuda.is_available() else None,
            trust_remote_code=True
        )
        
        # Add LoRA configuration
        lora_config = LoraConfig(
            r=32,  # LoRA rank
            lora_alpha=64,  # LoRA scaling
            target_modules=["q_proj", "v_proj"],  # Target attention projections
            lora_dropout=0.05,
            bias="none",
            task_type=TaskType.CAUSAL_LM
        )
        
        # Wrap model with LoRA
        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()
        
        logger.info("✓ Model and LoRA setup complete")
    
    def prepare_datasets(self, train_data: List[Dict], val_data: List[Dict]):
        """Prepare datasets for training"""
        logger.info("Preparing datasets...")
        
        self.train_dataset = BankingIntentDataset(
            train_data,
            self.tokenizer,
            max_length=512
        )
        
        self.val_dataset = BankingIntentDataset(
            val_data,
            self.tokenizer,
            max_length=512
        )
        
        logger.info(f"✓ Train: {len(self.train_dataset)}, Val: {len(self.val_dataset)}")
    
    def train(self,
              num_epochs: int = 3,
              batch_size: int = 4,
              learning_rate: float = 2e-4,
              warmup_steps: int = 100):
        """
        Train the model
        
        Args:
            num_epochs: Number of training epochs
            batch_size: Batch size
            learning_rate: Learning rate
            warmup_steps: Number of warmup steps
        """
        logger.info("Starting training...")
        
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            learning_rate=learning_rate,
            warmup_steps=warmup_steps,
            weight_decay=0.01,
            logging_dir='./logs',
            logging_steps=10,
            save_steps=50,
            eval_steps=50,
            save_total_limit=3,
            evaluation_strategy="steps",
            save_strategy="steps",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            gradient_accumulation_steps=4,
            mixed_precision="bf16" if torch.cuda.is_available() else "no",
            optim="paged_adamw_32bit",
            dataloader_pin_memory=True,
        )
        
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.val_dataset,
            data_collator=DataCollatorForLanguageModeling(self.tokenizer, mlm=False),
        )
        
        # Start training
        train_result = self.trainer.train()
        
        logger.info(f"✓ Training complete. Results: {train_result}")
        
        return train_result
    
    def save_model(self, save_path: Optional[str] = None):
        """Save trained model"""
        save_path = save_path or f"{self.output_dir}/best_model"
        
        logger.info(f"Saving model to {save_path}...")
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
        
        logger.info("✓ Model saved successfully")
    
    def evaluate(self) -> Dict:
        """Evaluate on validation set"""
        logger.info("Running evaluation...")
        
        eval_results = self.trainer.evaluate()
        
        logger.info(f"Evaluation Results: {eval_results}")
        
        return eval_results


def train_banking_llm():
    """Main training function"""
    
    # Initialize trainer
    trainer = BankingLLMTrainer(
        model_name="TinyLlama/TinyLlama-1.1B",
        data_path="data/finetuning/train.json",
        output_dir="models/banking_llm"
    )
    
    # Load data
    train_data, val_data = trainer.load_data()
    
    # Setup model and tokenizer
    trainer.setup_model_and_tokenizer()
    
    # Prepare datasets
    trainer.prepare_datasets(train_data, val_data)
    
    # Train
    trainer.train(
        num_epochs=3,
        batch_size=4,
        learning_rate=2e-4
    )
    
    # Evaluate
    eval_results = trainer.evaluate()
    
    # Save
    trainer.save_model()
    
    return trainer, eval_results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    train_banking_llm()
