
"""
Inference Handler for Custom Banking LLM
"""

import json
import logging
from pathlib import Path
from typing import List, Optional

import torch

from .tokenizer import SimpleTokenizer
from .transformer import BankingLLM

logger = logging.getLogger(__name__)

class CustomModelHandler:
    """Handles inference for the custom BankingLLM"""
    
    def __init__(self, model_dir: str = "models", model_name: str = "best_model.pt"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_path = Path(model_dir) / model_name
        self.tokenizer_path = Path(model_dir) / "tokenizer.json"
        
        self.model: Optional[BankingLLM] = None
        self.tokenizer: Optional[SimpleTokenizer] = None
        self.is_ready = False
        
        self._load_model()
    
    def _load_model(self):
        """Load model and tokenizer if available"""
        try:
            if not self.model_path.exists() or not self.tokenizer_path.exists():
                logger.warning(f"Custom model not found at {self.model_path}. Using placeholder.")
                return

            logger.info(f"Loading custom model from {self.model_path}...")
            
            # Load tokenizer
            self.tokenizer = SimpleTokenizer.load(str(self.tokenizer_path))
            
            # Initialize model architecture (must match training config)
            # TODO: Load config from file instead of hardcoding
            checkpoint = torch.load(str(self.model_path), map_location=self.device)
            
            # infer params from checkpoint if possible or use defaults
            # For now using Small model defaults from train.py
            self.model = BankingLLM(
                vocab_size=self.tokenizer.vocab_size,
                d_model=512,
                num_heads=8,
                num_layers=6,
                d_ff=2048,
                max_seq_len=512,
                dropout=0.1
            )
            
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.to(self.device)
            self.model.eval()
            
            self.is_ready = True
            logger.info("Custom banking model loaded successfully!")
            
        except Exception as e:
            logger.error(f"Failed to load custom model: {e}")
            self.is_ready = False

    def generate(
        self, 
        prompt: str, 
        max_new_tokens: int = 100, 
        temperature: float = 0.7
    ) -> str:
        """Generate text from prompt"""
        if not self.is_ready:
            return "Error: Custom model is not trained/loaded. Please run training script first."
            
        try:
            # Encode
            input_ids = torch.tensor([self.tokenizer.encode(prompt, add_special_tokens=True)]).to(self.device)
            
            # Generate
            with torch.no_grad():
                output_ids = self.model.generate(
                    input_ids,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_k=50
                )
            
            # Decode
            generated_text = self.tokenizer.decode(output_ids[0].tolist())
            
            # Remove prompt from output if present (simple check)
            if generated_text.startswith(prompt):
                 generated_text = generated_text[len(prompt):]
                 
            return generated_text.strip()
            
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            return f"Error generating response: {str(e)}"
