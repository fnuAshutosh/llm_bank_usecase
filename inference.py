"""
Interactive inference with trained Banking LLM

Load your trained model and generate responses to custom prompts
"""

from pathlib import Path

import torch

from src.llm_training.tokenizer import SimpleTokenizer
from src.llm_training.transformer import BankingLLM


class BankingLLMInference:
    """Interactive inference engine"""
    
    def __init__(self, model_path: str = 'models/best_model.pt', tokenizer_path: str = 'models/tokenizer.json', device: str = 'cpu'):
        self.device = device
        
        # Load tokenizer
        print("Loading tokenizer...")
        self.tokenizer = SimpleTokenizer()
        self.tokenizer.load(tokenizer_path)
        
        # Load model
        print("Loading model...")
        checkpoint = torch.load(model_path, map_location=device)
        
        # Create model with correct dimensions (match training config)
        vocab_size = self.tokenizer.vocab_size
        self.model = BankingLLM(
            vocab_size=vocab_size,
            d_model=512,
            num_heads=8,
            num_layers=6,
            d_ff=2048,
            max_seq_len=512,  # Must match training config
            dropout=0.1,
        )
        
        # Load weights
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(device)
        self.model.eval()
        
        print(f"✓ Model loaded ({self.model.count_parameters()['total_millions']:.2f}M parameters)")
        print(f"✓ Tokenizer loaded (vocab size: {vocab_size})")
        print("\n" + "="*80)
        print("🏦 Banking LLM - Interactive Inference")
        print("="*80)
        print("\nYour trained model is ready! Ask banking questions.\n")
    
    def generate(
        self,
        prompt: str,
        max_tokens: int = 50,
        temperature: float = 0.8,
        top_k: int = 40,
        top_p: float = 0.9,
    ) -> str:
        """Generate response to a prompt"""
        
        # Encode prompt
        input_ids = torch.tensor([self.tokenizer.encode(prompt, add_special_tokens=True)]).to(self.device)
        
        # Generate
        with torch.no_grad():
            generated = self.model.generate(
                input_ids,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
            )
        
        # Decode
        response = self.tokenizer.decode(generated[0].tolist())
        return response
    
    def chat(self):
        """Interactive chat loop"""
        print("Type 'exit' or 'quit' to stop\n")
        
        while True:
            try:
                # Get user input
                prompt = input("You: ").strip()
                
                if prompt.lower() in ['exit', 'quit', 'bye']:
                    print("\nGoodbye! 👋")
                    break
                
                if not prompt:
                    continue
                
                # Generate response
                print("\nThinking", end="", flush=True)
                response = self.generate(
                    prompt,
                    max_tokens=60,
                    temperature=0.8,
                    top_k=40,
                    top_p=0.9,
                )
                
                print("\r" + " "*30 + "\r", end="", flush=True)  # Clear "Thinking..."
                print(f"Agent: {response}\n")
                
            except KeyboardInterrupt:
                print("\n\nInterrupted. Goodbye! 👋")
                break
            except Exception as e:
                print(f"Error: {e}")
                print("Please try again.\n")


def main():
    """Main entry point"""
    import sys
    
    # Check if model exists
    if not Path('models/best_model.pt').exists():
        print("❌ Model not found! Train the model first with:")
        print("   python -m src.llm_training.train")
        sys.exit(1)
    
    # Detect device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}\n")
    
    # Create inference engine
    engine = BankingLLMInference(device=device)
    
    # Start chat
    engine.chat()


if __name__ == "__main__":
    main()
