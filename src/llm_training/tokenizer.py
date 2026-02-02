"""
Tokenizer for Banking LLM

Simple BPE (Byte-Pair Encoding) tokenizer for learning purposes.
In production, you'd use tiktoken or sentencepiece.
"""

import json
import re
from collections import Counter
from typing import Dict, List


class SimpleTokenizer:
    """
    Simple character-level tokenizer to start with
    Will upgrade to BPE later
    """
    
    def __init__(self):
        self.vocab: Dict[str, int] = {}
        self.inverse_vocab: Dict[int, str] = {}
        
        # Special tokens
        self.pad_token = "<PAD>"
        self.unk_token = "<UNK>"
        self.bos_token = "<BOS>"  # Beginning of sequence
        self.eos_token = "<EOS>"  # End of sequence
        
        self.pad_id = 0
        self.unk_id = 1
        self.bos_id = 2
        self.eos_id = 3
        
        # Initialize with special tokens
        self.vocab = {
            self.pad_token: self.pad_id,
            self.unk_token: self.unk_id,
            self.bos_token: self.bos_id,
            self.eos_token: self.eos_id,
        }
        self.inverse_vocab = {v: k for k, v in self.vocab.items()}
    
    def build_vocab(self, texts: List[str], vocab_size: int = 10000):
        """
        Build vocabulary from corpus
        
        Args:
            texts: List of text samples
            vocab_size: Target vocabulary size
        """
        # Count all characters/tokens
        token_counts = Counter()
        
        for text in texts:
            # Simple whitespace tokenization for now
            tokens = self._tokenize_text(text)
            token_counts.update(tokens)
        
        # Get most common tokens
        most_common = token_counts.most_common(vocab_size - len(self.vocab))
        
        # Add to vocabulary
        for token, _ in most_common:
            if token not in self.vocab:
                token_id = len(self.vocab)
                self.vocab[token] = token_id
                self.inverse_vocab[token_id] = token
        
        print(f"Vocabulary built: {len(self.vocab)} tokens")
    
    def _tokenize_text(self, text: str) -> List[str]:
        """Basic tokenization"""
        # Split on whitespace and punctuation
        text = text.lower()
        tokens = re.findall(r'\w+|[^\w\s]', text)
        return tokens
    
    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """
        Convert text to token IDs
        
        Args:
            text: Input text
            add_special_tokens: Whether to add BOS/EOS tokens
            
        Returns:
            List of token IDs
        """
        tokens = self._tokenize_text(text)
        
        # Convert to IDs
        token_ids = [
            self.vocab.get(token, self.unk_id)
            for token in tokens
        ]
        
        # Add special tokens
        if add_special_tokens:
            token_ids = [self.bos_id] + token_ids + [self.eos_id]
        
        return token_ids
    
    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        """
        Convert token IDs back to text
        
        Args:
            token_ids: List of token IDs
            skip_special_tokens: Whether to skip special tokens
            
        Returns:
            Decoded text
        """
        tokens = []
        
        for token_id in token_ids:
            token = self.inverse_vocab.get(token_id, self.unk_token)
            
            # Skip special tokens if requested
            if skip_special_tokens and token in [self.pad_token, self.unk_token, self.bos_token, self.eos_token]:
                continue
            
            tokens.append(token)
        
        # Simple join (will be smarter with BPE)
        return ' '.join(tokens)
    
    def save(self, filepath: str):
        """Save vocabulary to file"""
        with open(filepath, 'w') as f:
            json.dump({
                'vocab': self.vocab,
                'special_tokens': {
                    'pad': self.pad_token,
                    'unk': self.unk_token,
                    'bos': self.bos_token,
                    'eos': self.eos_token,
                }
            }, f, indent=2)
        print(f"Tokenizer saved to {filepath}")
    
    def load(self, filepath: str):
        """Load vocabulary from file"""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        self.vocab = data['vocab']
        self.inverse_vocab = {int(v): k for k, v in self.vocab.items()}
        
        special = data['special_tokens']
        self.pad_token = special['pad']
        self.unk_token = special['unk']
        self.bos_token = special['bos']
        self.eos_token = special['eos']
        
        self.pad_id = self.vocab[self.pad_token]
        self.unk_id = self.vocab[self.unk_token]
        self.bos_id = self.vocab[self.bos_token]
        self.eos_id = self.vocab[self.eos_token]
        
        print(f"Tokenizer loaded from {filepath}")
    
    @property
    def vocab_size(self) -> int:
        """Get vocabulary size"""
        return len(self.vocab)


class BPETokenizer:
    """
    Byte-Pair Encoding tokenizer (more advanced)
    
    This learns subword units by iteratively merging the most
    frequent adjacent pairs of characters/tokens.
    
    Example:
        "banking" -> ["bank", "ing"] instead of character-level
    
    TODO: Implement full BPE algorithm
    """
    
    def __init__(self):
        # Will implement later
        pass


# Banking-specific vocabulary examples
BANKING_VOCAB = {
    # Common banking terms
    'account', 'balance', 'transaction', 'deposit', 'withdraw',
    'transfer', 'payment', 'loan', 'credit', 'debit',
    'savings', 'checking', 'interest', 'rate', 'fee',
    'statement', 'card', 'atm', 'branch', 'bank',
    
    # Customer service
    'hello', 'help', 'question', 'issue', 'problem',
    'thanks', 'thank', 'you', 'please', 'sorry',
    
    # Actions
    'check', 'view', 'show', 'get', 'need',
    'want', 'can', 'could', 'would', 'how',
    
    # Common numbers/amounts
    'hundred', 'thousand', 'million', 'dollar', 'dollars',
}


def create_banking_tokenizer(corpus: List[str], vocab_size: int = 10000) -> SimpleTokenizer:
    """
    Create and train a tokenizer on banking conversations
    
    Args:
        corpus: List of banking conversation texts
        vocab_size: Target vocabulary size
        
    Returns:
        Trained tokenizer
    """
    tokenizer = SimpleTokenizer()
    tokenizer.build_vocab(corpus, vocab_size)
    return tokenizer


if __name__ == "__main__":
    # Test tokenizer
    tokenizer = SimpleTokenizer()
    
    # Sample banking conversations
    corpus = [
        "What is my account balance?",
        "I want to transfer money to my savings account.",
        "How much interest do I earn on my checking account?",
        "Can you help me with a recent transaction?",
        "I need to check my statement for last month.",
    ]
    
    # Build vocabulary
    tokenizer.build_vocab(corpus, vocab_size=100)
    
    # Test encoding
    text = "What is my balance?"
    encoded = tokenizer.encode(text)
    print(f"Text: {text}")
    print(f"Encoded: {encoded}")
    
    # Test decoding
    decoded = tokenizer.decode(encoded)
    print(f"Decoded: {decoded}")
    
    print(f"\nVocabulary size: {tokenizer.vocab_size}")
