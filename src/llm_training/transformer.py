"""
Transformer Architecture from Scratch
Educational implementation to understand LLM fundamentals
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    """
    Multi-head self-attention mechanism
    
    This is the core of transformers - allows model to focus on different
    parts of the input sequence simultaneously.
    """
    
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads  # Dimension per head
        
        # Linear projections for Q, K, V
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        
        # Output projection
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
        # For visualization
        self.attention_weights = None
    
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            query: (batch_size, seq_len, d_model)
            key: (batch_size, seq_len, d_model)
            value: (batch_size, seq_len, d_model)
            mask: (batch_size, seq_len, seq_len) - mask for padding/future tokens
            
        Returns:
            output: (batch_size, seq_len, d_model)
        """
        batch_size, seq_len, _ = query.size()
        
        # 1. Linear projections and reshape to (batch, num_heads, seq_len, d_k)
        Q = self.w_q(query).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        
        # 2. Scaled dot-product attention
        # scores = Q @ K^T / sqrt(d_k)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # 3. Apply mask (for padding and causal masking)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        # 4. Softmax to get attention weights
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Store for visualization
        self.attention_weights = attention_weights.detach()
        
        # 5. Apply attention to values
        context = torch.matmul(attention_weights, V)
        
        # 6. Concatenate heads and apply output projection
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        output = self.w_o(context)
        
        return output


class FeedForward(nn.Module):
    """
    Position-wise feed-forward network
    
    Applied to each position separately and identically
    """
    
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, seq_len, d_model)
        Returns:
            output: (batch_size, seq_len, d_model)
        """
        # FFN(x) = max(0, xW1 + b1)W2 + b2
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


class TransformerBlock(nn.Module):
    """
    Single transformer block (decoder-only, like GPT)
    
    Contains:
    1. Masked multi-head self-attention
    2. Feed-forward network
    3. Layer normalization and residual connections
    """
    
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        
        # Multi-head attention
        self.attention = MultiHeadAttention(d_model, num_heads, dropout)
        
        # Feed-forward
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        
        # Layer normalization (pre-norm architecture like GPT)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: (batch_size, seq_len, d_model)
            mask: (batch_size, seq_len, seq_len)
        Returns:
            output: (batch_size, seq_len, d_model)
        """
        # Pre-norm + attention + residual
        attn_output = self.attention(self.ln1(x), self.ln1(x), self.ln1(x), mask)
        x = x + self.dropout(attn_output)
        
        # Pre-norm + feed-forward + residual
        ff_output = self.feed_forward(self.ln2(x))
        x = x + self.dropout(ff_output)
        
        return x


class BankingLLM(nn.Module):
    """
    Banking Domain Language Model
    
    A decoder-only transformer (GPT-style) trained on banking conversations
    and knowledge.
    
    Architecture:
    - Token + Position embeddings
    - N transformer blocks
    - Language modeling head
    """
    
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        num_heads: int = 8,
        num_layers: int = 6,
        d_ff: int = 2048,
        max_seq_len: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        
        # Token embeddings
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        
        # Positional embeddings (learned, like GPT)
        self.position_embedding = nn.Embedding(max_seq_len, d_model)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        # Final layer norm
        self.ln_f = nn.LayerNorm(d_model)
        
        # Language modeling head
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        
        # Tie weights between token embedding and output projection (weight tying)
        self.lm_head.weight = self.token_embedding.weight
        
        self.dropout = nn.Dropout(dropout)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize weights (following GPT-2 initialization)"""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        targets: Optional[torch.Tensor] = None
    ) -> tuple:
        """
        Args:
            input_ids: (batch_size, seq_len) - token IDs
            targets: (batch_size, seq_len) - target token IDs for training
            
        Returns:
            logits: (batch_size, seq_len, vocab_size)
            loss: scalar (if targets provided)
        """
        batch_size, seq_len = input_ids.size()
        
        # Create causal mask (prevent attending to future tokens)
        mask = torch.tril(torch.ones(seq_len, seq_len)).unsqueeze(0).unsqueeze(0)
        mask = mask.to(input_ids.device)
        
        # Token embeddings
        token_emb = self.token_embedding(input_ids)  # (batch, seq_len, d_model)
        
        # Position embeddings
        positions = torch.arange(0, seq_len, dtype=torch.long, device=input_ids.device)
        position_emb = self.position_embedding(positions)  # (seq_len, d_model)
        
        # Combine embeddings
        x = self.dropout(token_emb + position_emb)
        
        # Pass through transformer blocks
        for block in self.blocks:
            x = block(x, mask)
        
        # Final layer norm
        x = self.ln_f(x)
        
        # Language modeling head
        logits = self.lm_head(x)  # (batch, seq_len, vocab_size)
        
        # Calculate loss if targets provided
        loss = None
        if targets is not None:
            # Flatten for cross-entropy
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1  # Ignore padding tokens
            )
        
        return logits, loss
    
    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
    ) -> torch.Tensor:
        """
        Generate text autoregressively
        
        Args:
            input_ids: (batch_size, seq_len) - prompt token IDs
            max_new_tokens: number of tokens to generate
            temperature: sampling temperature (higher = more random)
            top_k: keep only top k tokens
            top_p: nucleus sampling threshold
            
        Returns:
            generated: (batch_size, seq_len + max_new_tokens)
        """
        self.eval()
        
        for _ in range(max_new_tokens):
            # Crop sequence if too long
            input_ids_cond = input_ids if input_ids.size(1) <= self.max_seq_len else input_ids[:, -self.max_seq_len:]
            
            # Forward pass
            logits, _ = self(input_ids_cond)
            
            # Get logits for last token
            logits = logits[:, -1, :] / temperature
            
            # Top-k sampling
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')
            
            # Top-p (nucleus) sampling
            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                
                # Remove tokens with cumulative probability above threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                logits[indices_to_remove] = float('-inf')
            
            # Sample from distribution
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Append to sequence
            input_ids = torch.cat([input_ids, next_token], dim=1)
        
        return input_ids
    
    def count_parameters(self) -> dict:
        """Count model parameters"""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'total': total,
            'trainable': trainable,
            'total_millions': total / 1e6,
            'trainable_millions': trainable / 1e6,
        }


def create_small_model(vocab_size: int) -> BankingLLM:
    """Create a small model for experimentation (similar to GPT-2 Small)"""
    return BankingLLM(
        vocab_size=vocab_size,
        d_model=512,
        num_heads=8,
        num_layers=6,
        d_ff=2048,
        max_seq_len=512,
        dropout=0.1,
    )


def create_medium_model(vocab_size: int) -> BankingLLM:
    """Create a medium model (similar to GPT-2 Medium)"""
    return BankingLLM(
        vocab_size=vocab_size,
        d_model=768,
        num_heads=12,
        num_layers=12,
        d_ff=3072,
        max_seq_len=1024,
        dropout=0.1,
    )


if __name__ == "__main__":
    # Quick test
    vocab_size = 10000
    model = create_small_model(vocab_size)
    
    # Print model info
    params = model.count_parameters()
    print(f"Model Parameters: {params['total_millions']:.2f}M")
    print(f"Trainable: {params['trainable_millions']:.2f}M")
    
    # Test forward pass
    batch_size = 2
    seq_len = 128
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    logits, _ = model(input_ids)
    print(f"\nInput shape: {input_ids.shape}")
    print(f"Output shape: {logits.shape}")
    
    # Test generation
    prompt = torch.randint(0, vocab_size, (1, 10))
    generated = model.generate(prompt, max_new_tokens=20, temperature=0.8, top_k=50)
    print(f"\nPrompt length: {prompt.shape[1]}")
    print(f"Generated length: {generated.shape[1]}")
