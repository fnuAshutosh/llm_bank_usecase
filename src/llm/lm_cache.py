"""
LM-Cache Implementation for Fast Inference

Implements KV-Cache management to dramatically speed up inference for banking LLM.
Uses attention cache reuse to avoid redundant computations.
"""

import hashlib
import logging
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class KVCache:
    """Key-Value Cache for Transformer attention layers"""
    
    def __init__(self, 
                 max_batch_size: int = 32,
                 max_seq_len: int = 2048,
                 num_heads: int = 8,
                 head_dim: int = 64,
                 num_layers: int = 12,
                 device: str = "cpu"):
        """
        Initialize KV Cache
        
        Args:
            max_batch_size: Maximum batch size
            max_seq_len: Maximum sequence length
            num_heads: Number of attention heads
            head_dim: Dimension per head
            num_layers: Number of transformer layers
            device: Device to store cache on (cpu/cuda)
        """
        self.max_batch_size = max_batch_size
        self.max_seq_len = max_seq_len
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.num_layers = num_layers
        self.device = device
        
        # Initialize cache for all layers
        # Shape: [num_layers, max_batch_size, num_heads, max_seq_len, head_dim]
        self.k_cache = None
        self.v_cache = None
        self.cache_seq_lens = None
        
        self._init_cache()
    
    def _init_cache(self):
        """Initialize cache tensors"""
        cache_shape = (
            self.num_layers,
            self.max_batch_size,
            self.num_heads,
            self.max_seq_len,
            self.head_dim
        )
        
        self.k_cache = torch.zeros(cache_shape, dtype=torch.float16, device=self.device)
        self.v_cache = torch.zeros(cache_shape, dtype=torch.float16, device=self.device)
        self.cache_seq_lens = torch.zeros(self.max_batch_size, dtype=torch.long, device=self.device)
    
    def update(self, 
               layer_idx: int,
               k: torch.Tensor,
               v: torch.Tensor,
               batch_indices: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Update cache with new K, V values
        
        Args:
            layer_idx: Layer index
            k: Key tensor [batch_size, num_heads, seq_len, head_dim]
            v: Value tensor [batch_size, num_heads, seq_len, head_dim]
            batch_indices: Optional batch indices for selective update
            
        Returns:
            Updated full K, V caches
        """
        if batch_indices is None:
            batch_indices = torch.arange(k.shape[0], device=self.device)
        
        batch_size = k.shape[0]
        seq_len = k.shape[2]
        
        # Update cache
        for i, batch_idx in enumerate(batch_indices):
            current_pos = self.cache_seq_lens[batch_idx]
            self.k_cache[layer_idx, batch_idx, :, current_pos:current_pos + seq_len] = k[i]
            self.v_cache[layer_idx, batch_idx, :, current_pos:current_pos + seq_len] = v[i]
        
        # Update sequence lengths
        self.cache_seq_lens[batch_indices] += seq_len
        
        # Return full cached K, V
        k_cached = self.k_cache[layer_idx, batch_indices]
        v_cached = self.v_cache[layer_idx, batch_indices]
        
        return k_cached, v_cached
    
    def get_cached_kv(self, layer_idx: int, batch_indices: Optional[torch.Tensor] = None):
        """Get cached K, V values"""
        if batch_indices is None:
            return self.k_cache[layer_idx], self.v_cache[layer_idx]
        return self.k_cache[layer_idx, batch_indices], self.v_cache[layer_idx, batch_indices]
    
    def reset(self):
        """Reset cache"""
        self.k_cache.zero_()
        self.v_cache.zero_()
        self.cache_seq_lens.zero_()
    
    def reset_batch(self, batch_indices: torch.Tensor):
        """Reset cache for specific batches"""
        self.k_cache[:, batch_indices].zero_()
        self.v_cache[:, batch_indices].zero_()
        self.cache_seq_lens[batch_indices] = 0


class PromptCache:
    """Semantic prompt caching for banking questions"""
    
    def __init__(self, max_cache_size: int = 1000):
        """
        Initialize semantic cache
        
        Args:
            max_cache_size: Maximum number of cached prompts
        """
        self.max_cache_size = max_cache_size
        self.cache: Dict[str, Dict] = {}
        self.access_count: Dict[str, int] = {}
    
    def _hash_prompt(self, prompt: str) -> str:
        """Create hash of prompt for lookup"""
        return hashlib.sha256(prompt.encode()).hexdigest()
    
    def get(self, prompt: str) -> Optional[Dict]:
        """
        Get cached response if available
        
        Args:
            prompt: Input prompt
            
        Returns:
            Cached response or None
        """
        prompt_hash = self._hash_prompt(prompt)
        
        if prompt_hash in self.cache:
            self.access_count[prompt_hash] += 1
            logger.debug(f"Cache hit for prompt (access #{self.access_count[prompt_hash]})")
            return self.cache[prompt_hash]
        
        return None
    
    def put(self, prompt: str, response: Dict, metadata: Optional[Dict] = None) -> bool:
        """
        Cache response for prompt
        
        Args:
            prompt: Input prompt
            response: LLM response
            metadata: Optional metadata (context quality, accuracy score, etc)
            
        Returns:
            True if cached, False if cache full
        """
        if len(self.cache) >= self.max_cache_size:
            # Evict least accessed item
            if self.access_count:
                least_accessed = min(self.access_count, key=self.access_count.get)
                del self.cache[least_accessed]
                del self.access_count[least_accessed]
                logger.debug(f"Evicted cache entry: {least_accessed[:16]}...")
            else:
                logger.warning("Cache full and no items to evict")
                return False
        
        prompt_hash = self._hash_prompt(prompt)
        self.cache[prompt_hash] = {
            "response": response,
            "metadata": metadata or {},
            "hits": 0
        }
        self.access_count[prompt_hash] = 0
        
        logger.debug(f"Cached response for prompt (cache size: {len(self.cache)}/{self.max_cache_size})")
        return True
    
    def clear(self):
        """Clear all cache"""
        self.cache.clear()
        self.access_count.clear()
        logger.info("Prompt cache cleared")
    
    def stats(self) -> Dict:
        """Get cache statistics"""
        total_hits = sum(self.access_count.values())
        return {
            "cache_size": len(self.cache),
            "max_size": self.max_cache_size,
            "total_hits": total_hits,
            "hit_rate": total_hits / (total_hits + 1) if total_hits > 0 else 0
        }


class PrefixCache:
    """Cache for common banking question prefixes"""
    
    def __init__(self):
        """Initialize prefix cache with banking-specific patterns"""
        self.prefixes = {
            "what": "What is",
            "how": "How do I",
            "can": "Can I",
            "transfer": "transfer",
            "balance": "balance",
            "loan": "loan",
            "credit": "credit",
            "account": "account",
            "fee": "fee",
            "rate": "rate"
        }
        
        # Cached responses for common patterns
        self.pattern_cache: Dict[str, List[str]] = {}
    
    def find_matching_prefix(self, text: str) -> Optional[str]:
        """Find if text matches known prefix pattern"""
        text_lower = text.lower()
        
        for prefix, full_text in self.prefixes.items():
            if text_lower.startswith(prefix):
                return prefix
        
        return None
    
    def cache_pattern_response(self, pattern: str, response: str):
        """Cache response for pattern"""
        if pattern not in self.pattern_cache:
            self.pattern_cache[pattern] = []
        self.pattern_cache[pattern].append(response)
    
    def get_pattern_responses(self, pattern: str) -> List[str]:
        """Get cached responses for pattern"""
        return self.pattern_cache.get(pattern, [])


class LMCacheManager:
    """Unified manager for all caching strategies"""
    
    def __init__(self,
                 enable_kv_cache: bool = True,
                 enable_prompt_cache: bool = True,
                 enable_prefix_cache: bool = True,
                 device: str = "cpu"):
        """
        Initialize cache manager
        
        Args:
            enable_kv_cache: Enable KV cache
            enable_prompt_cache: Enable prompt caching
            enable_prefix_cache: Enable prefix caching
            device: Device for caching (cpu/cuda)
        """
        self.enable_kv_cache = enable_kv_cache
        self.enable_prompt_cache = enable_prompt_cache
        self.enable_prefix_cache = enable_prefix_cache
        self.device = device
        
        # Initialize caches
        self.kv_cache = KVCache(device=device) if enable_kv_cache else None
        self.prompt_cache = PromptCache() if enable_prompt_cache else None
        self.prefix_cache = PrefixCache() if enable_prefix_cache else None
        
        logger.info(f"LMCacheManager initialized - KV: {enable_kv_cache}, "
                   f"Prompt: {enable_prompt_cache}, Prefix: {enable_prefix_cache}")
    
    def check_prompt_cache(self, prompt: str) -> Optional[Dict]:
        """Check if prompt response is cached"""
        if not self.enable_prompt_cache:
            return None
        return self.prompt_cache.get(prompt)
    
    def cache_prompt_response(self, prompt: str, response: Dict, metadata: Optional[Dict] = None):
        """Cache a prompt-response pair"""
        if not self.enable_prompt_cache:
            return
        self.prompt_cache.put(prompt, response, metadata)
    
    def check_prefix_pattern(self, text: str) -> Optional[List[str]]:
        """Check if text matches cached prefix pattern"""
        if not self.enable_prefix_cache:
            return None
        pattern = self.prefix_cache.find_matching_prefix(text)
        if pattern:
            return self.prefix_cache.get_pattern_responses(pattern)
        return None
    
    def get_stats(self) -> Dict:
        """Get comprehensive cache statistics"""
        stats = {"caches": {}}
        
        if self.prompt_cache:
            stats["caches"]["prompt"] = self.prompt_cache.stats()
        
        if self.prefix_cache:
            stats["caches"]["prefix"] = {
                "patterns": len(self.prefix_cache.pattern_cache),
                "cached_responses": sum(len(v) for v in self.prefix_cache.pattern_cache.values())
            }
        
        if self.kv_cache:
            stats["caches"]["kv"] = {
                "max_batch_size": self.kv_cache.max_batch_size,
                "max_seq_len": self.kv_cache.max_seq_len,
                "device": self.device
            }
        
        return stats
    
    def clear_all(self):
        """Clear all caches"""
        if self.prompt_cache:
            self.prompt_cache.clear()
        if self.kv_cache:
            self.kv_cache.reset()
        logger.info("All caches cleared")
