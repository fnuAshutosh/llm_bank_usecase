"""
End-to-End LLM Integration Service

Combines:
1. Custom Fine-tuned LLM (TinyLlama + LoRA)
2. Pinecone RAG for context retrieval
3. LM Cache for fast inference
4. Real banking data (no mocks)
"""

import logging
import time
from typing import Dict, List, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.llm.lm_cache import LMCacheManager
from src.llm_training.lora_trainer import BankingLLMTrainer
from src.services.enhanced_rag_service import EnhancedRAGService

logger = logging.getLogger(__name__)


class BankingLLMIntegration:
    """Complete end-to-end banking LLM integration"""
    
    def __init__(self,
                 model_path: str = "models/banking_llm",
                 pinecone_api_key: str = "",
                 enable_rag: bool = True,
                 enable_cache: bool = True,
                 device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        """
        Initialize end-to-end integration
        
        Args:
            model_path: Path to fine-tuned model
            pinecone_api_key: Pinecone API key for RAG
            enable_rag: Enable RAG retrieval
            enable_cache: Enable LM caching
            device: Device for inference
        """
        self.model_path = model_path
        self.pinecone_api_key = pinecone_api_key
        self.device = device
        self.enable_rag = enable_rag
        self.enable_cache = enable_cache
        
        # Components
        self.tokenizer = None
        self.model = None
        self.rag_service = None
        self.cache_manager = None
        
        # Metrics
        self.inference_times = []
        self.cache_hits = 0
        self.cache_misses = 0
        
        # Initialize all components
        self._initialize()
    
    def _initialize(self):
        """Initialize all components"""
        logger.info("Initializing Banking LLM Integration...")
        
        # 1. Load tokenizer and model
        self._load_model()
        
        # 2. Initialize RAG if enabled
        if self.enable_rag and self.pinecone_api_key:
            try:
                self.rag_service = EnhancedRAGService(
                    pinecone_api_key=self.pinecone_api_key
                )
                logger.info("✓ RAG service initialized")
            except Exception as e:
                logger.warning(f"RAG initialization failed: {e}. Continuing without RAG.")
                self.enable_rag = False
        
        # 3. Initialize cache manager if enabled
        if self.enable_cache:
            self.cache_manager = LMCacheManager(
                enable_kv_cache=True,
                enable_prompt_cache=True,
                enable_prefix_cache=True,
                device=self.device
            )
            logger.info("✓ Cache manager initialized")
        
        logger.info("✓ Banking LLM Integration fully initialized")
    
    def _load_model(self):
        """Load fine-tuned model and tokenizer"""
        try:
            logger.info(f"Loading model from {self.model_path}...")
            
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                device_map=self.device,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                trust_remote_code=True
            )
            
            self.model.eval()
            logger.info("✓ Model loaded successfully")
        
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def _generate_with_llm(self, 
                          prompt: str,
                          max_new_tokens: int = 256,
                          temperature: float = 0.7) -> str:
        """
        Generate response from LLM
        
        Args:
            prompt: Input prompt
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            
        Returns:
            Generated response
        """
        try:
            # Tokenize
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                max_length=512,
                truncation=True
            ).to(self.device)
            
            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=0.95,
                    top_k=50,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id
                )
            
            # Decode
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Remove prompt from output
            if response.startswith(prompt):
                response = response[len(prompt):]
            
            return response.strip()
        
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            raise
    
    def process_query(self,
                     customer_query: str,
                     customer_id: str = "UNKNOWN",
                     session_id: str = "UNKNOWN",
                     use_rag: Optional[bool] = None,
                     use_cache: Optional[bool] = None,
                     max_new_tokens: int = 256) -> Dict:
        """
        Process customer query end-to-end
        
        Args:
            customer_query: Customer question
            customer_id: Customer ID
            session_id: Session ID
            use_rag: Override RAG setting
            use_cache: Override cache setting
            max_new_tokens: Max tokens to generate
            
        Returns:
            Complete response with metrics
        """
        start_time = time.time()
        
        use_rag = use_rag if use_rag is not None else self.enable_rag
        use_cache = use_cache if use_cache is not None else self.enable_cache
        
        logger.info(f"Processing query from {customer_id}: {customer_query[:50]}...")
        
        result = {
            "customer_id": customer_id,
            "session_id": session_id,
            "query": customer_query,
            "timestamp": start_time,
            "metrics": {
                "used_cache": False,
                "used_rag": False,
                "inference_time": 0,
                "total_time": 0,
                "context_quality": 0.0
            }
        }
        
        # 1. Check cache
        if use_cache and self.cache_manager:
            cached = self.cache_manager.check_prompt_cache(customer_query)
            if cached:
                logger.info("Cache HIT")
                self.cache_hits += 1
                result["response"] = cached["response"]
                result["metrics"]["used_cache"] = True
                result["metrics"]["total_time"] = time.time() - start_time
                return result
            else:
                self.cache_misses += 1
        
        # 2. Retrieve context via RAG
        augmented_prompt = customer_query
        if use_rag and self.rag_service:
            try:
                context = self.rag_service.retrieve_context(customer_query, top_k=3)
                if context:
                    augmented_prompt = self.rag_service.augment_prompt(
                        customer_query,
                        context
                    )
                    result["metrics"]["used_rag"] = True
                    result["metrics"]["context_quality"] = sum(c["score"] for c in context) / len(context)
                    result["retrieved_context"] = context
                    logger.info(f"Retrieved {len(context)} context items")
            except Exception as e:
                logger.warning(f"RAG retrieval failed: {e}")
        
        # 3. Generate response
        inference_start = time.time()
        try:
            response = self._generate_with_llm(
                augmented_prompt,
                max_new_tokens=max_new_tokens
            )
            result["response"] = response
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            result["response"] = f"Error: Unable to process query. {str(e)}"
        
        inference_time = time.time() - inference_start
        result["metrics"]["inference_time"] = inference_time
        self.inference_times.append(inference_time)
        
        # 4. Cache the response
        if use_cache and self.cache_manager:
            try:
                self.cache_manager.cache_prompt_response(
                    customer_query,
                    result["response"],
                    metadata={
                        "context_quality": result["metrics"]["context_quality"],
                        "inference_time": inference_time,
                        "used_rag": result["metrics"]["used_rag"]
                    }
                )
            except Exception as e:
                logger.warning(f"Failed to cache response: {e}")
        
        result["metrics"]["total_time"] = time.time() - start_time
        
        logger.info(f"Query processed in {result['metrics']['total_time']:.2f}s")
        
        return result
    
    def batch_process(self, queries: List[Tuple[str, str, str]]) -> List[Dict]:
        """
        Process batch of queries
        
        Args:
            queries: List of (customer_query, customer_id, session_id) tuples
            
        Returns:
            List of results
        """
        logger.info(f"Processing batch of {len(queries)} queries...")
        
        results = []
        for query, customer_id, session_id in queries:
            result = self.process_query(
                customer_query=query,
                customer_id=customer_id,
                session_id=session_id
            )
            results.append(result)
        
        return results
    
    def get_metrics(self) -> Dict:
        """Get performance metrics"""
        avg_inference_time = sum(self.inference_times) / len(self.inference_times) \
            if self.inference_times else 0
        
        total_requests = self.cache_hits + self.cache_misses
        cache_hit_rate = (self.cache_hits / total_requests * 100) if total_requests > 0 else 0
        
        metrics = {
            "total_queries": total_requests,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "cache_hit_rate": cache_hit_rate,
            "avg_inference_time": avg_inference_time,
            "model_loaded": self.model is not None,
            "rag_enabled": self.enable_rag and self.rag_service is not None,
            "cache_enabled": self.enable_cache and self.cache_manager is not None,
        }
        
        if self.cache_manager:
            metrics["cache_stats"] = self.cache_manager.get_stats()
        
        return metrics


def initialize_banking_llm(
    model_path: str = "models/banking_llm",
    pinecone_api_key: str = "",
    enable_rag: bool = True,
    enable_cache: bool = True
) -> BankingLLMIntegration:
    """
    Initialize banking LLM integration
    
    Args:
        model_path: Path to fine-tuned model
        pinecone_api_key: Pinecone API key
        enable_rag: Enable RAG
        enable_cache: Enable caching
        
    Returns:
        Initialized integration instance
    """
    try:
        integration = BankingLLMIntegration(
            model_path=model_path,
            pinecone_api_key=pinecone_api_key,
            enable_rag=enable_rag,
            enable_cache=enable_cache
        )
        return integration
    except Exception as e:
        logger.error(f"Failed to initialize Banking LLM: {e}")
        raise
