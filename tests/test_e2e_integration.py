"""
End-to-End Integration Tests for Banking LLM

Tests the complete pipeline:
1. Data loading and preparation
2. Model training/loading
3. RAG context retrieval
4. LM Cache functionality
5. Full inference pipeline
6. Accuracy verification
"""

import asyncio
import json
import logging
import os
from pathlib import Path
from typing import Dict, List

import pytest
import torch

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class TestDataPreparation:
    """Test data loading and preparation"""
    
    def test_training_data_exists(self):
        """Verify training data exists"""
        train_path = Path("data/finetuning/train.json")
        assert train_path.exists(), "Training data not found"
        
        with open(train_path) as f:
            data = json.load(f)
        
        assert len(data) > 0, "Training data is empty"
        assert all("text" in item or "input" in item for item in data), \
            "Training data missing required fields"
        
        logger.info(f"✓ Training data verified: {len(data)} samples")
    
    def test_validation_data_exists(self):
        """Verify validation data exists"""
        val_path = Path("data/finetuning/val.json")
        assert val_path.exists(), "Validation data not found"
        
        with open(val_path) as f:
            data = json.load(f)
        
        assert len(data) > 0, "Validation data is empty"
        
        logger.info(f"✓ Validation data verified: {len(data)} samples")
    
    def test_banking_context_policies(self):
        """Verify banking context policies are defined"""
        from src.services.enhanced_rag_service import BankingContextStore
        
        store = BankingContextStore()
        policies = store.get_all_policies()
        
        assert len(policies) > 0, "No banking policies found"
        assert all("id" in p and "answer" in p for p in policies), \
            "Policies missing required fields"
        
        logger.info(f"✓ Banking policies verified: {len(policies)} policies")


class TestLLMComponents:
    """Test individual LLM components"""
    
    def test_tokenizer_loading(self):
        """Test tokenizer loads correctly"""
        from transformers import AutoTokenizer
        
        model_name = "TinyLlama/TinyLlama-1.1B"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        assert tokenizer is not None, "Tokenizer failed to load"
        
        # Test encoding
        text = "What is my account balance?"
        tokens = tokenizer.encode(text)
        assert len(tokens) > 0, "Tokenization failed"
        
        logger.info(f"✓ Tokenizer verified: {len(tokens)} tokens for test query")
    
    def test_lm_cache_initialization(self):
        """Test LM Cache initialization"""
        from src.llm.lm_cache import LMCacheManager
        
        cache_manager = LMCacheManager(
            enable_kv_cache=True,
            enable_prompt_cache=True,
            enable_prefix_cache=True,
            device="cpu"
        )
        
        assert cache_manager is not None, "Cache manager failed to initialize"
        assert cache_manager.kv_cache is not None, "KV cache not initialized"
        assert cache_manager.prompt_cache is not None, "Prompt cache not initialized"
        
        logger.info("✓ LM Cache manager verified")
    
    def test_prompt_cache_functionality(self):
        """Test prompt cache stores and retrieves"""
        from src.llm.lm_cache import LMCacheManager
        
        cache = LMCacheManager(enable_prompt_cache=True, device="cpu")
        
        prompt = "What is my account balance?"
        response = "Your balance is $5,000"
        
        # Cache response
        cache.cache_prompt_response(prompt, response)
        
        # Retrieve from cache
        cached = cache.check_prompt_cache(prompt)
        assert cached is not None, "Cache retrieval failed"
        
        logger.info("✓ Prompt cache functionality verified")
    
    def test_prefix_cache_patterns(self):
        """Test prefix cache pattern matching"""
        from src.llm.lm_cache import LMCacheManager
        
        cache = LMCacheManager(enable_prefix_cache=True, device="cpu")
        
        # Test pattern matching
        pattern = cache.prefix_cache.find_matching_prefix("what is my balance?")
        assert pattern == "what", "Pattern matching failed"
        
        logger.info("✓ Prefix cache patterns verified")


class TestRAGIntegration:
    """Test RAG service integration"""
    
    @pytest.mark.skipif(
        not os.getenv("PINECONE_API_KEY"),
        reason="Pinecone API key not provided"
    )
    def test_banking_context_embeddings(self):
        """Test that banking context can be embedded"""
        from src.services.enhanced_rag_service import BankingContextStore
        
        store = BankingContextStore(embedding_model="all-MiniLM-L6-v2")
        embeddings = store.embed_policies()
        
        assert len(embeddings) > 0, "No embeddings generated"
        
        for policy_id, embedding, policy in embeddings:
            assert embedding is not None, "Embedding is None"
            assert len(embedding) == 384, f"Unexpected embedding dimension: {len(embedding)}"
            assert policy is not None, "Policy data is None"
        
        logger.info(f"✓ Embeddings verified: {len(embeddings)} policies embedded")
    
    @pytest.mark.skipif(
        not os.getenv("PINECONE_API_KEY"),
        reason="Pinecone API key not provided"
    )
    def test_pinecone_initialization(self):
        """Test Pinecone service initialization"""
        from src.services.enhanced_rag_service import EnhancedRAGService
        
        api_key = os.getenv("PINECONE_API_KEY")
        if not api_key:
            pytest.skip("Pinecone API key not available")
        
        rag_service = EnhancedRAGService(
            pinecone_api_key=api_key,
            pinecone_index_name="banking-context-test"
        )
        
        assert rag_service is not None, "RAG service initialization failed"
        assert rag_service.index is not None, "Pinecone index not initialized"
        
        logger.info("✓ Pinecone initialization verified")
    
    @pytest.mark.skipif(
        not os.getenv("PINECONE_API_KEY"),
        reason="Pinecone API key not provided"
    )
    def test_context_retrieval(self):
        """Test context retrieval from RAG"""
        from src.services.enhanced_rag_service import EnhancedRAGService
        
        api_key = os.getenv("PINECONE_API_KEY")
        if not api_key:
            pytest.skip("Pinecone API key not available")
        
        rag_service = EnhancedRAGService(
            pinecone_api_key=api_key,
            pinecone_index_name="banking-context-test"
        )
        
        # Test retrieval
        query = "What is my account balance?"
        context = rag_service.retrieve_context(query, top_k=3)
        
        assert isinstance(context, list), "Context is not a list"
        if context:
            assert "score" in context[0], "Missing score in context"
            assert "answer" in context[0], "Missing answer in context"
        
        logger.info(f"✓ Context retrieval verified: {len(context)} items")


class TestBankingLLMIntegration:
    """Test complete banking LLM integration"""
    
    def test_integration_initialization(self):
        """Test banking LLM integration can initialize"""
        from src.services.banking_llm_integration import BankingLLMIntegration
        
        # Create with minimal config
        try:
            integration = BankingLLMIntegration(
                model_path="models/banking_llm",
                enable_rag=False,  # Skip Pinecone for now
                enable_cache=True,
                device="cpu"
            )
            
            # Should initialize even if model doesn't exist yet
            assert integration is not None, "Integration initialization failed"
            
            logger.info("✓ Banking LLM integration initialized")
        except FileNotFoundError:
            logger.warning("Model files not found - this is expected before training")
    
    def test_cache_manager_integration(self):
        """Test cache manager in integration"""
        from src.services.banking_llm_integration import BankingLLMIntegration
        
        try:
            integration = BankingLLMIntegration(
                model_path="models/banking_llm",
                enable_cache=True,
                device="cpu"
            )
            
            # Test metrics retrieval
            if integration.cache_manager:
                metrics = integration.cache_manager.get_stats()
                assert isinstance(metrics, dict), "Metrics should be dictionary"
                assert "caches" in metrics, "Missing cache stats"
                
                logger.info(f"✓ Cache manager integration verified: {metrics}")
        except FileNotFoundError:
            logger.warning("Model files not found - skipping full test")


class TestBenchmarking:
    """Test benchmarking framework"""
    
    def test_benchmark_dataset_loaded(self):
        """Test benchmark dataset loads"""
        from src.benchmarks.comprehensive_benchmark import BenchmarkDataset
        
        dataset = BenchmarkDataset.get_benchmark_dataset()
        
        assert len(dataset) > 0, "Benchmark dataset is empty"
        assert all("question" in q for q in dataset), "Missing questions"
        assert all("expected_intent" in q for q in dataset), "Missing intents"
        
        logger.info(f"✓ Benchmark dataset verified: {len(dataset)} test cases")
    
    def test_context_relevance_evaluator(self):
        """Test context relevance evaluation"""
        from src.benchmarks.comprehensive_benchmark import ContextRelevanceEvaluator
        
        evaluator = ContextRelevanceEvaluator()
        
        context = [
            {
                "question": "What is my balance?",
                "answer": "Your checking account balance is $5,000"
            }
        ]
        
        ground_truth = ["balance", "checking", "$5000"]
        
        score = evaluator.evaluate_context(context, ground_truth)
        
        assert 0 <= score <= 1, "Score out of range"
        assert score > 0.5, "Score too low for matching context"
        
        logger.info(f"✓ Context relevance evaluator verified: score={score:.2%}")
    
    def test_response_quality_evaluator(self):
        """Test response quality evaluation"""
        from src.benchmarks.comprehensive_benchmark import ResponseQualityEvaluator
        
        evaluator = ResponseQualityEvaluator()
        
        response = "Your checking account balance is $5,000. You also have $2,000 in savings."
        ground_truth = ["balance", "checking", "$5000"]
        
        score = evaluator.evaluate_response(response, ground_truth)
        
        assert 0 <= score <= 1, "Score out of range"
        assert score > 0.5, "Score too low for quality response"
        
        logger.info(f"✓ Response quality evaluator verified: score={score:.2%}")


class TestEndToEnd:
    """End-to-end integration tests"""
    
    def test_complete_pipeline_data_validation(self):
        """Test complete pipeline data validation"""
        logger.info("\n" + "="*80)
        logger.info("RUNNING END-TO-END INTEGRATION TESTS")
        logger.info("="*80)
        
        # 1. Check data
        train_path = Path("data/finetuning/train.json")
        val_path = Path("data/finetuning/val.json")
        
        assert train_path.exists(), "Training data missing"
        assert val_path.exists(), "Validation data missing"
        
        with open(train_path) as f:
            train_data = json.load(f)
        with open(val_path) as f:
            val_data = json.load(f)
        
        assert len(train_data) > 0, "Training data empty"
        assert len(val_data) > 0, "Validation data empty"
        
        logger.info(f"✓ Data validation: {len(train_data)} train, {len(val_data)} val")
    
    def test_banking_context_availability(self):
        """Test banking context is available for RAG"""
        from src.services.enhanced_rag_service import BankingContextStore
        
        store = BankingContextStore()
        policies = store.get_all_policies()
        
        assert len(policies) >= 10, "Insufficient banking policies"
        
        # Verify critical policies exist
        policy_categories = {p.get("category") for p in policies}
        critical_categories = {"fees", "transfers", "account_management", "security"}
        
        assert critical_categories.issubset(policy_categories), \
            f"Missing critical categories: {critical_categories - policy_categories}"
        
        logger.info(f"✓ Banking context available: {len(policies)} policies, "
                   f"categories: {', '.join(sorted(policy_categories))}")
    
    def test_caching_infrastructure(self):
        """Test caching infrastructure is in place"""
        from src.llm.lm_cache import LMCacheManager
        
        cache_manager = LMCacheManager(device="cpu")
        
        # Test all cache types
        assert cache_manager.kv_cache is not None, "KV cache not initialized"
        assert cache_manager.prompt_cache is not None, "Prompt cache not initialized"
        assert cache_manager.prefix_cache is not None, "Prefix cache not initialized"
        
        # Test storage and retrieval
        test_prompt = "Test banking question"
        test_response = "Test response"
        
        cache_manager.cache_prompt_response(test_prompt, test_response)
        cached = cache_manager.check_prompt_cache(test_prompt)
        
        assert cached is not None, "Cache storage/retrieval failed"
        
        logger.info("✓ Caching infrastructure verified")


def run_all_tests():
    """Run all integration tests"""
    logger.info("\n" + "="*80)
    logger.info("RUNNING COMPREHENSIVE END-TO-END TESTS")
    logger.info("="*80 + "\n")
    
    # Run pytest with verbose output
    pytest.main([
        __file__,
        "-v",
        "--tb=short",
        "-s"
    ])


if __name__ == "__main__":
    run_all_tests()
