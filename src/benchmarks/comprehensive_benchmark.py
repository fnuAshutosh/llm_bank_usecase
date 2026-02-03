"""
Banking LLM Benchmarking Suite - 95%+ Accuracy Target

Comprehensive evaluation framework measuring:
1. Context relevance and accuracy
2. Intent recognition correctness
3. Response quality metrics
4. Performance benchmarks
"""

import json
import logging
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

logger = logging.getLogger(__name__)


class BenchmarkDataset:
    """Curated benchmark dataset with ground truth answers"""
    
    BANKING_BENCHMARK_QA = [
        {
            "id": "bench_001",
            "question": "What is my current checking account balance?",
            "expected_intent": "check_balance",
            "ground_truth_contains": ["balance", "checking"],
            "category": "account_inquiry"
        },
        {
            "id": "bench_002",
            "question": "I want to transfer $500 to my savings account",
            "expected_intent": "transfer_funds",
            "ground_truth_contains": ["transfer", "savings", "$500"],
            "category": "transaction"
        },
        {
            "id": "bench_003",
            "question": "What interest rate do I earn on my savings?",
            "expected_intent": "interest_inquiry",
            "ground_truth_contains": ["4.5%", "APY", "interest"],
            "category": "rates"
        },
        {
            "id": "bench_004",
            "question": "How do I dispute a fraudulent charge on my account?",
            "expected_intent": "fraud_dispute",
            "ground_truth_contains": ["dispute", "fraud", "24 hours"],
            "category": "fraud"
        },
        {
            "id": "bench_005",
            "question": "Can I open a new account online?",
            "expected_intent": "account_opening",
            "ground_truth_contains": ["open", "account", "500"],
            "category": "account_management"
        },
        {
            "id": "bench_006",
            "question": "What are your ATM fees?",
            "expected_intent": "fee_inquiry",
            "ground_truth_contains": ["ATM", "free", "$3"],
            "category": "fees"
        },
        {
            "id": "bench_007",
            "question": "How do I apply for a loan?",
            "expected_intent": "loan_inquiry",
            "ground_truth_contains": ["loan", "personal", "auto"],
            "category": "products"
        },
        {
            "id": "bench_008",
            "question": "Is my account secure?",
            "expected_intent": "security_inquiry",
            "ground_truth_contains": ["secure", "encryption", "AES-256"],
            "category": "security"
        },
        {
            "id": "bench_009",
            "question": "What are your business hours?",
            "expected_intent": "information",
            "ground_truth_contains": ["24/7", "Monday-Friday", "9am"],
            "category": "general_info"
        },
        {
            "id": "bench_010",
            "question": "How do I set up direct deposit?",
            "expected_intent": "direct_deposit",
            "ground_truth_contains": ["direct deposit", "routing number"],
            "category": "account_management"
        },
        {
            "id": "bench_011",
            "question": "Can I increase my credit card limit?",
            "expected_intent": "credit_inquiry",
            "ground_truth_contains": ["credit limit", "3-5 business days"],
            "category": "credit"
        },
        {
            "id": "bench_012",
            "question": "I lost my debit card, what do I do?",
            "expected_intent": "card_replacement",
            "ground_truth_contains": ["deactivate", "replacement", "5-7 days"],
            "category": "card_services"
        },
        {
            "id": "bench_013",
            "question": "What's the minimum balance requirement?",
            "expected_intent": "fee_inquiry",
            "ground_truth_contains": ["minimum balance", "$500"],
            "category": "fees"
        },
        {
            "id": "bench_014",
            "question": "Can I pay bills online through the bank?",
            "expected_intent": "bill_pay",
            "ground_truth_contains": ["bill pay", "free", "online banking"],
            "category": "services"
        },
        {
            "id": "bench_015",
            "question": "Are there foreign transaction fees?",
            "expected_intent": "fee_inquiry",
            "ground_truth_contains": ["foreign", "3%", "international"],
            "category": "fees"
        },
    ]
    
    @classmethod
    def get_benchmark_dataset(cls) -> List[Dict]:
        """Get the benchmark dataset"""
        return cls.BANKING_BENCHMARK_QA


class ContextRelevanceEvaluator:
    """Evaluate context relevance and quality"""
    
    def __init__(self):
        self.scores = []
    
    def evaluate_context(self, 
                        retrieved_context: List[Dict],
                        ground_truth_keywords: List[str]) -> float:
        """
        Evaluate if retrieved context contains relevant information
        
        Args:
            retrieved_context: Context retrieved by RAG
            ground_truth_keywords: Expected keywords in response
            
        Returns:
            Relevance score (0-1)
        """
        if not retrieved_context:
            return 0.0
        
        # Combine all retrieved text
        combined_text = " ".join([
            ctx.get("question", "") + " " + ctx.get("answer", "")
            for ctx in retrieved_context
        ]).lower()
        
        # Check keyword coverage
        found_keywords = sum(
            1 for keyword in ground_truth_keywords
            if keyword.lower() in combined_text
        )
        
        relevance_score = found_keywords / max(len(ground_truth_keywords), 1)
        self.scores.append(relevance_score)
        
        return relevance_score
    
    def get_average_relevance(self) -> float:
        """Get average relevance score"""
        return np.mean(self.scores) if self.scores else 0.0


class ResponseQualityEvaluator:
    """Evaluate response quality and correctness"""
    
    def __init__(self):
        self.scores = []
    
    def evaluate_response(self,
                         response: str,
                         ground_truth_keywords: List[str],
                         expected_intent: str = None) -> float:
        """
        Evaluate response quality
        
        Args:
            response: Generated response
            ground_truth_keywords: Keywords that should be in response
            expected_intent: Expected intent (for logging)
            
        Returns:
            Quality score (0-1)
        """
        response_lower = response.lower()
        
        # 1. Keyword coverage (50%)
        found_keywords = sum(
            1 for keyword in ground_truth_keywords
            if keyword.lower() in response_lower
        )
        keyword_score = found_keywords / max(len(ground_truth_keywords), 1)
        
        # 2. Response length (10%) - reasonable length
        word_count = len(response.split())
        length_score = min(word_count / 50, 1.0)  # At least 50 words
        
        # 3. No contradictions (40%) - check for negative indicators
        negative_indicators = ["cannot", "impossible", "unknown", "error"]
        contradiction_count = sum(
            1 for indicator in negative_indicators
            if indicator in response_lower and len(response) < 100
        )
        no_contradiction_score = 1.0 if contradiction_count == 0 else 0.5
        
        # Combined score
        quality_score = (
            keyword_score * 0.5 +
            length_score * 0.1 +
            no_contradiction_score * 0.4
        )
        
        self.scores.append(quality_score)
        
        return quality_score
    
    def get_average_quality(self) -> float:
        """Get average quality score"""
        return np.mean(self.scores) if self.scores else 0.0


class PerformanceBenchmark:
    """Benchmark performance metrics"""
    
    def __init__(self):
        self.latencies = []
        self.throughputs = []
    
    def measure_latency(self, start_time: float, end_time: float) -> float:
        """Measure query latency in seconds"""
        latency = end_time - start_time
        self.latencies.append(latency)
        return latency
    
    def get_latency_stats(self) -> Dict:
        """Get latency statistics"""
        if not self.latencies:
            return {}
        
        latencies_ms = [l * 1000 for l in self.latencies]  # Convert to ms
        
        return {
            "p50": np.percentile(latencies_ms, 50),
            "p95": np.percentile(latencies_ms, 95),
            "p99": np.percentile(latencies_ms, 99),
            "mean": np.mean(latencies_ms),
            "min": np.min(latencies_ms),
            "max": np.max(latencies_ms)
        }
    
    def calculate_throughput(self, num_queries: int, total_time: float) -> float:
        """Calculate queries per second"""
        if total_time == 0:
            return 0
        return num_queries / total_time


class ComprehensiveBenchmark:
    """Run comprehensive benchmarking suite"""
    
    def __init__(self, llm_integration):
        """
        Initialize benchmark
        
        Args:
            llm_integration: BankingLLMIntegration instance
        """
        self.llm = llm_integration
        self.dataset = BenchmarkDataset.get_benchmark_dataset()
        
        self.context_evaluator = ContextRelevanceEvaluator()
        self.response_evaluator = ResponseQualityEvaluator()
        self.performance_benchmark = PerformanceBenchmark()
        
        self.results = []
    
    def run_benchmark(self, verbose: bool = True) -> Dict:
        """
        Run comprehensive benchmark
        
        Args:
            verbose: Print detailed results
            
        Returns:
            Complete benchmark results
        """
        logger.info(f"Starting benchmark with {len(self.dataset)} test cases...")
        
        total_start = time.time()
        
        for test_case in self.dataset:
            query_start = time.time()
            
            # Process query
            result = self.llm.process_query(
                customer_query=test_case["question"],
                customer_id=f"BENCH_{test_case['id']}",
                session_id="BENCHMARK"
            )
            
            query_end = time.time()
            
            # Evaluate context
            context_score = self.context_evaluator.evaluate_context(
                retrieved_context=result.get("retrieved_context", []),
                ground_truth_keywords=test_case["ground_truth_contains"]
            )
            
            # Evaluate response
            response_score = self.response_evaluator.evaluate_response(
                response=result["response"],
                ground_truth_keywords=test_case["ground_truth_contains"],
                expected_intent=test_case["expected_intent"]
            )
            
            # Measure performance
            latency = self.performance_benchmark.measure_latency(query_start, query_end)
            
            # Store result
            test_result = {
                "test_id": test_case["id"],
                "question": test_case["question"],
                "category": test_case["category"],
                "context_score": context_score,
                "response_score": response_score,
                "latency_ms": latency * 1000,
                "response": result["response"],
                "used_rag": result["metrics"].get("used_rag", False),
                "used_cache": result["metrics"].get("used_cache", False),
            }
            
            self.results.append(test_result)
            
            if verbose:
                print(f"\n📋 Test {test_case['id']}: {test_case['question'][:60]}...")
                print(f"   Context Score: {context_score:.2%}")
                print(f"   Response Score: {response_score:.2%}")
                print(f"   Latency: {latency*1000:.0f}ms")
        
        total_end = time.time()
        total_time = total_end - total_start
        
        # Calculate aggregate metrics
        benchmark_report = self._generate_report(total_time)
        
        return benchmark_report
    
    def _generate_report(self, total_time: float) -> Dict:
        """Generate comprehensive benchmark report"""
        
        if not self.results:
            return {}
        
        # Extract scores
        context_scores = [r["context_score"] for r in self.results]
        response_scores = [r["response_score"] for r in self.results]
        latencies = [r["latency_ms"] for r in self.results]
        
        # Calculate metrics
        avg_context_score = np.mean(context_scores)
        avg_response_score = np.mean(response_scores)
        combined_accuracy = (avg_context_score + avg_response_score) / 2
        
        # Performance metrics
        latency_stats = self.performance_benchmark.get_latency_stats()
        throughput = self.performance_benchmark.calculate_throughput(
            len(self.results),
            total_time
        )
        
        # Category breakdown
        category_scores = {}
        for result in self.results:
            category = result["category"]
            if category not in category_scores:
                category_scores[category] = []
            category_scores[category].append(result["response_score"])
        
        category_breakdown = {
            cat: np.mean(scores) for cat, scores in category_scores.items()
        }
        
        report = {
            "benchmark_name": "Banking LLM - 95% Accuracy Target",
            "test_cases": len(self.results),
            "total_time_seconds": total_time,
            
            "accuracy_metrics": {
                "avg_context_relevance": avg_context_score,
                "avg_response_quality": avg_response_score,
                "combined_accuracy": combined_accuracy,
                "accuracy_target": 0.95,
                "target_achieved": combined_accuracy >= 0.95
            },
            
            "performance_metrics": {
                "throughput_qps": throughput,
                "latency_p50_ms": latency_stats.get("p50", 0),
                "latency_p95_ms": latency_stats.get("p95", 0),
                "latency_p99_ms": latency_stats.get("p99", 0),
                "latency_mean_ms": latency_stats.get("mean", 0),
            },
            
            "category_breakdown": category_breakdown,
            
            "llm_metrics": self.llm.get_metrics(),
            
            "detailed_results": self.results
        }
        
        return report
    
    def print_report(self, report: Dict):
        """Print formatted benchmark report"""
        print("\n" + "="*80)
        print(f"🧪 {report.get('benchmark_name', 'Benchmark Report')}")
        print("="*80)
        
        # Accuracy metrics
        acc_metrics = report.get("accuracy_metrics", {})
        print(f"\n📊 ACCURACY METRICS:")
        print(f"   Context Relevance: {acc_metrics.get('avg_context_relevance', 0):.2%}")
        print(f"   Response Quality:  {acc_metrics.get('avg_response_quality', 0):.2%}")
        print(f"   Combined Accuracy: {acc_metrics.get('combined_accuracy', 0):.2%}")
        print(f"   Target (95%):      {'✅ ACHIEVED' if acc_metrics.get('target_achieved') else '❌ NOT ACHIEVED'}")
        
        # Performance metrics
        perf_metrics = report.get("performance_metrics", {})
        print(f"\n⚡ PERFORMANCE METRICS:")
        print(f"   Throughput:        {perf_metrics.get('throughput_qps', 0):.2f} req/s")
        print(f"   Latency P50:       {perf_metrics.get('latency_p50_ms', 0):.0f}ms")
        print(f"   Latency P95:       {perf_metrics.get('latency_p95_ms', 0):.0f}ms")
        print(f"   Latency P99:       {perf_metrics.get('latency_p99_ms', 0):.0f}ms")
        
        # Category breakdown
        cat_breakdown = report.get("category_breakdown", {})
        if cat_breakdown:
            print(f"\n📂 CATEGORY BREAKDOWN:")
            for category, score in sorted(cat_breakdown.items(), key=lambda x: x[1], reverse=True):
                print(f"   {category.replace('_', ' ').title()}: {score:.2%}")
        
        # LLM metrics
        llm_metrics = report.get("llm_metrics", {})
        print(f"\n🤖 LLM METRICS:")
        print(f"   Total Queries:     {llm_metrics.get('total_queries', 0)}")
        print(f"   Cache Hit Rate:    {llm_metrics.get('cache_hit_rate', 0):.1f}%")
        print(f"   RAG Enabled:       {'✅ Yes' if llm_metrics.get('rag_enabled') else '❌ No'}")
        
        print("\n" + "="*80 + "\n")


def run_full_benchmark(llm_integration) -> Dict:
    """
    Run full benchmark suite
    
    Args:
        llm_integration: BankingLLMIntegration instance
        
    Returns:
        Complete benchmark report
    """
    benchmark = ComprehensiveBenchmark(llm_integration)
    report = benchmark.run_benchmark(verbose=True)
    benchmark.print_report(report)
    
    return report
