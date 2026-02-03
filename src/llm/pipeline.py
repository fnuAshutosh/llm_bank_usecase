"""Production pipeline with automatic benchmarking"""

import asyncio
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from src.llm.model_factory import ModelFactory
from src.llm.models.base_adapter import ModelAdapter


class BankingLLMPipeline:
    """
    Unified inference pipeline with automatic benchmarking.
    
    Features:
    - Model abstraction (works with any adapter)
    - Automatic benchmarking at intervals
    - Comparative metrics across all models
    - JSONL logging of all benchmarks
    - Runtime model switching
    """
    
    def __init__(self, benchmark_interval: int = 100):
        """
        Initialize pipeline.
        
        Args:
            benchmark_interval: Run benchmark every N requests
        """
        
        self.factory = ModelFactory
        self.model: Optional[ModelAdapter] = None
        self.request_counter = 0
        self.benchmark_interval = benchmark_interval
        self.benchmark_results: List[Dict] = []
        self.benchmark_file = Path('benchmark_results.jsonl')
        
        self.load_model()
    
    def load_model(self):
        """Load active model from factory"""
        try:
            self.model = self.factory.get_active_model()
            print(f"✓ Pipeline loaded model: {self.model.version}")
            
        except Exception as e:
            print(f"✗ Failed to load model: {e}")
            raise
    
    async def generate(
        self,
        prompt: str,
        conversation_id: Optional[str] = None,
        **kwargs
    ) -> Dict:
        """
        Generate response with automatic benchmarking.
        
        Args:
            prompt: Input text
            conversation_id: For tracking conversations
            **kwargs: Passed to model.generate()
            
        Returns:
            Response with metadata
        """
        
        self.request_counter += 1
        
        # Generate response
        result = await self.model.generate(prompt, **kwargs)
        
        # Trigger benchmark if interval reached
        should_benchmark = (
            self.request_counter % self.benchmark_interval == 0
        )
        
        if should_benchmark:
            await self._run_benchmark()
        
        return {
            **result,
            'conversation_id': conversation_id,
            'request_number': self.request_counter,
            'pipeline_version': '1.0.0'
        }
    
    async def _run_benchmark(self):
        """Run comparative benchmark across all models"""
        
        print(f"\n📊 Running benchmark at request #{self.request_counter}...")
        
        test_prompts = [
            "What is my account balance?",
            "How do I transfer money?",
            "What are ATM fees?"
        ]
        
        benchmark_round = {
            'timestamp': datetime.now().isoformat(),
            'request_count': self.request_counter,
            'models': {}
        }
        
        # Test each model
        for model_name in self.factory.REGISTRY.keys():
            try:
                model = self.factory.create(model_name)
                
                # Generate responses for each test prompt
                latencies = []
                for prompt in test_prompts:
                    result = await model.generate(prompt, max_tokens=100)
                    latencies.append(result['latency_ms'])
                
                # Calculate metrics
                metrics = model.get_metrics()
                
                benchmark_round['models'][model_name] = {
                    'version': model.version,
                    'device': result.get('device', 'unknown'),
                    'avg_latency_ms': sum(latencies) / len(latencies),
                    'p99_latency_ms': sorted(latencies)[-1],
                    'p50_latency_ms': sorted(latencies)[len(latencies)//2],
                    'confidence': result.get('confidence', 0.0),
                    'status': 'healthy',
                    'full_metrics': metrics
                }
                
            except Exception as e:
                benchmark_round['models'][model_name] = {
                    'status': 'error',
                    'error': str(e)
                }
        
        self.benchmark_results.append(benchmark_round)
        
        # Persist to JSONL
        with open(self.benchmark_file, 'a') as f:
            f.write(json.dumps(benchmark_round) + '\n')
        
        # Display results
        self._print_benchmark_results(benchmark_round)
    
    def _print_benchmark_results(self, round_data: Dict):
        """Pretty print benchmark results"""
        
        print("\n" + "="*90)
        print("COMPARATIVE BENCHMARK RESULTS")
        print("="*90)
        
        print(f"Timestamp: {round_data['timestamp']}")
        print(f"Request Count: {round_data['request_count']}\n")
        
        print(f"{'Model':<25} {'Device':<8} {'Avg Latency':<15} "
              f"{'P50':<12} {'P99':<12} {'Confidence':<12}")
        print("-" * 90)
        
        for model_name, data in round_data['models'].items():
            if data['status'] == 'healthy':
                print(
                    f"{model_name:<25} {data['device']:<8} "
                    f"{data['avg_latency_ms']:>10.2f}ms    "
                    f"{data['p50_latency_ms']:>8.2f}ms   "
                    f"{data['p99_latency_ms']:>8.2f}ms   "
                    f"{data['confidence']:>8.1%}"
                )
            else:
                print(f"{model_name:<25} ERROR   {data.get('error', 'unknown')[:40]}")
        
        print("="*90 + "\n")
    
    def switch_model(self, model_version: str):
        """
        Switch to different model at runtime.
        
        Args:
            model_version: Model version string
                          (e.g., "qlora_vllm_finetuned_model_v1.0.0")
        """
        
        import os
        
        os.environ['LLM_MODEL_VERSION'] = model_version
        self.load_model()
        print(f"✓ Switched to model: {model_version}")
    
    def get_benchmark_summary(self) -> Dict:
        """
        Return summary of latest benchmark.
        
        Returns:
            Summary with comparison and recommendation
        """
        
        if not self.benchmark_results:
            return {'status': 'no_benchmarks_yet'}
        
        latest = self.benchmark_results[-1]
        
        summary = {
            'timestamp': latest['timestamp'],
            'request_count': latest['request_count'],
            'comparison': {}
        }
        
        # Extract key metrics for each model
        for model_name, data in latest['models'].items():
            if data['status'] == 'healthy':
                summary['comparison'][model_name] = {
                    'device': data.get('device'),
                    'avg_latency_ms': data.get('avg_latency_ms'),
                    'p99_latency_ms': data.get('p99_latency_ms'),
                    'confidence': data.get('confidence'),
                    'status': 'healthy'
                }
        
        # Find fastest and most accurate
        if summary['comparison']:
            fastest = min(
                summary['comparison'].items(),
                key=lambda x: x[1]['avg_latency_ms']
            )
            most_accurate = max(
                summary['comparison'].items(),
                key=lambda x: x[1]['confidence']
            )
            
            summary['fastest_model'] = fastest[0]
            summary['most_accurate_model'] = most_accurate[0]
            summary['recommendation'] = (
                f"Latency: {fastest[0]} ({fastest[1]['avg_latency_ms']:.1f}ms) | "
                f"Accuracy: {most_accurate[0]} ({most_accurate[1]['confidence']:.1%})"
            )
        
        return summary
    
    def get_all_benchmarks(self, limit: Optional[int] = None) -> List[Dict]:
        """
        Get benchmark history.
        
        Args:
            limit: Maximum number of recent benchmarks to return
            
        Returns:
            List of benchmark rounds
        """
        
        results = self.benchmark_results
        
        if limit:
            results = results[-limit:]
        
        return results
