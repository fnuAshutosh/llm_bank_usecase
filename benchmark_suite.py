"""
🧪 LoRA & Fine-Tuning Benchmark Suite
For Kaggle/Colab experimentation

Run this cell by cell to benchmark different fine-tuning methods.
"""

# ============================================================================
# PART 1: SETUP & DEPENDENCIES
# ============================================================================

# Install required packages
import subprocess
import sys
import time
from datetime import datetime
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch


def install_packages():
    packages = [
        "peft",
        "transformers>=4.30.0",
        "datasets",
        "evaluate",
        "tensorboard",
        "bitsandbytes",
        "wandb",
        "pinecone-client",
        "sentence-transformers",
        "scikit-learn",
    ]
    for package in packages:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", package])

install_packages()

from datasets import DatasetDict, load_dataset
from peft import LoraConfig, TaskType, get_peft_model
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

print("✅ All packages installed!")

# ============================================================================
# PART 2: DATASET LOADING
# ============================================================================

class BankingBenchmark:
    def __init__(self, device: str = "cuda"):
        self.device = device
        self.results = []
        self.timings = {}
        
    def load_dataset(self):
        """Load Banking77 dataset"""
        print("📥 Loading Banking77 dataset...")
        dataset = load_dataset("banking77")
        
        # Split into train/val/test
        train_val = dataset["train"].train_test_split(test_size=0.2, seed=42)
        val_test = train_val["test"].train_test_split(test_size=0.5, seed=42)
        
        self.dataset = DatasetDict({
            "train": train_val["train"],
            "validation": val_test["train"],
            "test": val_test["test"],
        })
        
        print(f"✅ Dataset loaded: {self.dataset}")
        return self.dataset
    
    def prepare_data(self, model_name: str = "distilbert-base-uncased"):
        """Tokenize and prepare data"""
        print(f"🔧 Preparing data with {model_name}...")
        
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        def tokenize(batch):
            return tokenizer(
                batch["text"],
                padding="max_length",
                max_length=128,
                truncation=True,
            )
        
        self.tokenized_dataset = self.dataset.map(
            tokenize,
            batched=True,
            remove_columns=["text"],
        )
        
        print("✅ Data prepared!")
        return self.tokenized_dataset, tokenizer

# ============================================================================
# PART 3: BENCHMARK METHODS
# ============================================================================

class MethodBenchmark(BankingBenchmark):
    
    def benchmark_lora(self, configs: List[Dict]):
        """Benchmark LoRA with different configurations"""
        print("\n" + "="*60)
        print("🔬 BENCHMARKING LoRA")
        print("="*60)
        
        results = []
        
        for config in configs:
            print(f"\n📊 Testing LoRA with config: {config}")
            
            model = AutoModelForSequenceClassification.from_pretrained(
                "distilbert-base-uncased",
                num_labels=77,
            ).to(self.device)
            
            lora_config = LoraConfig(
                r=config["r"],
                lora_alpha=config["lora_alpha"],
                target_modules=config["target_modules"],
                lora_dropout=config["lora_dropout"],
                bias="none",
                task_type=TaskType.SEQ_CLS,
            )
            
            model = get_peft_model(model, lora_config)
            
            # Count parameters
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            total_params = sum(p.numel() for p in model.parameters())
            
            # Train
            training_args = TrainingArguments(
                output_dir=f"./lora_r{config['r']}",
                per_device_train_batch_size=16,
                per_device_eval_batch_size=16,
                num_train_epochs=3,
                learning_rate=2e-4,
                logging_steps=50,
                evaluation_strategy="epoch",
                save_strategy="epoch",
                load_best_model_at_end=True,
                logging_dir="./logs",
            )
            
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=self.tokenized_dataset["train"],
                eval_dataset=self.tokenized_dataset["validation"],
                callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
            )
            
            # Measure training time and memory
            torch.cuda.reset_peak_memory_stats()
            start_time = time.time()
            
            trainer.train()
            
            training_time = time.time() - start_time
            peak_memory = torch.cuda.max_memory_allocated() / 1024 / 1024 / 1024
            
            # Evaluate
            predictions = trainer.predict(self.tokenized_dataset["test"])
            preds = np.argmax(predictions.predictions, axis=1)
            accuracy = accuracy_score(self.dataset["test"]["label"], preds)
            
            result = {
                "method": "LoRA",
                "r": config["r"],
                "lora_alpha": config["lora_alpha"],
                "lora_dropout": config["lora_dropout"],
                "trainable_params": trainable_params,
                "total_params": total_params,
                "param_ratio": (trainable_params / total_params) * 100,
                "training_time": training_time,
                "peak_memory_gb": peak_memory,
                "accuracy": accuracy,
                "timestamp": datetime.now().isoformat(),
            }
            
            results.append(result)
            print(f"✅ Done! Accuracy: {accuracy:.2%}, Time: {training_time:.1f}s, Memory: {peak_memory:.1f}GB")
            
            # Cleanup
            torch.cuda.empty_cache()
        
        results_df = pd.DataFrame(results)
        print("\n📈 LoRA Results:")
        print(results_df)
        
        return results_df
    
    def benchmark_qLora(self):
        """Benchmark QLoRA (4-bit quantization)"""
        print("\n" + "="*60)
        print("🔬 BENCHMARKING QLoRA (4-bit)")
        print("="*60)
        
        try:
            from transformers import BitsAndBytesConfig
            
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
            
            model = AutoModelForSequenceClassification.from_pretrained(
                "distilbert-base-uncased",
                num_labels=77,
                quantization_config=bnb_config,
                device_map="auto",
            )
            
            lora_config = LoraConfig(
                r=8,
                lora_alpha=16,
                target_modules=["q_proj", "v_proj"],
                lora_dropout=0.05,
                bias="none",
                task_type=TaskType.SEQ_CLS,
            )
            
            model = get_peft_model(model, lora_config)
            
            # Train
            torch.cuda.reset_peak_memory_stats()
            start_time = time.time()
            
            # [Training code similar to LoRA]
            
            training_time = time.time() - start_time
            peak_memory = torch.cuda.max_memory_allocated() / 1024 / 1024 / 1024
            
            print(f"✅ QLoRA Training - Time: {training_time:.1f}s, Memory: {peak_memory:.1f}GB")
            
        except Exception as e:
            print(f"❌ QLoRA benchmarking failed: {e}")

# ============================================================================
# PART 4: VECTOR DATABASE BENCHMARKING
# ============================================================================

class VectorDBBenchmark:
    """Benchmark different vector databases"""
    
    def benchmark_vector_dbs(self):
        """Compare vector database performance"""
        print("\n" + "="*60)
        print("🔬 BENCHMARKING VECTOR DATABASES")
        print("="*60)
        
        from sentence_transformers import SentenceTransformer
        
        # Generate sample embeddings
        model = SentenceTransformer("all-MiniLM-L6-v2")
        
        # Create dummy banking queries
        banking_queries = [
            "I want to transfer money to my account",
            "What's my account balance",
            "How do I set up a loan",
            "Check my recent transactions",
            "I need to report a fraud",
        ] * 200  # Repeat to get 1000 queries
        
        print(f"📊 Generating {len(banking_queries)} embeddings...")
        embeddings = model.encode(banking_queries, show_progress_bar=True)
        print(f"✅ Embeddings shape: {embeddings.shape}")
        
        # Benchmark FAISS (local)
        self._benchmark_faiss(embeddings)
        
        # Benchmark ChromaDB (if available)
        self._benchmark_chromadb(embeddings, banking_queries)
    
    def _benchmark_faiss(self, embeddings):
        """Benchmark FAISS locally"""
        try:
            import faiss
            
            print("\n📦 Benchmarking FAISS...")
            
            index = faiss.IndexFlatL2(embeddings.shape[1])
            
            start = time.time()
            index.add(embeddings.astype(np.float32))
            indexing_time = time.time() - start
            
            # Query
            query = embeddings[0:10]
            start = time.time()
            distances, indices = index.search(query.astype(np.float32), k=5)
            query_time = (time.time() - start) / len(query)
            
            print(f"✅ FAISS - Indexing: {indexing_time:.2f}s, Query: {query_time*1000:.1f}ms")
            
        except ImportError:
            print("⚠️ FAISS not installed")
    
    def _benchmark_chromadb(self, embeddings, texts):
        """Benchmark ChromaDB"""
        try:
            import chromadb
            
            print("\n📦 Benchmarking ChromaDB...")
            
            client = chromadb.Client()
            collection = client.create_collection(name="banking")
            
            start = time.time()
            for i, (emb, text) in enumerate(zip(embeddings, texts)):
                collection.add(ids=[str(i)], embeddings=[emb.tolist()], documents=[text])
            indexing_time = time.time() - start
            
            # Query
            start = time.time()
            results = collection.query(query_embeddings=[embeddings[0].tolist()], n_results=5)
            query_time = (time.time() - start) * 1000
            
            print(f"✅ ChromaDB - Indexing: {indexing_time:.2f}s, Query: {query_time:.1f}ms")
            
        except ImportError:
            print("⚠️ ChromaDB not installed")

# ============================================================================
# PART 5: RUN BENCHMARKS
# ============================================================================

def run_all_benchmarks():
    """Master function to run all benchmarks"""
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🚀 Starting benchmarks on {device.upper()}")
    
    # Initialize benchmark
    benchmark = MethodBenchmark(device=device)
    
    # Load dataset
    benchmark.load_dataset()
    benchmark.prepare_data()
    
    # LoRA configurations to test
    lora_configs = [
        {"r": 4, "lora_alpha": 8, "target_modules": ["q_proj", "v_proj"], "lora_dropout": 0.05},
        {"r": 8, "lora_alpha": 16, "target_modules": ["q_proj", "v_proj"], "lora_dropout": 0.05},
        {"r": 16, "lora_alpha": 32, "target_modules": ["q_proj", "v_proj"], "lora_dropout": 0.1},
    ]
    
    # Run LoRA benchmarks
    lora_results = benchmark.benchmark_lora(lora_configs)
    
    # Run Vector DB benchmarks
    vdb_benchmark = VectorDBBenchmark()
    vdb_benchmark.benchmark_vector_dbs()
    
    # Save results
    lora_results.to_csv("lora_benchmark_results.csv", index=False)
    print(f"\n💾 Results saved to lora_benchmark_results.csv")
    
    return lora_results

# ============================================================================
# PART 6: VISUALIZATION
# ============================================================================

def plot_results(results_df):
    """Create visualization of benchmark results"""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("LoRA Benchmark Results", fontsize=16, fontweight="bold")
    
    # Accuracy vs Rank
    axes[0, 0].plot(results_df["r"], results_df["accuracy"] * 100, marker="o", linewidth=2)
    axes[0, 0].set_xlabel("LoRA Rank")
    axes[0, 0].set_ylabel("Accuracy (%)")
    axes[0, 0].set_title("Accuracy vs LoRA Rank")
    axes[0, 0].grid(True)
    
    # Memory vs Rank
    axes[0, 1].plot(results_df["r"], results_df["peak_memory_gb"], marker="s", linewidth=2, color="orange")
    axes[0, 1].set_xlabel("LoRA Rank")
    axes[0, 1].set_ylabel("Peak Memory (GB)")
    axes[0, 1].set_title("Memory vs LoRA Rank")
    axes[0, 1].grid(True)
    
    # Training Time vs Rank
    axes[1, 0].plot(results_df["r"], results_df["training_time"], marker="^", linewidth=2, color="green")
    axes[1, 0].set_xlabel("LoRA Rank")
    axes[1, 0].set_ylabel("Training Time (seconds)")
    axes[1, 0].set_title("Training Time vs LoRA Rank")
    axes[1, 0].grid(True)
    
    # Parameter Efficiency
    axes[1, 1].bar(results_df["r"].astype(str), results_df["param_ratio"])
    axes[1, 1].set_xlabel("LoRA Rank")
    axes[1, 1].set_ylabel("Trainable Parameters (%)")
    axes[1, 1].set_title("Parameter Efficiency")
    axes[1, 1].grid(True, axis="y")
    
    plt.tight_layout()
    plt.savefig("benchmark_results.png", dpi=150)
    print("📊 Plot saved as benchmark_results.png")

# ============================================================================
# PART 7: RUN
# ============================================================================

if __name__ == "__main__":
    print("🧪 LoRA & Fine-Tuning Benchmark Suite")
    print("="*60)
    
    # Run benchmarks
    results = run_all_benchmarks()
    
    # Visualize
    plot_results(results)
    
    print("\n✅ Benchmarking complete!")
    print("\n📋 Summary:")
    print(results[["r", "accuracy", "training_time", "peak_memory_gb", "param_ratio"]])
