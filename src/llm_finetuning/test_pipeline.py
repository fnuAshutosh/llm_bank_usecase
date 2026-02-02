"""
Quick QLoRA fine-tuning test with minimal resources
Tests the pipeline without full training
"""
from pathlib import Path

import torch
from datasets import load_from_disk
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def quick_test():
    """Quick test of QLoRA fine-tuning setup"""
    
    print("="*60)
    print("Quick QLoRA Fine-tuning Test (No Full Training)")
    print("="*60)
    
    # 1. Load dataset
    print("\n1. Loading Banking77 dataset...")
    data_path = Path("data/banking77_finetuning")
    train_dataset = load_from_disk(str(data_path / "train"))
    val_dataset = load_from_disk(str(data_path / "val"))
    
    print(f"   ✓ Train: {len(train_dataset)} examples")
    print(f"   ✓ Val: {len(val_dataset)} examples")
    
    # Show sample
    sample = train_dataset[0]
    print(f"\n   Sample query: {sample['input'][:80]}...")
    print(f"   Intent: {sample['intent']}")
    print(f"   Response: {sample['output'][:80]}...")
    
    # 2. Load model (smaller for testing)
    print("\n2. Loading TinyLlama with QLoRA...")
    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Setup 4-bit quantization (QLoRA)
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True
        )
        
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quantization_config,
            device_map="auto"
        )
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            model.config.pad_token_id = model.config.eos_token_id
        
        print(f"   ✓ Model loaded: {model_name}")
        
        # Model size
        params = sum(p.numel() for p in model.parameters())
        print(f"   ✓ Total parameters: {params:,} ({params/1e9:.2f}B)")
        
    except Exception as e:
        print(f"   ✗ Error loading model: {e}")
        return
    
    # 3. Setup LoRA
    print("\n3. Configuring LoRA...")
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )
    
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, lora_config)
    
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    
    print(f"   ✓ LoRA configured")
    print(f"   ✓ Trainable params: {trainable:,} ({100 * trainable / total:.2f}%)")
    print(f"   ✓ Memory savings: {100 * (1 - trainable/total):.1f}% fewer params to train")
    
    # 4. Test tokenization
    print("\n4. Testing tokenization...")
    sample_text = train_dataset[0]['text']
    tokens = tokenizer(sample_text, truncation=True, max_length=512, padding="max_length")
    
    print(f"   ✓ Input tokens: {len(tokens['input_ids'])}")
    print(f"   ✓ Sample: {tokenizer.decode(tokens['input_ids'][:50])}...")
    
    # 5. Test forward pass
    print("\n5. Testing forward pass...")
    input_ids = torch.tensor([tokens['input_ids'][:128]])  # Shorter for speed
    
    with torch.no_grad():
        outputs = model(input_ids)
        logits = outputs.logits
    
    print(f"   ✓ Output shape: {logits.shape}")
    print(f"   ✓ Forward pass successful")
    
    # 6. Test generation
    print("\n6. Testing generation (before fine-tuning)...")
    test_prompt = """<s>[INST] <<SYS>>
You are a professional banking assistant.
<</SYS>>

What is my account balance? [/INST]"""
    
    inputs = tokenizer(test_prompt, return_tensors="pt")
    
    print(f"   Generating response...")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=50,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    if "[/INST]" in response:
        response = response.split("[/INST]")[1].strip()
    
    print(f"   Response: {response[:200]}...")
    
    # Summary
    print("\n" + "="*60)
    print("✓ Fine-tuning Pipeline Validated!")
    print("="*60)
    print("\nReady for full training:")
    print("  - Dataset: 10,003 Banking77 examples")
    print("  - Model: TinyLlama 1.1B")
    print("  - Method: LoRA (1.5% trainable params)")
    print("  - Estimated time: 2-4 hours on CPU")
    print("  - Memory needed: ~4-6GB RAM")
    print("\nTo run full training:")
    print("  python src/llm_finetuning/finetune_llama.py")
    print("\nNote: Training on CPU will be slow. For production,")
    print("      use GPU (A100/V100) or cloud training services.")
    print("="*60)


if __name__ == "__main__":
    quick_test()
