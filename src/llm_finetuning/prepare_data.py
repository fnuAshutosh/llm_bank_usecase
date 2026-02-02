"""
Prepare banking conversation data for Llama 2 fine-tuning
"""
import json
from pathlib import Path
from typing import Dict, List

from datasets import Dataset


def load_banking_conversations() -> List[str]:
    """Load our sample banking conversations"""
    sample_data = [
        "Customer: What is my current account balance? Agent: Your checking account balance is $2,450.32.",
        "Customer: I need to transfer $500 to savings. Agent: Transfer initiated. Your new checking balance is $1,950.32.",
        "Customer: How much interest do I earn? Agent: Your savings account earns 2.5% APY.",
        "Customer: Can you help with a declined transaction? Agent: Let me check. The transaction was declined due to insufficient funds.",
        "Customer: What are your ATM fees? Agent: ATM withdrawals at our network are free. Out-of-network fees are $3.",
        "Customer: I want to open a new account. Agent: Great! We offer checking, savings, and money market accounts.",
        "Customer: How do I dispute a charge? Agent: You can file a dispute through our mobile app or by calling customer service.",
        "Customer: What's my credit card limit? Agent: Your current credit limit is $10,000 with $7,200 available.",
        "Customer: Can I get a loan? Agent: Yes, we offer personal loans, auto loans, and mortgages. What are you looking for?",
        "Customer: I lost my debit card. Agent: I'll deactivate it immediately and order a replacement. It will arrive in 5-7 days.",
        "Customer: What's the overdraft fee? Agent: Our overdraft protection fee is $35 per transaction.",
        "Customer: Can I increase my credit limit? Agent: Yes, I can submit a request. It typically takes 3-5 business days.",
        "Customer: How do I set up direct deposit? Agent: You'll need our routing number (123456789) and your account number.",
        "Customer: What are your business hours? Agent: We're open Monday-Friday 9am-5pm, and Saturday 9am-1pm.",
        "Customer: I need a cashier's check. Agent: Cashier's checks cost $10. How much do you need it for?",
        "Customer: Can I pay bills online? Agent: Yes, our bill pay service is free and available through online banking.",
        "Customer: What's the minimum balance? Agent: Checking accounts require a $500 minimum to avoid monthly fees.",
        "Customer: How do I close my account? Agent: You can close it in person, by phone, or through secure message.",
        "Customer: Are there foreign transaction fees? Agent: Yes, there's a 3% fee on international purchases.",
        "Customer: Can I get a credit score report? Agent: Yes, we provide free monthly credit score updates through your account.",
    ] * 50  # 1000 examples
    
    return sample_data


def format_for_llama(conversations: List[str]) -> List[Dict[str, str]]:
    """
    Format conversations for Llama 2 instruction fine-tuning
    
    Format:
    <s>[INST] <<SYS>>
    You are a helpful banking assistant.
    <</SYS>>
    
    {user_message} [/INST] {assistant_response} </s>
    """
    formatted_data = []
    
    for conv in conversations:
        # Split into customer and agent parts
        parts = conv.split(" Agent: ")
        if len(parts) != 2:
            continue
            
        customer_msg = parts[0].replace("Customer: ", "").strip()
        agent_msg = parts[1].strip()
        
        # Llama 2 chat format
        instruction = f"""<s>[INST] <<SYS>>
You are a helpful banking assistant. Provide accurate and professional responses to customer inquiries.
<</SYS>>

{customer_msg} [/INST] {agent_msg} </s>"""
        
        formatted_data.append({
            "text": instruction,
            "input": customer_msg,
            "output": agent_msg
        })
    
    return formatted_data


def create_dataset(output_dir: str = "data/finetuning"):
    """Create and save HuggingFace dataset"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load and format data
    conversations = load_banking_conversations()
    formatted = format_for_llama(conversations)
    
    # Split train/val
    split_idx = int(len(formatted) * 0.9)
    train_data = formatted[:split_idx]
    val_data = formatted[split_idx:]
    
    # Create datasets
    train_dataset = Dataset.from_list(train_data)
    val_dataset = Dataset.from_list(val_data)
    
    # Save
    train_dataset.save_to_disk(str(output_path / "train"))
    val_dataset.save_to_disk(str(output_path / "val"))
    
    # Also save as JSON for inspection
    with open(output_path / "train.json", "w") as f:
        json.dump(train_data, f, indent=2)
    
    with open(output_path / "val.json", "w") as f:
        json.dump(val_data, f, indent=2)
    
    print(f"✓ Created dataset:")
    print(f"  Train: {len(train_data)} examples")
    print(f"  Val: {len(val_data)} examples")
    print(f"  Saved to: {output_path}")
    
    return train_dataset, val_dataset


if __name__ == "__main__":
    create_dataset()
