"""
Load and prepare Banking77 dataset for fine-tuning
Banking77: 13,083 customer queries across 77 banking intents
Source: https://huggingface.co/datasets/banking77
"""
import json
import random
from pathlib import Path
from typing import Dict, List

from datasets import Dataset, load_dataset

# Intent-to-response templates for Banking77
# We'll use these to generate appropriate responses for each intent category
INTENT_RESPONSE_TEMPLATES = {
    "activate_my_card": "I've activated your card. It should be ready to use within 24 hours.",
    "age_limit": "You must be 18 years or older to open an account with us.",
    "apple_pay_or_google_pay": "Yes, we support both Apple Pay and Google Pay. You can add your card through the respective app.",
    "atm_support": "We have over 40,000 ATMs nationwide. Use our app to find the nearest one.",
    "automatic_top_up": "I can set up automatic top-up for you. What amount and frequency would you prefer?",
    "balance_not_updated_after_bank_transfer": "Bank transfers can take 1-3 business days. If it's been longer, I'll investigate.",
    "balance_not_updated_after_cheque_or_cash_deposit": "Deposits typically process within 24 hours. Let me check the status for you.",
    "beneficiary_not_allowed": "This beneficiary may be restricted. Can you provide more details about the transfer?",
    "cancel_transfer": "I can cancel the transfer if it hasn't been processed yet. Let me check the status.",
    "card_about_to_expire": "Your new card should arrive 2-3 weeks before expiration. Contact us if you haven't received it.",
    "card_acceptance": "Our cards are accepted at millions of locations worldwide wherever Visa/Mastercard is accepted.",
    "card_arrival": "New cards typically arrive within 7-10 business days. I can check the delivery status for you.",
    "card_delivery_estimate": "Your card should arrive in 7-10 business days. You'll receive tracking information via email.",
    "card_linking": "You can link your card through the app under Settings > Linked Accounts.",
    "card_not_working": "Let me troubleshoot this. Is the card being declined or not reading properly?",
    "card_payment_fee_charged": "We don't charge fees for standard card payments. If you see a charge, it may be from the merchant.",
    "card_payment_not_recognised": "I don't see that transaction. It may take 24-48 hours to appear. Can you provide more details?",
    "card_payment_wrong_exchange_rate": "Exchange rates are set at the time of transaction. I can review this charge for you.",
    "card_swallowed": "If an ATM retained your card, contact us immediately to block it and order a replacement.",
    "cash_withdrawal_charge": "ATM withdrawals at our network are free. Out-of-network ATMs may charge $3.",
    "cash_withdrawal_not_recognised": "I don't see that withdrawal. Can you provide the date, amount, and ATM location?",
    "change_pin": "You can change your PIN at any ATM or through our mobile app under Security settings.",
    "compromised_card": "I'll block this card immediately and order a replacement. Have you noticed any unauthorized charges?",
    "contactless_not_working": "Try inserting the card instead. If contactless still doesn't work, we'll send a replacement.",
    "country_support": "We operate in 45 countries. Which country are you asking about?",
    "declined_card_payment": "Payments can be declined for several reasons: insufficient funds, security hold, or expired card. Let me check.",
    "declined_cash_withdrawal": "This could be due to daily limits or insufficient funds. Let me review your account.",
    "declined_transfer": "Transfers may be declined if the recipient's information is incorrect or limits are exceeded.",
    "direct_debit_payment_not_recognised": "I'll investigate this direct debit. Can you provide the merchant name and amount?",
    "disposable_card_limits": "Virtual disposable cards have a limit of $5,000 per transaction and $10,000 per day.",
    "edit_personal_details": "You can update your details in the app under Profile, or I can assist you here.",
    "exchange_charge": "Foreign exchange fees are 2.5% for currency conversion. Premium accounts have reduced rates.",
    "exchange_rate": "Current exchange rates are available in our app. They update every hour based on market rates.",
    "exchange_via_app": "Yes, you can exchange currency in the app under Accounts > Currency Exchange.",
    "extra_charge_on_statement": "Let me review that charge. Can you provide the date and merchant name?",
    "failed_transfer": "I see the transfer failed. This could be due to incorrect details or insufficient funds. Let me investigate.",
    "fiat_currency_support": "We support USD, EUR, GBP, CAD, AUD, and 15 other major currencies.",
    "get_disposable_virtual_card": "You can create a virtual card instantly in the app under Cards > Add Virtual Card.",
    "get_physical_card": "Physical cards are free with our accounts. Would you like to order one?",
    "getting_spare_card": "You can request an additional card for $5. It will have the same account access.",
    "getting_virtual_card": "Virtual cards are free and instant. Create one in the app under Cards section.",
    "lost_or_stolen_card": "I'll block your card immediately to prevent unauthorized use and order a replacement.",
    "lost_or_stolen_phone": "Block your card via web banking or call us at 1-800-XXX-XXXX to secure your account.",
    "order_physical_card": "I can order a physical card for you. It will arrive in 7-10 business days at your registered address.",
    "passcode_forgotten": "You can reset your passcode through the app using SMS verification or security questions.",
    "pending_card_payment": "Pending payments typically clear within 2-3 business days. Some merchants may take longer.",
    "pending_cash_withdrawal": "ATM withdrawals usually process immediately. If it's pending, it should clear within 24 hours.",
    "pending_top_up": "Top-ups can take 1-3 business days depending on your funding source.",
    "pending_transfer": "Transfers are usually instant within our network. External transfers take 1-3 business days.",
    "pin_blocked": "Your PIN is blocked after 3 incorrect attempts. Visit any branch with ID to reset it.",
    "receiving_money": "To receive money, share your account number and routing number with the sender.",
    "Refund_not_showing_up": "Refunds can take 5-10 business days to appear. Let me check if it's been processed.",
    "request_refund": "I can help initiate a refund request. Can you provide the transaction details?",
    "reverted_card_payment?": "Some merchants place temporary holds that get reversed. This is normal and should clear soon.",
    "supported_cards_and_currencies": "We support Visa and Mastercard in USD, EUR, GBP, and 15 other currencies.",
    "terminate_account": "I can help close your account. Please note this is permanent. Any remaining balance will be transferred.",
    "top_up_by_bank_transfer_charge": "Bank transfers are free. The transfer should complete in 1-3 business days.",
    "top_up_by_card_charge": "Card top-ups are free for debit cards. Credit cards incur a 2.5% fee.",
    "top_up_by_cash_or_cheque": "You can deposit cash or checks at any branch. Deposits typically process within 24 hours.",
    "top_up_failed": "Top-ups can fail if payment details are incorrect or insufficient funds. Let me check the error.",
    "top_up_limits": "Daily top-up limit is $10,000. Monthly limit is $50,000 for standard accounts.",
    "top_up_reverted": "If a top-up was reverted, it's usually due to insufficient funds from the source. Funds will be returned.",
    "topping_up_by_card": "You can top up instantly using any debit or credit card through the app.",
    "transaction_charged_twice": "Duplicate charges are usually temporary holds. If both clear, we'll investigate and refund one.",
    "transfer_fee_charged": "Internal transfers are free. External transfers may incur a $5 fee depending on the recipient bank.",
    "transfer_into_account": "You can receive transfers using your account number and routing number.",
    "transfer_not_received_by_recipient": "Let me track this transfer. Can you provide the recipient's details and transfer date?",
    "transfer_timing": "Internal transfers are instant. External transfers take 1-3 business days.",
    "unable_to_verify_identity": "Identity verification requires a government ID. You can upload it in the app or visit a branch.",
    "verify_my_identity": "To verify your identity, upload a photo of your driver's license or passport through the app.",
    "verify_source_of_funds": "For compliance, we may need documentation for large deposits. This could include pay stubs or invoices.",
    "verify_top_up": "Please confirm the top-up amount and source. I'll verify it's been processed correctly.",
    "virtual_card_not_working": "Virtual cards should work for online purchases. Let me check if there's a technical issue.",
    "visa_or_mastercard": "We offer both Visa and Mastercard. You can choose your preference when ordering.",
    "why_verify_identity": "Identity verification is required by law to prevent fraud and comply with banking regulations.",
    "wrong_amount_of_cash_received": "If the ATM dispensed incorrect cash, report it immediately. We'll investigate with the ATM operator.",
    "wrong_exchange_rate_for_cash_withdrawal": "Exchange rates for ATM withdrawals are set at transaction time. Premium accounts get better rates.",
}


def load_banking77_dataset() -> tuple[Dataset, Dataset]:
    """Load Banking77 dataset from HuggingFace"""
    print("Loading Banking77 dataset from HuggingFace...")
    
    # Load dataset
    dataset = load_dataset("banking77")
    
    # Get train and test splits
    train_data = dataset['train']
    test_data = dataset['test']
    
    print(f"✓ Loaded {len(train_data)} training examples")
    print(f"✓ Loaded {len(test_data)} test examples")
    
    return train_data, test_data


def get_intent_label(label_id: int, dataset) -> str:
    """Convert numeric label to intent string"""
    # Banking77 has 77 intent classes
    intent_names = dataset.features['label'].names
    return intent_names[label_id]


def generate_response_for_intent(query: str, intent: str) -> str:
    """
    Generate appropriate response based on intent
    Uses template-based generation for consistency
    """
    # Get template for this intent
    response = INTENT_RESPONSE_TEMPLATES.get(
        intent,
        "I'd be happy to help you with that. Let me look into your account."
    )
    
    # Could enhance with GPT-4 API here for more varied responses
    return response


def convert_to_llama_format(
    examples: Dataset,
    max_examples: int = None
) -> List[Dict[str, str]]:
    """
    Convert Banking77 examples to Llama 2 instruction format
    """
    formatted_data = []
    
    total = len(examples) if max_examples is None else min(max_examples, len(examples))
    
    for i in range(total):
        query = examples[i]['text']
        label_id = examples[i]['label']
        
        # Get intent name
        intent = get_intent_label(label_id, examples)
        
        # Generate response
        response = generate_response_for_intent(query, intent)
        
        # Format for Llama 2
        instruction = f"""<s>[INST] <<SYS>>
You are a professional banking assistant. Provide accurate, helpful, and compliant responses to customer inquiries.
<</SYS>>

{query} [/INST] {response} </s>"""
        
        formatted_data.append({
            "text": instruction,
            "input": query,
            "output": response,
            "intent": intent,
            "label": label_id
        })
        
        if (i + 1) % 1000 == 0:
            print(f"  Processed {i + 1}/{total} examples...")
    
    return formatted_data


def create_banking77_dataset(
    output_dir: str = "data/banking77_finetuning",
    use_full_dataset: bool = True
):
    """
    Download Banking77 and prepare for fine-tuning
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load Banking77
    train_data, test_data = load_banking77_dataset()
    
    # Show sample
    print("\nSample from Banking77:")
    print(f"  Query: {train_data[0]['text']}")
    print(f"  Intent: {get_intent_label(train_data[0]['label'], train_data)}")
    print()
    
    # Convert to Llama format
    print("Converting to Llama 2 instruction format...")
    train_formatted = convert_to_llama_format(train_data)
    test_formatted = convert_to_llama_format(test_data)
    
    # Split test into val and test
    val_size = len(test_formatted) // 2
    val_formatted = test_formatted[:val_size]
    final_test_formatted = test_formatted[val_size:]
    
    # Create HF datasets
    train_dataset = Dataset.from_list(train_formatted)
    val_dataset = Dataset.from_list(val_formatted)
    test_dataset = Dataset.from_list(final_test_formatted)
    
    # Save
    print("\nSaving datasets...")
    train_dataset.save_to_disk(str(output_path / "train"))
    val_dataset.save_to_disk(str(output_path / "val"))
    test_dataset.save_to_disk(str(output_path / "test"))
    
    # Save JSON for inspection
    with open(output_path / "train_sample.json", "w") as f:
        json.dump(train_formatted[:10], f, indent=2)
    
    # Save statistics
    stats = {
        "dataset": "Banking77",
        "source": "https://huggingface.co/datasets/banking77",
        "train_size": len(train_formatted),
        "val_size": len(val_formatted),
        "test_size": len(final_test_formatted),
        "total_intents": 77,
        "format": "Llama 2 instruction format"
    }
    
    with open(output_path / "dataset_info.json", "w") as f:
        json.dump(stats, f, indent=2)
    
    print(f"\n{'='*60}")
    print("✓ Banking77 Dataset Prepared Successfully!")
    print(f"{'='*60}")
    print(f"  Train: {len(train_formatted)} examples")
    print(f"  Val: {len(val_formatted)} examples")
    print(f"  Test: {len(final_test_formatted)} examples")
    print(f"  Total intents: 77")
    print(f"  Output: {output_path}")
    print(f"{'='*60}\n")
    
    return train_dataset, val_dataset, test_dataset


if __name__ == "__main__":
    create_banking77_dataset()
