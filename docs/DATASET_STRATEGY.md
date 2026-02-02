# Banking Domain Datasets - Quality Assessment

## What Data Do We Need?

### For Fine-tuning a Banking LLM:

**1. Instruction-Response Pairs**
- Customer query → Agent response
- Question → Answer format
- Task → Completion pairs

**2. Domain Coverage**
- Account management (balance, statements, history)
- Transactions (transfers, payments, deposits)
- Cards (credit/debit, limits, disputes)
- Loans (applications, rates, eligibility)
- Customer service (complaints, FAQs, support)

**3. Quality Requirements**
- ✅ Accurate banking information
- ✅ Professional language
- ✅ Privacy-compliant (no real PII)
- ✅ Diverse query types
- ✅ Multiple conversation styles

---

## Peer-Reviewed Banking Datasets

### 1. **Banking77** (Recommended ⭐⭐⭐)
- **Source**: HuggingFace, University of Edinburgh
- **Size**: 13,083 queries across 77 intents
- **Quality**: Peer-reviewed, published paper
- **License**: CC BY 4.0 (Commercial use OK)
- **Link**: https://huggingface.co/datasets/banking77
- **Citation**: Casanueva et al., 2020

**Sample Data**:
```json
{
  "text": "What is the fee for international transfer?",
  "label": 26  // (transfer_fee_charged intent)
}
```

**Pros**:
- ✅ High quality, human-labeled
- ✅ Real customer queries
- ✅ Published research (NeurIPS 2020)
- ✅ Wide coverage of banking intents

**Cons**:
- ⚠️ No responses (only queries + labels)
- ⚠️ Need to generate responses ourselves

---

### 2. **ConvFinQA** (Finance Q&A)
- **Source**: HuggingFace, University of Washington
- **Size**: 3,892 conversations
- **Quality**: Academic research dataset
- **License**: MIT
- **Link**: https://huggingface.co/datasets/FinQA/convfinqa
- **Citation**: Chen et al., 2022

**Sample Data**:
```json
{
  "question": "What is the company's revenue in 2020?",
  "answer": "The revenue was $1.2 billion",
  "context": "Financial table/document"
}
```

**Pros**:
- ✅ Question-answer pairs
- ✅ Multi-turn conversations
- ✅ Financial domain expertise

**Cons**:
- ⚠️ Corporate finance focused (not retail banking)
- ⚠️ Complex numerical reasoning

---

### 3. **MultiWOZ 2.2** (Banking Subset)
- **Source**: Cambridge University
- **Size**: ~10,000 dialogues (subset with banking)
- **Quality**: Research-grade, multiple papers
- **License**: Apache 2.0
- **Link**: https://huggingface.co/datasets/multi_woz_v22

**Pros**:
- ✅ Multi-turn dialogues
- ✅ Task-oriented conversations
- ✅ Well-structured annotations

**Cons**:
- ⚠️ Small banking subset (mostly hotel/restaurant)
- ⚠️ UK-centric

---

### 4. **FiQA (Financial Q&A)**
- **Source**: WWW 2018 Challenge
- **Size**: 6,648 question-answer pairs
- **Quality**: Competition dataset, curated
- **License**: Research use
- **Link**: https://sites.google.com/view/fiqa/home

**Sample**:
```
Q: Should I pay off my credit card in full?
A: Yes, paying in full avoids interest charges...
```

**Pros**:
- ✅ Real financial advice
- ✅ Community-sourced (Reddit, Stack Exchange)
- ✅ Long-form answers

**Cons**:
- ⚠️ Personal finance (not commercial banking)
- ⚠️ Variable quality

---

### 5. **TaskMaster (Banking Dialogues)**
- **Source**: Google Research
- **Size**: 5,507 task-oriented dialogues
- **Quality**: Human-written, QA validated
- **License**: CC BY 4.0
- **Link**: https://github.com/google-research-datasets/Taskmaster

**Pros**:
- ✅ Natural conversations
- ✅ Task completion focused
- ✅ High annotation quality

**Cons**:
- ⚠️ Limited banking examples
- ⚠️ Requires parsing complex format

---

## Recommended Approach: Combine Multiple Sources

### **Best Strategy for Our Banking LLM**:

```
Banking77 (13K queries)
    ↓ (match to intents)
+ GPT-4/Claude to generate responses
    ↓ (quality check)
+ FiQA (7K finance Q&A)
    ↓ (filter banking-related)
+ Manual curation (500 examples)
    ↓ (domain expertise)
= 20K high-quality training examples
```

---

## Data Quality Checklist

### Essential Criteria:
- [ ] **Accuracy**: Banking info is correct
- [ ] **Diversity**: Covers all major use cases
- [ ] **Privacy**: No real customer data
- [ ] **Consistency**: Similar format across examples
- [ ] **Licensing**: Commercial use allowed

### Validation Process:
1. **Automated checks**:
   - Grammar/spelling correctness
   - Response relevance to query
   - No PII leakage
   
2. **Manual review** (sample 10%):
   - Banking accuracy
   - Professional tone
   - Completeness

3. **Domain expert review**:
   - Regulatory compliance
   - Product accuracy
   - Policy correctness

---

## Implementation Plan

### Phase 1: Acquire Banking77 (Primary Dataset)
- Download from HuggingFace
- 13K real customer queries
- Map to our banking intents

### Phase 2: Generate Responses
- Use GPT-4/Claude API to generate responses
- Follow banking compliance guidelines
- Human review sample set

### Phase 3: Augment with FiQA
- Filter for banking-relevant Q&A
- ~2K additional examples
- Diverse response styles

### Phase 4: Add Domain-Specific Examples
- Your 20 hand-crafted examples
- Edge cases not in public datasets
- Company-specific policies

### Final Dataset Target:
- **Size**: 15,000+ examples
- **Quality**: >95% accuracy
- **Coverage**: 77+ banking intents
- **Format**: Llama 2 instruction format

---

## Next Steps

1. **Download Banking77**: Integrate with our pipeline
2. **Generate responses**: Use LLM API for high-quality answers
3. **Validate**: Manual review of 500 examples
4. **Fine-tune**: Train on combined dataset
5. **Evaluate**: Test on held-out banking queries

Would you like me to implement the Banking77 integration now?
