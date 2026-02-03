"""
Enhanced RAG Service with Pinecone Integration

Implements Retrieval-Augmented Generation (RAG) for banking context:
1. Retrieves relevant banking policies/FAQs from Pinecone
2. Augments LLM prompts with real context
3. Reduces hallucinations and ensures accuracy
"""

import json
import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class BankingContextStore:
    """Stores banking policies and FAQs as embeddings in Pinecone"""
    
    BANKING_POLICIES = [
        {
            "id": "policy_001",
            "category": "account_management",
            "question": "How do I open a new account?",
            "answer": "You can open a new account through our mobile app, website, or by visiting a branch. We offer checking, savings, and money market accounts. You'll need a valid ID and initial deposit of $500.",
            "metadata": {"verified": True, "last_updated": "2026-02-01"}
        },
        {
            "id": "policy_002",
            "category": "fees",
            "question": "What are the account fees?",
            "answer": "Checking accounts have no monthly fee if you maintain a $500 minimum balance. ATM withdrawals at our network are free. Out-of-network ATM fees are $3. Overdraft protection fee is $35 per transaction.",
            "metadata": {"verified": True, "last_updated": "2026-02-01"}
        },
        {
            "id": "policy_003",
            "category": "transfers",
            "question": "How do I transfer money between accounts?",
            "answer": "You can transfer funds between your own accounts instantly using online banking or mobile app. Transfers to other banks take 1-3 business days via ACH. Wire transfers are processed same-day for a $25 fee.",
            "metadata": {"verified": True, "last_updated": "2026-02-01"}
        },
        {
            "id": "policy_004",
            "category": "loans",
            "question": "What loan options are available?",
            "answer": "We offer personal loans ($1,000-$50,000), auto loans, home mortgages, and home equity lines of credit. Interest rates vary from 3.5% to 18% depending on creditworthiness and loan term.",
            "metadata": {"verified": True, "last_updated": "2026-02-01"}
        },
        {
            "id": "policy_005",
            "category": "interest_rates",
            "question": "What is the current savings rate?",
            "answer": "Savings accounts earn 4.5% APY. Money market accounts earn 5.1% APY. Interest is compounded daily and credited monthly.",
            "metadata": {"verified": True, "last_updated": "2026-02-03"}
        },
        {
            "id": "policy_006",
            "category": "credit_cards",
            "question": "What credit card rewards do you offer?",
            "answer": "Our premium credit card offers 2% cashback on all purchases, 5% on dining and travel. Annual fee is $95. Additional perks include travel insurance and airport lounge access.",
            "metadata": {"verified": True, "last_updated": "2026-02-01"}
        },
        {
            "id": "policy_007",
            "category": "fraud_protection",
            "question": "How are fraudulent transactions handled?",
            "answer": "Fraudulent transactions are refunded within 24 hours. We monitor accounts 24/7 and notify you of suspicious activity. You have zero liability for unauthorized transactions.",
            "metadata": {"verified": True, "last_updated": "2026-02-01"}
        },
        {
            "id": "policy_008",
            "category": "security",
            "question": "How is my data protected?",
            "answer": "All customer data is encrypted with AES-256. We use multi-factor authentication, biometric login, and SSL/TLS for all connections. Regular security audits are performed quarterly.",
            "metadata": {"verified": True, "last_updated": "2026-02-01"}
        },
        {
            "id": "policy_009",
            "category": "business_hours",
            "question": "What are your business hours?",
            "answer": "Online banking available 24/7. Customer service: Monday-Friday 8am-8pm EST, Saturday 9am-5pm EST, Sunday 10am-4pm EST. Physical branches open Monday-Friday 9am-5pm, Saturday 9am-1pm.",
            "metadata": {"verified": True, "last_updated": "2026-02-01"}
        },
        {
            "id": "policy_010",
            "category": "direct_deposit",
            "question": "How do I set up direct deposit?",
            "answer": "Provide our routing number (021000021) and your account number to your employer. Direct deposits typically arrive on payday. You can set up multiple direct deposits for different income sources.",
            "metadata": {"verified": True, "last_updated": "2026-02-01"}
        },
    ]
    
    def __init__(self, embedding_model: str = "all-MiniLM-L6-v2"):
        """Initialize with embedding model"""
        self.embedding_model = SentenceTransformer(embedding_model)
        logger.info(f"Initialized BankingContextStore with {embedding_model}")
    
    def get_all_policies(self) -> List[Dict]:
        """Get all banking policies"""
        return self.BANKING_POLICIES
    
    def embed_policies(self) -> List[Tuple[str, np.ndarray, Dict]]:
        """Generate embeddings for all policies"""
        embeddings = []
        
        for policy in self.BANKING_POLICIES:
            # Create searchable text from policy
            text = f"{policy['question']} {policy['answer']}"
            
            # Generate embedding
            embedding = self.embedding_model.encode(text, convert_to_numpy=True)
            
            embeddings.append((
                policy["id"],
                embedding,
                policy
            ))
        
        logger.info(f"Generated embeddings for {len(embeddings)} policies")
        return embeddings


class EnhancedRAGService:
    """Enhanced RAG Service with Pinecone + Banking Context"""
    
    def __init__(self, 
                 pinecone_api_key: str,
                 pinecone_index_name: str = "banking-context",
                 embedding_model: str = "all-MiniLM-L6-v2"):
        """
        Initialize RAG service
        
        Args:
            pinecone_api_key: Pinecone API key
            pinecone_index_name: Index name for policies
            embedding_model: Embedding model name
        """
        self.embedding_model = SentenceTransformer(embedding_model)
        self.index_name = pinecone_index_name
        
        # Initialize Pinecone
        self.pc = Pinecone(api_key=pinecone_api_key)
        
        # Initialize banking context store
        self.context_store = BankingContextStore(embedding_model)
        
        # Get or create index
        self._ensure_index_exists()
        self.index = self.pc.Index(self.index_name)
        
        # Load policies into vector store
        self._load_policies_to_pinecone()
        
        logger.info("✓ EnhancedRAGService initialized")
    
    def _ensure_index_exists(self):
        """Create index if it doesn't exist"""
        try:
            indexes = self.pc.list_indexes()
            index_names = [idx.name for idx in indexes]
            
            if self.index_name not in index_names:
                logger.info(f"Creating Pinecone index: {self.index_name}")
                self.pc.create_index(
                    name=self.index_name,
                    dimension=384,  # all-MiniLM-L6-v2 dimension
                    metric="cosine",
                    spec={
                        "serverless": {
                            "cloud": "aws",
                            "region": "us-east-1"
                        }
                    }
                )
                logger.info(f"✓ Index created: {self.index_name}")
            else:
                logger.info(f"✓ Index exists: {self.index_name}")
        except Exception as e:
            logger.warning(f"Could not verify index: {e}")
    
    def _load_policies_to_pinecone(self):
        """Load banking policies into Pinecone"""
        try:
            logger.info("Loading banking policies to Pinecone...")
            
            embeddings = self.context_store.embed_policies()
            
            # Upsert to Pinecone in batches
            batch_size = 100
            for i in range(0, len(embeddings), batch_size):
                batch = embeddings[i:i + batch_size]
                
                vectors_to_upsert = [
                    (policy_id, embedding.tolist(), {"policy": json.dumps(policy)})
                    for policy_id, embedding, policy in batch
                ]
                
                self.index.upsert(vectors=vectors_to_upsert)
            
            logger.info(f"✓ Loaded {len(embeddings)} policies to Pinecone")
        except Exception as e:
            logger.error(f"Failed to load policies: {e}")
    
    def retrieve_context(self, query: str, top_k: int = 3) -> List[Dict]:
        """
        Retrieve relevant banking context for a query
        
        Args:
            query: User query
            top_k: Number of results to retrieve
            
        Returns:
            List of relevant policies with context
        """
        try:
            # Embed query
            query_embedding = self.embedding_model.encode(query, convert_to_numpy=True)
            
            # Search Pinecone
            results = self.index.query(
                vector=query_embedding.tolist(),
                top_k=top_k,
                include_metadata=True
            )
            
            context = []
            for match in results["matches"]:
                score = match["score"]
                metadata = match.get("metadata", {})
                
                # Parse policy from metadata
                if "policy" in metadata:
                    policy = json.loads(metadata["policy"])
                    context.append({
                        "score": score,
                        "question": policy.get("question", ""),
                        "answer": policy.get("answer", ""),
                        "category": policy.get("category", ""),
                        "id": match.get("id", "")
                    })
            
            logger.debug(f"Retrieved {len(context)} context items for query: {query[:50]}...")
            return context
        
        except Exception as e:
            logger.error(f"Failed to retrieve context: {e}")
            return []
    
    def augment_prompt(self, user_query: str, retrieved_context: List[Dict]) -> str:
        """
        Augment user query with retrieved banking context
        
        Args:
            user_query: Original user query
            retrieved_context: Retrieved banking policies
            
        Returns:
            Augmented prompt with context
        """
        augmented_prompt = f"""You are a professional banking assistant with access to the following banking policies and information:

BANKING CONTEXT:
"""
        
        for ctx in retrieved_context:
            augmented_prompt += f"\n- Q: {ctx['question']}\n  A: {ctx['answer']}\n"
        
        augmented_prompt += f"""
Using the above banking information as context, answer the following customer query:
CUSTOMER QUERY: {user_query}

RESPONSE: """
        
        return augmented_prompt
    
    def process_with_rag(self, user_query: str, llm_generator, top_k: int = 3) -> Dict:
        """
        Process query through full RAG pipeline
        
        Args:
            user_query: Customer query
            llm_generator: Function to generate LLM response
            top_k: Number of contexts to retrieve
            
        Returns:
            Response with context and generation
        """
        # Retrieve context
        context = self.retrieve_context(user_query, top_k=top_k)
        
        # Augment prompt
        augmented_prompt = self.augment_prompt(user_query, context)
        
        # Generate response
        try:
            response = llm_generator(augmented_prompt)
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            response = "Unable to generate response at this time."
        
        return {
            "query": user_query,
            "context_retrieved": len(context),
            "contexts": context,
            "augmented_prompt": augmented_prompt,
            "response": response,
            "confidence_score": np.mean([ctx["score"] for ctx in context]) if context else 0.0
        }


def initialize_rag_service(api_key: str) -> EnhancedRAGService:
    """Initialize RAG service with Pinecone"""
    try:
        rag_service = EnhancedRAGService(
            pinecone_api_key=api_key,
            pinecone_index_name="banking-context"
        )
        return rag_service
    except Exception as e:
        logger.error(f"Failed to initialize RAG service: {e}")
        raise
