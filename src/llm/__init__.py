"""LLM Service - Real LLM integration with Ollama and Together.ai"""

import json
import logging
from typing import Any, AsyncIterator, Dict, List, Optional

import httpx
from openai import AsyncOpenAI

from ..observability.metrics import track_model_inference
from ..observability.tracing import trace_function
from ..utils.config import settings
from ..llm_training.inference import CustomModelHandler

logger = logging.getLogger(__name__)


class LLMService:
    """Handle LLM interactions with multiple providers"""
    
    def __init__(self):
        self.ollama_base_url = settings.OLLAMA_BASE_URL
        self.ollama_model = settings.OLLAMA_MODEL
        self.together_api_key = settings.TOGETHER_API_KEY
        self.together_model = settings.TOGETHER_MODEL
        self.openai_api_key = settings.OPENAI_API_KEY
        self.openai_model = settings.OPENAI_MODEL
        self.llm_provider = settings.LLM_PROVIDER
        
        # Initialize clients
        self.openai_client = None
        if self.openai_api_key:
            self.openai_client = AsyncOpenAI(api_key=self.openai_api_key)
        
        self.together_client = None
        if self.together_api_key:
            self.together_client = AsyncOpenAI(
                api_key=self.together_api_key,
                base_url="https://api.together.xyz/v1"
            )
            
        self.custom_handler = None
        if self.llm_provider == "custom":
            self.custom_handler = CustomModelHandler()

    @property
    def model(self) -> str:
        if self.llm_provider == "ollama":
            return self.ollama_model
        if self.llm_provider == "together":
            return self.together_model
        if self.llm_provider == "openai":
            return self.openai_model
        return "unknown"
    
    @trace_function("llm_generate")
    async def generate_response(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 1000,
        temperature: float = 0.7,
        stream: bool = False
    ) -> str:
        """
        Generate response from LLM
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            stream: Whether to stream response
            
        Returns:
            Generated text response
        """
        if self.llm_provider == "ollama":
            return await self._generate_ollama(messages, max_tokens, temperature, stream)
        elif self.llm_provider == "together":
            return await self._generate_together(messages, max_tokens, temperature, stream)
        elif self.llm_provider == "openai":
            return await self._generate_openai(messages, max_tokens, temperature, stream)
        elif self.llm_provider == "custom":
            return await self._generate_custom(messages, max_tokens, temperature)
        else:
            raise ValueError(f"Unknown LLM provider: {self.llm_provider}")
    
    async def _generate_ollama(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int,
        temperature: float,
        stream: bool
    ) -> str:
        """Generate response using Ollama Chat API"""
        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                payload = {
                    "model": self.ollama_model,
                    "messages": messages,
                    "stream": False,
                    "options": {
                        "temperature": temperature,
                        "num_predict": max_tokens
                    }
                }
                
                response = await client.post(
                    f"{self.ollama_base_url}/api/chat",
                    json=payload
                )
                
                response.raise_for_status()
                result = response.json()
                
                # Extract content from chat response
                generated_text = result["message"]["content"]
                
                # Track metrics
                track_model_inference(
                    model=self.ollama_model,
                    duration=result.get("total_duration", 0) / 1e9,  # Convert to seconds
                    tokens=result.get("eval_count", 0),
                    cost=0.0  # Ollama is free
                )
                
                return generated_text
                
        except Exception as e:
            logger.error(f"Ollama generation failed: {e}")
            raise
    
    async def _generate_together(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int,
        temperature: float,
        stream: bool
    ) -> str:
        """Generate response using Together.ai"""
        try:
            if not self.together_client:
                raise ValueError("Together.ai API key not configured")
            
            response = await self.together_client.chat.completions.create(
                model=self.together_model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                stream=False
            )
            
            generated_text = response.choices[0].message.content
            
            # Track metrics
            usage = response.usage
            # Together.ai pricing: ~$0.2-0.6 per 1M tokens depending on model
            cost = (usage.total_tokens / 1_000_000) * 0.4
            
            track_model_inference(
                model_name=self.together_model,
                duration=0.0,  # Not provided by API
                tokens_generated=usage.completion_tokens,
                cost=cost
            )
            
            return generated_text
            
        except Exception as e:
            logger.error(f"Together.ai generation failed: {e}")
            raise
    
    async def _generate_openai(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int,
        temperature: float,
        stream: bool
    ) -> str:
        """Generate response using OpenAI"""
        try:
            if not self.openai_client:
                raise ValueError("OpenAI API key not configured")
            
            response = await self.openai_client.chat.completions.create(
                model=self.openai_model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                stream=False
            )
            
            generated_text = response.choices[0].message.content
            
            # Track metrics
            usage = response.usage
            # OpenAI pricing varies by model
            cost = self._calculate_openai_cost(self.openai_model, usage.prompt_tokens, usage.completion_tokens)
            
            track_model_inference(
                model_name=self.openai_model,
                duration=0.0,  # Not provided by API
                tokens_generated=usage.completion_tokens,
                cost=cost
            )
            
            return generated_text
            
        except Exception as e:
            logger.error(f"OpenAI generation failed: {e}")
            raise
    
            return generated_text
            
        except Exception as e:
            logger.error(f"OpenAI generation failed: {e}")
            raise

    async def _generate_custom(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int,
        temperature: float
    ) -> str:
        """Generate response using Custom Banking LLM"""
        try:
            if not self.custom_handler or not self.custom_handler.is_ready:
                return "Custom model is not ready. Please run training script first."
            
            # Simple conversion: just use the last user message for now
            # TODO: Better prompt formatting for custom model
            last_msg = messages[-1]["content"] if messages else ""
            
            generated_text = self.custom_handler.generate(
                prompt=last_msg,
                max_new_tokens=max_tokens,
                temperature=temperature
            )
            
            # Track metrics (approximate)
            track_model_inference(
                model="custom-banking-llm",
                duration=0.5, # Placeholder
                tokens=len(generated_text.split()),
                cost=0.0
            )
            
            return generated_text
            
        except Exception as e:
            logger.error(f"Custom model generation failed: {e}")
            return f"Error: {str(e)}"

    async def generate_stream(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 1000,
        temperature: float = 0.7
    ) -> AsyncIterator[str]:
        """
        Generate streaming response from LLM
        
        Args:
            messages: List of message dicts
            max_tokens: Maximum tokens
            temperature: Sampling temperature
            
        Yields:
            Text chunks
        """
        if self.llm_provider == "ollama":
            async for chunk in self._stream_ollama(messages, max_tokens, temperature):
                yield chunk
        elif self.llm_provider == "together":
            async for chunk in self._stream_together(messages, max_tokens, temperature):
                yield chunk
        elif self.llm_provider == "openai":
            async for chunk in self._stream_openai(messages, max_tokens, temperature):
                yield chunk
    
    async def _stream_ollama(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int,
        temperature: float
    ) -> AsyncIterator[str]:
        """Stream response from Ollama using Chat API"""
        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                payload = {
                    "model": self.ollama_model,
                    "messages": messages,
                    "stream": True,
                    "options": {
                        "temperature": temperature,
                        "num_predict": max_tokens
                    }
                }
                
                async with client.stream(
                    "POST",
                    f"{self.ollama_base_url}/api/chat",
                    json=payload
                ) as response:
                    response.raise_for_status()
                    async for line in response.aiter_lines():
                        if line:
                            try:
                                data = json.loads(line)
                                if "message" in data and "content" in data["message"]:
                                    yield data["message"]["content"]
                            except json.JSONDecodeError:
                                continue
                                
        except Exception as e:
            logger.error(f"Ollama streaming failed: {e}")
            raise
    
    async def _stream_together(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int,
        temperature: float
    ) -> AsyncIterator[str]:
        """Stream response from Together.ai"""
        try:
            if not self.together_client:
                raise ValueError("Together.ai API key not configured")
            
            stream = await self.together_client.chat.completions.create(
                model=self.together_model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                stream=True
            )
            
            async for chunk in stream:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
                    
        except Exception as e:
            logger.error(f"Together.ai streaming failed: {e}")
            raise
    
    async def _stream_openai(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int,
        temperature: float
    ) -> AsyncIterator[str]:
        """Stream response from OpenAI"""
        try:
            if not self.openai_client:
                raise ValueError("OpenAI API key not configured")
            
            stream = await self.openai_client.chat.completions.create(
                model=self.openai_model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                stream=True
            )
            
            async for chunk in stream:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
                    
        except Exception as e:
            logger.error(f"OpenAI streaming failed: {e}")
            raise
    
    def _messages_to_prompt(self, messages: List[Dict[str, str]]) -> str:
        """Convert OpenAI message format to single prompt"""
        prompt_parts = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            
            if role == "system":
                prompt_parts.append(f"System: {content}")
            elif role == "user":
                prompt_parts.append(f"User: {content}")
            elif role == "assistant":
                prompt_parts.append(f"Assistant: {content}")
        
        prompt_parts.append("Assistant:")
        return "\n\n".join(prompt_parts)
    
    def _calculate_openai_cost(self, model: str, prompt_tokens: int, completion_tokens: int) -> float:
        """Calculate OpenAI API cost"""
        # Pricing as of 2024 (per 1M tokens)
        pricing = {
            "gpt-4-turbo-preview": {"input": 10.0, "output": 30.0},
            "gpt-4": {"input": 30.0, "output": 60.0},
            "gpt-3.5-turbo": {"input": 0.5, "output": 1.5},
            "gpt-4o": {"input": 5.0, "output": 15.0},
            "gpt-4o-mini": {"input": 0.15, "output": 0.6}
        }
        
        model_pricing = pricing.get(model, {"input": 1.0, "output": 2.0})
        
        input_cost = (prompt_tokens / 1_000_000) * model_pricing["input"]
        output_cost = (completion_tokens / 1_000_000) * model_pricing["output"]
        
        return input_cost + output_cost
    
    async def generate_banking_response(
        self,
        user_message: str,
        customer_context: Dict[str, Any],
        conversation_history: List[Dict[str, str]] = None
    ) -> str:
        """
        Generate banking-specific response with context
        
        Args:
            user_message: User's question/request
            customer_context: Customer data from BankingService
            conversation_history: Previous messages
            
        Returns:
            Generated response
        """
        # Build system prompt with banking context
        system_prompt = self._build_banking_system_prompt(customer_context)
        
        # Build message list
        messages = [{"role": "system", "content": system_prompt}]
        
        # Add conversation history
        if conversation_history:
            messages.extend(conversation_history[-10:])  # Last 10 messages
        
        # Add current user message
        messages.append({"role": "user", "content": user_message})
        
        # Generate response
        response = await self.generate_response(
            messages=messages,
            max_tokens=1000,
            temperature=0.7
        )
        
        return response
    
    def _build_banking_system_prompt(self, customer_context: Dict[str, Any]) -> str:
        """Build system prompt with customer context"""
        customer = customer_context.get("customer", {})
        accounts = customer_context.get("accounts", [])
        recent_transactions = customer_context.get("recent_transactions", [])
        
        prompt = f"""You are a helpful banking assistant for a secure financial institution.

Customer Information:
- Name: {customer.get('first_name', '')} {customer.get('last_name', '')}
- Customer ID: {customer.get('customer_id', 'N/A')}
- KYC Status: {customer.get('kyc_status', 'pending')}
- Risk Score: {customer.get('risk_score', 0)}/100

Accounts:
"""
        
        for acc in accounts:
            prompt += f"- {acc.get('account_type', 'Unknown')} Account ({acc.get('account_number', 'N/A')}): ${acc.get('balance', 0):.2f}\n"
        
        if recent_transactions:
            prompt += f"\nRecent Transactions (Last {len(recent_transactions)}):\n"
            for txn in recent_transactions[:5]:
                prompt += f"- {txn.get('transaction_type', 'Unknown')}: ${txn.get('amount', 0):.2f} on {txn.get('created_at', 'N/A')}\n"
        
        prompt += """
Your role:
1. Answer banking questions accurately and clearly
2. Help with account inquiries, transactions, and financial advice
3. Detect and flag suspicious activity
4. Maintain professional and helpful tone
5. Protect customer privacy - never share sensitive data unnecessarily
6. Follow banking regulations and compliance requirements

Important:
- Always verify customer identity for sensitive operations
- Flag unusual transaction patterns
- Provide clear explanations of fees and terms
- Offer relevant financial products when appropriate
- If unsure, escalate to human banker

Respond naturally and professionally."""
        
        return prompt


# Global instance
llm_service = LLMService()
