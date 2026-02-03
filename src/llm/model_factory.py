"""Factory for creating and managing model instances"""

import os
from typing import Dict, Optional, Type

from src.llm.models.base_adapter import ModelAdapter
from src.llm.models.custom_transformer_adapter import CustomTransformerAdapter
from src.llm.models.gemini_adapter import GeminiAdapter
from src.llm.models.qlora_adapter import QLoRASambaNovaAdapter


class ModelFactory:
    """
    Factory for creating model instances with singleton management.
    
    Implements:
    - Registry pattern for model types
    - Singleton instances (one per model type)
    - Dynamic model loading from .env
    - Runtime model registration
    """
    
    # Registry of available models
    REGISTRY: Dict[str, Type[ModelAdapter]] = {
        'custom': CustomTransformerAdapter,
        'qlora': QLoRASambaNovaAdapter,
        'gemini': GeminiAdapter,
    }
    
    # Singleton instances
    _instances: Dict[str, ModelAdapter] = {}
    
    @classmethod
    def create(cls, model_name: str) -> ModelAdapter:
        """
        Create or retrieve model instance (singleton per type).
        
        Args:
            model_name: Name of model type ('custom', 'qlora', 'gemini')
            
        Returns:
            ModelAdapter instance
            
        Raises:
            ValueError: If model_name not in registry
        """
        
        # Return existing instance
        if model_name in cls._instances:
            return cls._instances[model_name]
        
        # Validate model exists
        if model_name not in cls.REGISTRY:
            available = ', '.join(cls.REGISTRY.keys())
            raise ValueError(
                f"Unknown model: {model_name}. "
                f"Available: {available}"
            )
        
        # Create new instance
        adapter_class = cls.REGISTRY[model_name]
        instance = adapter_class()
        cls._instances[model_name] = instance
        
        print(f"✓ Created {model_name} adapter: {instance.version}")
        
        return instance
    
    @classmethod
    def get_active_model(cls) -> ModelAdapter:
        """
        Get active model from .env configuration.
        
        Reads LLM_MODEL_VERSION from environment and loads corresponding model.
        
        Returns:
            ModelAdapter instance for active model
        """
        
        # Get from .env
        model_version = os.getenv(
            'LLM_MODEL_VERSION',
            'custom_model_built_0.0.0'
        )
        
        # Parse model type from version string
        # Format: "<model_type>_<description>_<semver>"
        # Examples:
        # - "custom_model_built_0.0.0" → "custom"
        # - "qlora_vllm_finetuned_model_v1.0.0" → "qlora"
        # - "gemini_2.0_flash_v1.0.0" → "gemini"
        
        model_type = model_version.split('_')[0]
        
        return cls.create(model_type)
    
    @classmethod
    def register(cls, name: str, adapter_class: Type[ModelAdapter]):
        """
        Register new model type at runtime.
        
        Args:
            name: Name for model type
            adapter_class: ModelAdapter subclass
        """
        
        cls.REGISTRY[name] = adapter_class
        print(f"✓ Registered new model type: {name}")
    
    @classmethod
    def list_models(cls) -> Dict[str, str]:
        """
        List all available models and their status.
        
        Returns:
            Dict with model name as key, status as value
        """
        
        status = {}
        
        for name in cls.REGISTRY.keys():
            try:
                instance = cls.create(name)
                health = __import__('asyncio').run(instance.health_check())
                status[name] = "healthy" if health else "unhealthy"
            except Exception as e:
                status[name] = f"error: {str(e)[:50]}"
        
        return status
    
    @classmethod
    def clear_cache(cls):
        """Clear all cached instances (use with caution)"""
        cls._instances.clear()
        print("⚠️ Model cache cleared")
