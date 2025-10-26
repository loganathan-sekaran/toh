"""
Model Factory - Auto-discovers and registers all available model architectures
"""

import os
import importlib
import inspect
from pathlib import Path
from typing import Dict, List, Type
from .base_architecture import ModelArchitecture


class ModelFactory:
    """
    Factory class that auto-discovers and instantiates model architectures.
    """
    
    _architectures: Dict[str, Type[ModelArchitecture]] = {}
    _initialized = False
    
    @classmethod
    def initialize(cls):
        """Auto-discover all model architectures in this package."""
        if cls._initialized:
            return
        
        # Get the directory containing model architecture files
        arch_dir = Path(__file__).parent
        
        # Find all Python files except __init__ and base
        for file in arch_dir.glob("*.py"):
            if file.stem in ['__init__', 'base_architecture', 'model_factory']:
                continue
            
            try:
                # Import the module
                module_name = f"model_architectures.{file.stem}"
                module = importlib.import_module(module_name)
                
                # Find all classes that inherit from ModelArchitecture
                for name, obj in inspect.getmembers(module, inspect.isclass):
                    if (issubclass(obj, ModelArchitecture) and 
                        obj is not ModelArchitecture and
                        obj.__module__ == module_name):
                        
                        # Register the architecture
                        instance = obj()
                        cls._architectures[instance.name] = obj
                        
            except Exception as e:
                print(f"Warning: Failed to load architecture from {file.stem}: {e}")
        
        cls._initialized = True
    
    @classmethod
    def get_architecture_names(cls) -> List[str]:
        """Get list of all available architecture names."""
        cls.initialize()
        return sorted(cls._architectures.keys())
    
    @classmethod
    def get_architecture(cls, name: str) -> ModelArchitecture:
        """
        Get an instance of a model architecture by name.
        
        Args:
            name: Name of the architecture (e.g., "Large (128-64-32)")
            
        Returns:
            Instance of the requested ModelArchitecture
            
        Raises:
            ValueError: If architecture name is not found
        """
        cls.initialize()
        
        if name not in cls._architectures:
            available = ", ".join(cls.get_architecture_names())
            raise ValueError(f"Architecture '{name}' not found. Available: {available}")
        
        return cls._architectures[name]()
    
    @classmethod
    def get_all_architectures(cls) -> Dict[str, ModelArchitecture]:
        """Get all available architectures as a dict {name: instance}."""
        cls.initialize()
        return {name: arch_class() for name, arch_class in cls._architectures.items()}
    
    @classmethod
    def get_architecture_info(cls, name: str) -> dict:
        """Get information about a specific architecture."""
        arch = cls.get_architecture(name)
        return arch.get_info()
    
    @classmethod
    def list_architectures(cls) -> None:
        """Print all available architectures with details."""
        cls.initialize()
        print("\n📐 Available Model Architectures:")
        print("=" * 80)
        
        for name in sorted(cls._architectures.keys()):
            arch = cls.get_architecture(name)
            info = arch.get_info()
            print(f"\n🔹 {info['name']}")
            print(f"   Description: {info['description']}")
            print(f"   Complexity: {info['complexity']}")
            print(f"   Recommended Episodes: {info['recommended_episodes']}")
        
        print("\n" + "=" * 80)


# Auto-initialize when module is imported
ModelFactory.initialize()
