"""
Model Architectures Package
Auto-discovers and registers all available neural network architectures
"""

from .base_architecture import ModelArchitecture
from .model_factory import ModelFactory

__all__ = ['ModelArchitecture', 'ModelFactory']
