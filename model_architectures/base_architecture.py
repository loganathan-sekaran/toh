"""
Base class for all model architectures.
Each model architecture defines how to build a neural network.
"""

from abc import ABC, abstractmethod
from tensorflow.keras import Sequential


class ModelArchitecture(ABC):
    """
    Abstract base class for model architectures.
    All custom architectures should inherit from this class.
    """
    
    def __init__(self):
        self.name = self.__class__.__name__
        self.description = "No description provided"
        self.recommended_episodes = 1000
        self.complexity = "Medium"  # Low, Medium, High, Very High
    
    @abstractmethod
    def build(self, state_size: int, action_size: int, learning_rate: float) -> Sequential:
        """
        Build and return a Keras Sequential model.
        
        Args:
            state_size: Size of the input state
            action_size: Number of possible actions
            learning_rate: Learning rate for the optimizer
            
        Returns:
            Compiled Keras Sequential model
        """
        pass
    
    def get_info(self) -> dict:
        """Return information about this architecture."""
        return {
            'name': self.name,
            'description': self.description,
            'recommended_episodes': self.recommended_episodes,
            'complexity': self.complexity
        }
    
    def __str__(self):
        return f"{self.name}: {self.description}"
    
    def __repr__(self):
        return f"<ModelArchitecture: {self.name}>"
