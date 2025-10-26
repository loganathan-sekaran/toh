"""
Small Model Architecture (Original)
Fast training, good for quick experiments
"""

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
import tensorflow as tf
from .base_architecture import ModelArchitecture


class SmallModel(ModelArchitecture):
    """
    Small 24-24 architecture (original model)
    Good for: Quick experiments, baseline comparison
    """
    
    def __init__(self):
        super().__init__()
        self.name = "Small (24-24)"
        self.description = "Original small model: 24→24 hidden layers. Fast training, baseline performance."
        self.recommended_episodes = 500
        self.complexity = "Low"
    
    def build(self, state_size: int, action_size: int, learning_rate: float) -> Sequential:
        """Build the small 24-24 model."""
        model = Sequential([
            Dense(24, input_dim=state_size, activation='relu'),
            Dense(24, activation='relu'),
            Dense(action_size, activation='linear')
        ])
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss='mse',
            metrics=['mae']
        )
        
        return model
