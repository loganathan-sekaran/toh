"""
Medium Model Architecture
Balanced performance and training speed
"""

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
import tensorflow as tf
from .base_architecture import ModelArchitecture


class MediumModel(ModelArchitecture):
    """
    Medium 64-32 architecture
    Good for: Balanced training time and performance
    """
    
    def __init__(self):
        super().__init__()
        self.name = "Medium (64-32)"
        self.description = "Balanced model: 64→32 hidden layers. Good performance with reasonable training time."
        self.recommended_episodes = 1000
        self.complexity = "Medium"
    
    def build(self, state_size: int, action_size: int, learning_rate: float) -> Sequential:
        """Build the medium 64-32 model."""
        model = Sequential([
            Dense(64, input_dim=state_size, activation='relu', 
                  kernel_initializer='he_uniform'),
            Dense(32, activation='relu', kernel_initializer='he_uniform'),
            Dense(action_size, activation='linear', 
                  kernel_initializer='he_uniform')
        ])
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss='huber',
            metrics=['mae']
        )
        
        return model
