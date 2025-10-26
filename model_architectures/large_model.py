"""
Large Model Architecture (Current Improved)
High capacity for complex learning
"""

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
import tensorflow as tf
from .base_architecture import ModelArchitecture


class LargeModel(ModelArchitecture):
    """
    Large 128-64-32 architecture (current improved model)
    Good for: Best performance, handles complexity well
    """
    
    def __init__(self):
        super().__init__()
        self.name = "Large (128-64-32)"
        self.description = "Large funnel model: 128→64→32 hidden layers. Best performance, learns optimal solutions."
        self.recommended_episodes = 1500
        self.complexity = "High"
    
    def build(self, state_size: int, action_size: int, learning_rate: float) -> Sequential:
        """Build the large 128-64-32 model."""
        model = Sequential([
            Dense(128, input_dim=state_size, activation='relu', 
                  kernel_initializer='he_uniform'),
            Dense(64, activation='relu', kernel_initializer='he_uniform'),
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
