"""
Extra Large Model Architecture
Maximum capacity for very complex problems
"""

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout
import tensorflow as tf
from .base_architecture import ModelArchitecture


class ExtraLargeModel(ModelArchitecture):
    """
    Extra Large 256-128-64 architecture
    Good for: Maximum learning capacity, complex state spaces
    """
    
    def __init__(self):
        super().__init__()
        self.name = "Extra Large (256-128-64)"
        self.description = "Extra large model: 256→128→64 hidden layers with dropout. Maximum capacity for complex problems."
        self.recommended_episodes = 2000
        self.complexity = "Very High"
    
    def build(self, state_size: int, action_size: int, learning_rate: float) -> Sequential:
        """Build the extra large 256-128-64 model with dropout."""
        model = Sequential([
            Dense(256, input_dim=state_size, activation='relu', 
                  kernel_initializer='he_uniform'),
            Dropout(0.2),  # Prevent overfitting
            Dense(128, activation='relu', kernel_initializer='he_uniform'),
            Dropout(0.2),
            Dense(64, activation='relu', kernel_initializer='he_uniform'),
            Dense(action_size, activation='linear', 
                  kernel_initializer='he_uniform')
        ])
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss='huber',
            metrics=['mae']
        )
        
        return model
