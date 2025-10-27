"""
Model Manager for DQN Models
Handles saving, loading, and managing trained models with metadata
"""

import os
import json
from datetime import datetime
from pathlib import Path
import tensorflow as tf


class ModelManager:
    """Manages DQN model persistence and metadata"""
    
    def __init__(self, models_dir="models"):
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(exist_ok=True)
        self.metadata_file = "metadata.json"
    
    def save_model(self, agent, name=None, metadata=None):
        """
        Save a trained model with metadata
        
        Args:
            agent: DQNAgent instance
            name: Optional name for the model (auto-generated if None)
            metadata: Dict with training metadata (episodes, success_rate, etc.)
        
        Returns:
            Path to saved model directory
        """
        # Generate model name if not provided
        if name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            name = f"dqn_model_{timestamp}"
        
        # Create model directory
        model_dir = self.models_dir / name
        model_dir.mkdir(exist_ok=True)
        
        # Save the Keras model
        model_path = model_dir / "model.keras"
        agent.model.save(model_path)
        
        # Prepare metadata
        model_metadata = {
            "name": name,
            "created_at": datetime.now().isoformat(),
            "state_size": agent.state_size,
            "action_size": agent.action_size,
            "architecture": agent.architecture_name,  # Save architecture name
            "epsilon": float(agent.epsilon),
            "gamma": float(agent.gamma),
            "learning_rate": float(agent.learning_rate),
        }
        
        # Add custom metadata
        if metadata:
            model_metadata.update(metadata)
        
        # Save metadata
        metadata_path = model_dir / self.metadata_file
        with open(metadata_path, 'w') as f:
            json.dump(model_metadata, f, indent=2)
        
        # Removed print() - not thread-safe in PyQt6
        return model_dir
    
    def load_model(self, name, agent=None):
        """
        Load a trained model
        
        Args:
            name: Model name or path
            agent: Optional DQNAgent to load into (creates new if None)
        
        Returns:
            Tuple of (agent, metadata)
        """
        # Find model directory
        if os.path.isabs(name):
            model_dir = Path(name)
        else:
            model_dir = self.models_dir / name
        
        if not model_dir.exists():
            raise FileNotFoundError(f"Model not found: {model_dir}")
        
        # Load metadata
        metadata_path = model_dir / self.metadata_file
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
        else:
            metadata = {}
        
        # Load the Keras model
        model_path = model_dir / "model.keras"
        if not model_path.exists():
            # Try legacy .h5 format
            model_path = model_dir / "model.h5"
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found in: {model_dir}")
        
        loaded_model = tf.keras.models.load_model(model_path)
        
        # Create or update agent
        if agent is None:
            from dqn_agent import DQNAgent
            state_size = metadata.get('state_size', 27)
            action_size = metadata.get('action_size', 9)
            architecture = metadata.get('architecture', 'Large (128-64-32)')  # Default to Large
            agent = DQNAgent(state_size, action_size, architecture_name=architecture)
        
        agent.model = loaded_model
        agent.epsilon = metadata.get('epsilon', agent.epsilon)
        
        # Removed print() - not thread-safe in PyQt6
        return agent, metadata
    
    def list_models(self):
        """
        List all available models with their metadata
        
        Returns:
            List of (model_name, metadata) tuples, sorted by creation time (newest first)
        """
        models = []
        
        for model_dir in self.models_dir.iterdir():
            if model_dir.is_dir():
                metadata_path = model_dir / self.metadata_file
                if metadata_path.exists():
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                    models.append((model_dir.name, metadata))
                else:
                    # Model without metadata
                    models.append((model_dir.name, {"name": model_dir.name}))
        
        # Sort by creation time (newest first)
        models.sort(key=lambda x: x[1].get('created_at', ''), reverse=True)
        return models
    
    def get_latest_model(self):
        """
        Get the most recently trained model
        
        Returns:
            Tuple of (model_name, metadata) or (None, None) if no models
        """
        models = self.list_models()
        if models:
            return models[0]
        return None, None
    
    def delete_model(self, name):
        """Delete a model and its metadata"""
        model_dir = self.models_dir / name
        if model_dir.exists():
            import shutil
            shutil.rmtree(model_dir)
            # Removed print() - not thread-safe in PyQt6
            return True
        return False
    
    def get_model_info(self, name):
        """Get detailed information about a model"""
        model_dir = self.models_dir / name
        if not model_dir.exists():
            return None
        
        metadata_path = model_dir / self.metadata_file
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                return json.load(f)
        return {"name": name}
