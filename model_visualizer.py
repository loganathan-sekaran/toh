"""
Neural Network Architecture Visualizer for DQN Models
Displays layers, nodes, and connections in PyQt6
"""

from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel, QScrollArea
from PyQt6.QtCore import Qt, QRectF, QPointF
from PyQt6.QtGui import QPainter, QColor, QPen, QBrush, QFont, QPainterPath
import math


class ModelArchitectureCanvas(QWidget):
    """Canvas widget that draws the neural network architecture"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.layers = []  # List of layer configs: [(name, num_nodes), ...]
        self.connections = []  # List of connections between layers
        self.setMinimumSize(800, 400)
        
        # Visual settings
        self.node_radius = 8
        self.layer_spacing = 150
        self.node_spacing = 30
        self.max_nodes_display = 10  # Max nodes to show per layer (for large layers)
        
        # Colors
        self.bg_color = QColor(250, 250, 255)
        self.input_color = QColor(100, 200, 100)
        self.hidden_color = QColor(100, 150, 255)
        self.output_color = QColor(255, 150, 100)
        self.connection_color = QColor(150, 150, 150, 80)
        self.text_color = QColor(50, 50, 50)
    
    def set_model(self, model):
        """Extract layer information from Keras model"""
        self.layers = []
        
        if model is None:
            self.update()
            return
        
        try:
            # Extract layer information from Keras model
            for i, layer in enumerate(model.layers):
                layer_name = layer.name
                layer_type = layer.__class__.__name__
                
                # Get number of units/nodes
                if hasattr(layer, 'units'):
                    num_nodes = layer.units
                elif hasattr(layer, 'output_shape'):
                    output_shape = layer.output_shape
                    if isinstance(output_shape, tuple) and len(output_shape) > 1:
                        num_nodes = output_shape[-1]
                    else:
                        num_nodes = 1
                else:
                    num_nodes = 1
                
                self.layers.append({
                    'name': layer_name,
                    'type': layer_type,
                    'nodes': num_nodes,
                    'index': i
                })
        except Exception as e:
            print(f"Error extracting model architecture: {e}")
        
        self.update()
    
    def paintEvent(self, event):
        """Draw the neural network"""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        # Background
        painter.fillRect(self.rect(), self.bg_color)
        
        if not self.layers:
            # No model to display
            painter.setPen(QPen(self.text_color))
            painter.setFont(QFont("Arial", 14))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, 
                           "No model loaded\nTrain or load a model to visualize")
            return
        
        # Calculate layout
        width = self.width()
        height = self.height()
        
        num_layers = len(self.layers)
        total_width = (num_layers - 1) * self.layer_spacing
        start_x = (width - total_width) / 2
        
        # Store node positions for drawing connections
        node_positions = []
        
        # Draw connections first (so they appear behind nodes)
        painter.setPen(QPen(self.connection_color, 1))
        for layer_idx in range(len(self.layers) - 1):
            layer = self.layers[layer_idx]
            next_layer = self.layers[layer_idx + 1]
            
            curr_nodes = min(layer['nodes'], self.max_nodes_display)
            next_nodes = min(next_layer['nodes'], self.max_nodes_display)
            
            layer_x = start_x + layer_idx * self.layer_spacing
            next_layer_x = start_x + (layer_idx + 1) * self.layer_spacing
            
            curr_height = (curr_nodes - 1) * self.node_spacing
            next_height = (next_nodes - 1) * self.node_spacing
            
            curr_start_y = (height - curr_height) / 2
            next_start_y = (height - next_height) / 2
            
            # Draw sample connections (not all to avoid clutter)
            step = max(1, curr_nodes // 5)
            for i in range(0, curr_nodes, step):
                from_y = curr_start_y + i * self.node_spacing
                for j in range(0, next_nodes, step):
                    to_y = next_start_y + j * self.node_spacing
                    painter.drawLine(int(layer_x), int(from_y), 
                                   int(next_layer_x), int(to_y))
        
        # Draw layers and nodes
        for layer_idx, layer in enumerate(self.layers):
            layer_x = start_x + layer_idx * self.layer_spacing
            
            # Determine layer color
            if layer_idx == 0:
                color = self.input_color
            elif layer_idx == len(self.layers) - 1:
                color = self.output_color
            else:
                color = self.hidden_color
            
            # Calculate node positions
            num_nodes = layer['nodes']
            display_nodes = min(num_nodes, self.max_nodes_display)
            layer_height = (display_nodes - 1) * self.node_spacing
            start_y = (height - layer_height) / 2
            
            # Draw nodes
            painter.setBrush(QBrush(color))
            painter.setPen(QPen(color.darker(150), 2))
            
            for node_idx in range(display_nodes):
                node_y = start_y + node_idx * self.node_spacing
                painter.drawEllipse(QPointF(layer_x, node_y), 
                                  self.node_radius, self.node_radius)
            
            # Draw ellipsis if too many nodes
            if num_nodes > self.max_nodes_display:
                painter.setPen(QPen(self.text_color))
                painter.setFont(QFont("Arial", 12, QFont.Weight.Bold))
                ellipsis_y = start_y + display_nodes * self.node_spacing
                painter.drawText(QRectF(layer_x - 15, ellipsis_y, 30, 20),
                               Qt.AlignmentFlag.AlignCenter, "⋮")
            
            # Draw layer label
            painter.setPen(QPen(self.text_color))
            painter.setFont(QFont("Arial", 10, QFont.Weight.Bold))
            label_y = start_y - 30
            painter.drawText(QRectF(layer_x - 60, label_y, 120, 20),
                           Qt.AlignmentFlag.AlignCenter, layer['type'])
            
            # Draw node count
            painter.setFont(QFont("Arial", 9))
            count_y = start_y - 15
            painter.drawText(QRectF(layer_x - 60, count_y, 120, 15),
                           Qt.AlignmentFlag.AlignCenter, f"({num_nodes} nodes)")


class ModelVisualizerWidget(QWidget):
    """Complete model visualizer widget with title and canvas"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()
    
    def init_ui(self):
        """Initialize the UI"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        
        # Title
        title = QLabel("Neural Network Architecture")
        title.setFont(QFont("Arial", 14, QFont.Weight.Bold))
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)
        
        # Canvas
        self.canvas = ModelArchitectureCanvas()
        scroll = QScrollArea()
        scroll.setWidget(self.canvas)
        scroll.setWidgetResizable(True)
        layout.addWidget(scroll)
        
        # Info label
        self.info_label = QLabel("No model loaded")
        self.info_label.setFont(QFont("Arial", 10))
        self.info_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.info_label.setStyleSheet("color: #666; padding: 5px;")
        layout.addWidget(self.info_label)
    
    def set_model(self, model, model_info=""):
        """Update the displayed model"""
        self.canvas.set_model(model)
        
        if model is None:
            self.info_label.setText("No model loaded")
        else:
            # Extract model summary info
            try:
                total_params = model.count_params()
                num_layers = len(model.layers)
                info_text = f"Layers: {num_layers} | Parameters: {total_params:,}"
                if model_info:
                    info_text += f" | {model_info}"
                self.info_label.setText(info_text)
            except:
                self.info_label.setText("Model loaded")
