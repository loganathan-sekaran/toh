"""
Model Selection Dialog
Browse and select trained models with their performance metrics
"""

from PyQt6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QPushButton, 
                              QLabel, QTableWidget, QTableWidgetItem, QHeaderView,
                              QMessageBox, QTextEdit)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont
from model_manager import ModelManager
from datetime import datetime


class ModelSelectionDialog(QDialog):
    """Dialog for selecting a trained model"""
    
    def __init__(self, parent=None, auto_select_latest=False):
        super().__init__(parent)
        self.model_manager = ModelManager()
        self.selected_model = None
        self.selected_metadata = None
        self.auto_select_latest = auto_select_latest
        
        self.setWindowTitle("Select Model")
        self.setMinimumSize(800, 500)
        self.init_ui()
        self.load_models()
        
        if self.auto_select_latest:
            self.select_latest_model()
    
    def init_ui(self):
        """Initialize the UI"""
        layout = QVBoxLayout(self)
        
        # Title
        title = QLabel("Select a Trained Model")
        title.setFont(QFont("Arial", 16, QFont.Weight.Bold))
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)
        
        # Instructions
        instructions = QLabel("Select a model to load for testing or evaluation")
        instructions.setFont(QFont("Arial", 10))
        instructions.setAlignment(Qt.AlignmentFlag.AlignCenter)
        instructions.setStyleSheet("color: #666; padding: 5px;")
        layout.addWidget(instructions)
        
        # Models table
        self.table = QTableWidget()
        self.table.setColumnCount(7)
        self.table.setHorizontalHeaderLabels([
            "Model Name", "Architecture", "Created", "Episodes", "Success Rate", "Avg Steps", "Epsilon"
        ])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self.table.itemSelectionChanged.connect(self.on_selection_changed)
        self.table.itemDoubleClicked.connect(self.accept)
        layout.addWidget(self.table)
        
        # Model details
        details_label = QLabel("Model Details:")
        details_label.setFont(QFont("Arial", 11, QFont.Weight.Bold))
        layout.addWidget(details_label)
        
        self.details_text = QTextEdit()
        self.details_text.setMaximumHeight(100)
        self.details_text.setReadOnly(True)
        self.details_text.setPlaceholderText("Select a model to view details")
        layout.addWidget(self.details_text)
        
        # Buttons
        button_layout = QHBoxLayout()
        
        refresh_btn = QPushButton("🔄 Refresh")
        refresh_btn.clicked.connect(self.load_models)
        button_layout.addWidget(refresh_btn)
        
        preview_btn = QPushButton("👁️ Preview Architecture")
        preview_btn.setToolTip("Preview the neural network architecture")
        preview_btn.setStyleSheet("""
            QPushButton {
                background-color: #17a2b8;
                color: white;
                padding: 8px 15px;
            }
            QPushButton:hover {
                background-color: #138496;
            }
        """)
        preview_btn.clicked.connect(self.preview_selected_architecture)
        button_layout.addWidget(preview_btn)
        
        delete_btn = QPushButton("🗑️ Delete")
        delete_btn.clicked.connect(self.delete_selected_model)
        button_layout.addWidget(delete_btn)
        
        button_layout.addStretch()
        
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        button_layout.addWidget(cancel_btn)
        
        load_btn = QPushButton("Load Model")
        load_btn.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                padding: 8px 20px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """)
        load_btn.clicked.connect(self.accept)
        button_layout.addWidget(load_btn)
        
        layout.addLayout(button_layout)
    
    def load_models(self):
        """Load and display available models"""
        self.table.setRowCount(0)
        models = self.model_manager.list_models()
        
        if not models:
            self.table.setRowCount(1)
            item = QTableWidgetItem("No trained models found")
            item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.table.setItem(0, 0, item)
            self.table.setSpan(0, 0, 1, 7)
            return
        
        self.table.setRowCount(len(models))
        
        for row, (model_name, metadata) in enumerate(models):
            # Model name
            self.table.setItem(row, 0, QTableWidgetItem(model_name))
            
            # Architecture
            architecture = metadata.get('architecture', 'Unknown')
            self.table.setItem(row, 1, QTableWidgetItem(architecture))
            
            # Created date
            created_at = metadata.get('created_at', 'Unknown')
            if created_at != 'Unknown':
                try:
                    dt = datetime.fromisoformat(created_at)
                    created_at = dt.strftime("%Y-%m-%d %H:%M")
                except:
                    pass
            self.table.setItem(row, 2, QTableWidgetItem(created_at))
            
            # Episodes
            episodes = metadata.get('episodes', '-')
            self.table.setItem(row, 3, QTableWidgetItem(str(episodes)))
            
            # Success rate
            success_rate = metadata.get('success_rate', '-')
            if success_rate != '-':
                success_rate = f"{success_rate:.1f}%"
            self.table.setItem(row, 4, QTableWidgetItem(str(success_rate)))
            
            # Average steps
            avg_steps = metadata.get('avg_steps', '-')
            if avg_steps != '-':
                avg_steps = f"{avg_steps:.1f}"
            self.table.setItem(row, 5, QTableWidgetItem(str(avg_steps)))
            
            # Epsilon
            epsilon = metadata.get('epsilon', '-')
            if epsilon != '-':
                epsilon = f"{epsilon:.3f}"
            self.table.setItem(row, 6, QTableWidgetItem(str(epsilon)))
            
            # Store metadata in first cell
            self.table.item(row, 0).setData(Qt.ItemDataRole.UserRole, (model_name, metadata))
    
    def on_selection_changed(self):
        """Handle model selection change"""
        selected_items = self.table.selectedItems()
        if not selected_items:
            self.details_text.clear()
            return
        
        row = selected_items[0].row()
        data = self.table.item(row, 0).data(Qt.ItemDataRole.UserRole)
        
        if data:
            model_name, metadata = data
            self.selected_model = model_name
            self.selected_metadata = metadata
            
            # Display detailed metadata
            details = f"<b>Model:</b> {model_name}<br>"
            details += f"<b>Created:</b> {metadata.get('created_at', 'Unknown')}<br>"
            details += f"<b>State Size:</b> {metadata.get('state_size', '-')}<br>"
            details += f"<b>Action Size:</b> {metadata.get('action_size', '-')}<br>"
            details += f"<b>Gamma:</b> {metadata.get('gamma', '-')}<br>"
            details += f"<b>Learning Rate:</b> {metadata.get('learning_rate', '-')}<br>"
            
            if 'notes' in metadata:
                details += f"<b>Notes:</b> {metadata['notes']}<br>"
            
            self.details_text.setHtml(details)
    
    def delete_selected_model(self):
        """Delete the selected model"""
        if not self.selected_model:
            QMessageBox.warning(self, "No Selection", "Please select a model to delete.")
            return
        
        reply = QMessageBox.question(
            self,
            "Confirm Delete",
            f"Are you sure you want to delete model '{self.selected_model}'?\nThis action cannot be undone.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            if self.model_manager.delete_model(self.selected_model):
                QMessageBox.information(self, "Success", f"Model '{self.selected_model}' deleted.")
                self.load_models()
            else:
                QMessageBox.warning(self, "Error", "Failed to delete model.")
    
    def select_latest_model(self):
        """Auto-select the latest model"""
        if self.table.rowCount() > 0:
            self.table.selectRow(0)
            self.on_selection_changed()
    
    def preview_selected_architecture(self):
        """Preview the architecture of the selected model."""
        if not self.selected_model or not self.selected_metadata:
            QMessageBox.warning(self, "No Selection", "Please select a model to preview its architecture.")
            return
        
        try:
            # Load the model
            agent, metadata = self.model_manager.load_model(self.selected_model)
            
            # Show architecture
            from model_visualizer import ModelVisualizerWidget
            
            dialog = QDialog(self)
            arch_name = metadata.get('architecture', 'Unknown')
            dialog.setWindowTitle(f"Architecture: {arch_name}")
            dialog.setMinimumSize(900, 650)
            
            layout = QVBoxLayout(dialog)
            
            # Model info
            info_text = f"<h3>{self.selected_model}</h3>"
            info_text += f"<p><b>Architecture:</b> {arch_name}<br>"
            info_text += f"<b>Created:</b> {metadata.get('created_at', 'Unknown')}<br>"
            info_text += f"<b>Episodes Trained:</b> {metadata.get('episodes', '-')}<br>"
            info_text += f"<b>Success Rate:</b> {metadata.get('success_rate', '-'):.1f}%<br>"
            info_text += f"<b>Avg Steps:</b> {metadata.get('avg_steps', '-'):.1f}<br>"
            info_text += f"<b>Total Parameters:</b> {agent.model.count_params():,}</p>"
            
            info_label = QLabel(info_text)
            info_label.setFont(QFont("Arial", 11))
            info_label.setStyleSheet("padding: 10px; background-color: #f8f9fa; border-radius: 5px;")
            layout.addWidget(info_label)
            
            # Model visualizer
            viz = ModelVisualizerWidget()
            viz_info = f"State: {metadata.get('state_size', '-')} | Actions: {metadata.get('action_size', '-')} | {agent.model.count_params():,} params"
            viz.set_model(agent.model, viz_info)
            layout.addWidget(viz)
            
            # Layer details
            layers_text = "<b>Layer Details:</b><br>"
            for i, layer in enumerate(agent.model.layers):
                layer_type = layer.__class__.__name__
                if hasattr(layer, 'units'):
                    layers_text += f"• Layer {i+1}: {layer_type} - {layer.units} units"
                    if hasattr(layer, 'activation'):
                        layers_text += f" ({layer.activation.__name__})"
                    layers_text += "<br>"
                elif layer_type == 'Dropout':
                    layers_text += f"• Layer {i+1}: Dropout - rate {layer.rate}<br>"
            
            details_label = QLabel(layers_text)
            details_label.setWordWrap(True)
            details_label.setFont(QFont("Arial", 10))
            details_label.setStyleSheet("padding: 10px; background-color: #fff; border: 1px solid #ddd; border-radius: 5px;")
            layout.addWidget(details_label)
            
            # Close button
            close_btn = QPushButton("Close")
            close_btn.setStyleSheet("""
                QPushButton {
                    background-color: #6c757d;
                    color: white;
                    padding: 8px 20px;
                    border-radius: 4px;
                }
                QPushButton:hover {
                    background-color: #5a6268;
                }
            """)
            close_btn.clicked.connect(dialog.accept)
            layout.addWidget(close_btn)
            
            dialog.exec()
            
        except Exception as e:
            QMessageBox.critical(self, "Preview Error", f"Could not preview architecture:\n{str(e)}")
    
    def get_selected_model(self):
        """Get the selected model name and metadata"""
        return self.selected_model, self.selected_metadata
