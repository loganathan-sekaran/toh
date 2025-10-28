"""
Model Selection Dialog
Browse and select trained models with their performance metrics
"""

from PyQt6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QPushButton, 
                              QLabel, QTableWidget, QTableWidgetItem, QHeaderView,
                              QMessageBox, QTextEdit, QInputDialog, QLineEdit)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont, QColor
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
        self.table.setColumnCount(8)
        self.table.setHorizontalHeaderLabels([
            "★", "Model Name", "Architecture", "Created", "Episodes", "Success Rate", "Avg Steps", "Epsilon"
        ])
        
        # Make table sortable
        self.table.setSortingEnabled(True)
        
        # Set column resize modes
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Fixed)  # Bookmark column fixed width
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Interactive)  # Model name resizable
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.Interactive)  # Architecture resizable
        for i in range(3, 8):
            header.setSectionResizeMode(i, QHeaderView.ResizeMode.Interactive)
        
        self.table.setColumnWidth(0, 40)  # Bookmark column width
        
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
        
        bookmark_btn = QPushButton("⭐ Bookmark")
        bookmark_btn.setToolTip("Toggle bookmark for selected model")
        bookmark_btn.clicked.connect(self.toggle_bookmark)
        button_layout.addWidget(bookmark_btn)
        
        comment_btn = QPushButton("💬 Comment")
        comment_btn.setToolTip("Add or edit comment for selected model")
        comment_btn.clicked.connect(self.edit_comment)
        button_layout.addWidget(comment_btn)
        
        rename_btn = QPushButton("✏️ Rename")
        rename_btn.setToolTip("Rename the selected model")
        rename_btn.clicked.connect(self.rename_model)
        button_layout.addWidget(rename_btn)
        
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
        # Disable sorting while loading to prevent issues
        self.table.setSortingEnabled(False)
        self.table.setRowCount(0)
        models = self.model_manager.list_models()
        
        if not models:
            self.table.setRowCount(1)
            item = QTableWidgetItem("No trained models found")
            item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.table.setItem(0, 0, item)
            self.table.setSpan(0, 0, 1, 8)
            return
        
        self.table.setRowCount(len(models))
        
        for row, (model_name, metadata) in enumerate(models):
            # Bookmark indicator
            bookmark_item = QTableWidgetItem()
            is_bookmarked = metadata.get('bookmarked', False)
            bookmark_item.setText("⭐" if is_bookmarked else "")
            bookmark_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            if is_bookmarked:
                bookmark_item.setBackground(QColor("#fff9c4"))
            self.table.setItem(row, 0, bookmark_item)
            
            # Model name
            name_item = QTableWidgetItem(model_name)
            name_item.setData(Qt.ItemDataRole.UserRole, (model_name, metadata))  # Store data in name column
            self.table.setItem(row, 1, name_item)
            
            # Architecture
            architecture = metadata.get('architecture', 'Unknown')
            self.table.setItem(row, 2, QTableWidgetItem(architecture))
            
            # Created date
            created_at = metadata.get('created_at', 'Unknown')
            if created_at != 'Unknown':
                try:
                    dt = datetime.fromisoformat(created_at)
                    created_at = dt.strftime("%Y-%m-%d %H:%M")
                except:
                    pass
            self.table.setItem(row, 3, QTableWidgetItem(created_at))
            
            # Episodes
            episodes = metadata.get('episodes', '-')
            episodes_item = QTableWidgetItem()
            episodes_item.setData(Qt.ItemDataRole.DisplayRole, episodes)
            self.table.setItem(row, 4, episodes_item)
            
            # Success rate
            success_rate = metadata.get('success_rate', '-')
            success_item = QTableWidgetItem()
            if success_rate != '-':
                success_item.setData(Qt.ItemDataRole.DisplayRole, float(success_rate))
                success_item.setText(f"{success_rate:.1f}%")
            else:
                success_item.setText('-')
            self.table.setItem(row, 5, success_item)
            
            # Average steps
            avg_steps = metadata.get('avg_steps', '-')
            steps_item = QTableWidgetItem()
            if avg_steps != '-':
                steps_item.setData(Qt.ItemDataRole.DisplayRole, float(avg_steps))
                steps_item.setText(f"{avg_steps:.1f}")
            else:
                steps_item.setText('-')
            self.table.setItem(row, 6, steps_item)
            
            # Epsilon
            epsilon = metadata.get('epsilon', '-')
            epsilon_item = QTableWidgetItem()
            if epsilon != '-':
                epsilon_item.setData(Qt.ItemDataRole.DisplayRole, float(epsilon))
                epsilon_item.setText(f"{epsilon:.3f}")
            else:
                epsilon_item.setText('-')
            self.table.setItem(row, 7, epsilon_item)
        
        # Re-enable sorting
        self.table.setSortingEnabled(True)
    
    def on_selection_changed(self):
        """Handle model selection change"""
        selected_items = self.table.selectedItems()
        if not selected_items:
            self.details_text.clear()
            return
        
        row = selected_items[0].row()
        # Get data from name column (column 1)
        name_item = self.table.item(row, 1)
        if not name_item:
            return
            
        data = name_item.data(Qt.ItemDataRole.UserRole)
        
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
            
            if metadata.get('bookmarked'):
                details += f"<b>Bookmarked:</b> ⭐ Yes<br>"
            
            if 'comment' in metadata and metadata['comment']:
                details += f"<b>Comment:</b> {metadata['comment']}<br>"
            
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
    
    def toggle_bookmark(self):
        """Toggle bookmark status for selected model"""
        if not self.selected_model or not self.selected_metadata:
            QMessageBox.warning(self, "No Selection", "Please select a model to bookmark.")
            return
        
        is_bookmarked = self.selected_metadata.get('bookmarked', False)
        new_status = not is_bookmarked
        
        if self.model_manager.update_metadata(self.selected_model, {'bookmarked': new_status}):
            status_text = "bookmarked" if new_status else "unbookmarked"
            QMessageBox.information(self, "Success", f"Model '{self.selected_model}' {status_text}.")
            self.load_models()
            # Re-select the same model
            for row in range(self.table.rowCount()):
                name_item = self.table.item(row, 1)
                if name_item and name_item.text() == self.selected_model:
                    self.table.selectRow(row)
                    break
        else:
            QMessageBox.warning(self, "Error", "Failed to update bookmark.")
    
    def edit_comment(self):
        """Add or edit comment for selected model"""
        if not self.selected_model or not self.selected_metadata:
            QMessageBox.warning(self, "No Selection", "Please select a model to add a comment.")
            return
        
        current_comment = self.selected_metadata.get('comment', '')
        
        comment, ok = QInputDialog.getMultiLineText(
            self,
            "Edit Comment",
            f"Enter comment for '{self.selected_model}':",
            current_comment
        )
        
        if ok:
            if self.model_manager.update_metadata(self.selected_model, {'comment': comment}):
                QMessageBox.information(self, "Success", "Comment updated successfully.")
                self.load_models()
                # Re-select the same model
                for row in range(self.table.rowCount()):
                    name_item = self.table.item(row, 1)
                    if name_item and name_item.text() == self.selected_model:
                        self.table.selectRow(row)
                        break
            else:
                QMessageBox.warning(self, "Error", "Failed to update comment.")
    
    def rename_model(self):
        """Rename the selected model"""
        if not self.selected_model:
            QMessageBox.warning(self, "No Selection", "Please select a model to rename.")
            return
        
        new_name, ok = QInputDialog.getText(
            self,
            "Rename Model",
            f"Enter new name for '{self.selected_model}':",
            QLineEdit.EchoMode.Normal,
            self.selected_model
        )
        
        if ok and new_name:
            if new_name == self.selected_model:
                return  # No change
            
            # Validate name
            if not new_name.replace('_', '').replace('-', '').isalnum():
                QMessageBox.warning(self, "Invalid Name", "Model name can only contain letters, numbers, underscores, and hyphens.")
                return
            
            if self.model_manager.rename_model(self.selected_model, new_name):
                QMessageBox.information(self, "Success", f"Model renamed to '{new_name}'.")
                self.selected_model = new_name
                self.load_models()
                # Select the renamed model
                for row in range(self.table.rowCount()):
                    name_item = self.table.item(row, 1)
                    if name_item and name_item.text() == new_name:
                        self.table.selectRow(row)
                        break
            else:
                QMessageBox.warning(self, "Error", "Failed to rename model. Name may already exist.")
    
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
