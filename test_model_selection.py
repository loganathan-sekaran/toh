"""
Test Model Selection Dialog with new features
"""
import sys
from PyQt6.QtWidgets import QApplication
from model_selection_dialog import ModelSelectionDialog
from model_manager import ModelManager

def test_model_selection_features():
    """Test bookmark, comment, rename, and sorting features"""
    app = QApplication(sys.argv)
    
    # Create model manager and check if we have models
    manager = ModelManager()
    models = manager.list_models()
    
    print("=" * 80)
    print("MODEL SELECTION DIALOG - FEATURE TEST")
    print("=" * 80)
    print(f"Found {len(models)} models\n")
    
    if not models:
        print("No models found. Please train a model first.")
        return
    
    # Show existing model metadata
    print("Existing models:")
    for model_name, metadata in models[:3]:  # Show first 3
        print(f"\n  {model_name}:")
        print(f"    - Bookmarked: {metadata.get('bookmarked', False)}")
        print(f"    - Comment: {metadata.get('comment', 'None')}")
        print(f"    - Architecture: {metadata.get('architecture', 'Unknown')}")
    
    print("\n" + "=" * 80)
    print("OPENING MODEL SELECTION DIALOG")
    print("=" * 80)
    print("\nFeatures to test:")
    print("  ✓ Table columns are sortable (click headers)")
    print("  ✓ Columns are resizable (drag column edges)")
    print("  ✓ Bookmark button (⭐) - toggle bookmark status")
    print("  ✓ Comment button (💬) - add/edit comments")
    print("  ✓ Rename button (✏️) - rename models")
    print("  ✓ Bookmarked models show ⭐ with yellow background")
    print("\nTry these actions and close the dialog when done.")
    print("=" * 80 + "\n")
    
    # Open dialog
    dialog = ModelSelectionDialog()
    dialog.exec()
    
    # Show updated metadata after dialog closes
    print("\n" + "=" * 80)
    print("UPDATED MODEL METADATA")
    print("=" * 80)
    models = manager.list_models()
    for model_name, metadata in models[:3]:
        print(f"\n  {model_name}:")
        print(f"    - Bookmarked: {metadata.get('bookmarked', False)}")
        print(f"    - Comment: {metadata.get('comment', 'None')}")
    
    print("\n" + "=" * 80)
    print("✅ TEST COMPLETE")
    print("=" * 80)

if __name__ == "__main__":
    test_model_selection_features()
