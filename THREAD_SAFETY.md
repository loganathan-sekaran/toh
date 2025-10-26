# Thread Safety in Qt GUI

## Problem: GUI Operations from Background Thread

### The Error
```
NSWindow should only be instantiated on the main thread!
QObject::setParent: Cannot set parent, new parent is in a different thread
```

**Cause:** Called `QMessageBox.information()` from a Python threading.Thread (background thread).

**Qt Rule:** All GUI operations MUST happen on the main thread.

## Solution: QThread with Signals

### Architecture

```
Main Thread (GUI)          Worker Thread (Training)
─────────────────          ────────────────────────
MainLauncher               TrainingWorker
  │                          │
  ├─ Creates Worker          ├─ run() method
  ├─ Creates QThread         │   └─ Training loop
  ├─ moveToThread()          │
  │                          │
  ├─ Connects signals  <─────┼─ emit progress(...)
  │   progress.connect()     │   emit finished(...)
  │   finished.connect()     │
  │                          │
  ├─ Starts thread           │
  │                          │
  ├─ Receives signals ───────┘
  │   on_training_progress()
  │   on_training_finished()
  │     └─ QMessageBox (SAFE!)
  │
```

### Implementation

**1. Create Worker Class**

```python
class TrainingWorker(QObject):
    # Define signals
    finished = pyqtSignal(str)  # Completion message
    progress = pyqtSignal(int, float, float)  # episode, epsilon, success
    
    def __init__(self, env, agent, visualizer, config):
        super().__init__()
        self.env = env
        self.agent = agent
        self.visualizer = visualizer
        self.config = config
        self.should_stop = False
    
    def run(self):
        """Training loop - runs in worker thread."""
        for episode in range(1, self.config['episodes'] + 1):
            if self.should_stop:
                self.finished.emit("Training stopped by user")
                return
            
            # ... training code ...
            
            # Emit signal (thread-safe!)
            self.progress.emit(episode, self.agent.epsilon, 0.0)
        
        self.finished.emit("Training completed!")
    
    def stop(self):
        """Request stop from main thread."""
        self.should_stop = True
```

**2. Use in MainLauncher**

```python
def on_train(self):
    # Create worker and thread
    self.training_worker = TrainingWorker(env, agent, visualizer, config)
    self.training_thread = QThread()
    
    # Move worker to thread
    self.training_worker.moveToThread(self.training_thread)
    
    # Connect signals (these run on main thread!)
    self.training_thread.started.connect(self.training_worker.run)
    self.training_worker.progress.connect(self.on_training_progress)
    self.training_worker.finished.connect(self.on_training_finished)
    self.training_worker.finished.connect(self.training_thread.quit)
    
    # Start thread
    self.training_thread.start()

def on_training_progress(self, episode, epsilon, success_rate):
    """Handle progress - runs on MAIN thread."""
    self.current_visualizer.update_info(episode, epsilon, success_rate)

def on_training_finished(self, message):
    """Handle completion - runs on MAIN thread."""
    QMessageBox.information(self, "Training Complete", message)
```

## Key Concepts

### Signals and Slots

**Thread-Safe Communication:**
- Worker thread emits signals
- Main thread receives via slots
- Qt handles thread synchronization

```python
# Worker thread
self.progress.emit(episode, epsilon, 0.0)  # Emits signal

# Main thread (automatic via Qt)
def on_training_progress(self, episode, epsilon, success_rate):
    # This runs on main thread - safe for GUI operations!
    self.visualizer.update_info(episode, epsilon, success_rate)
```

### QThread vs threading.Thread

**Before (WRONG):**
```python
from threading import Thread

def training_loop():
    # ... training ...
    QMessageBox.information(self, "Done", "Complete")  # ERROR!

thread = Thread(target=training_loop, daemon=True)
thread.start()
```

**After (CORRECT):**
```python
from PyQt6.QtCore import QThread, pyqtSignal, QObject

class Worker(QObject):
    finished = pyqtSignal(str)
    
    def run(self):
        # ... training ...
        self.finished.emit("Complete")  # Signal, not direct GUI call

worker = Worker()
thread = QThread()
worker.moveToThread(thread)
thread.started.connect(worker.run)
worker.finished.connect(self.show_message)  # Slot runs on main thread
thread.start()
```

### Stopping Training Safely

**Stop from Main Thread:**
```python
def show_menu_page(self):
    # Stop worker
    if self.training_worker:
        self.training_worker.stop()  # Sets should_stop flag
    
    # Stop thread
    if self.training_thread and self.training_thread.isRunning():
        self.training_thread.quit()  # Request stop
        self.training_thread.wait(1000)  # Wait up to 1 second
    
    # Switch page
    self.stacked_widget.setCurrentWidget(self.menu_page)
```

**Check in Worker:**
```python
def run(self):
    for episode in range(1, max_episodes + 1):
        if self.should_stop:  # Check flag
            self.finished.emit("Stopped by user")
            return
        
        # ... training step ...
```

## Benefits

### Thread Safety
✅ No GUI operations from background thread  
✅ Qt handles synchronization automatically  
✅ No race conditions  
✅ Clean separation of concerns  

### Clean Shutdown
✅ Can stop training cleanly  
✅ Thread cleanup handled by Qt  
✅ No orphaned threads  
✅ Safe to restart training  

### Better UX
✅ Responsive UI during training  
✅ Progress updates work smoothly  
✅ Can navigate away safely  
✅ No crashes or freezes  

## Common Patterns

### Progress Updates

```python
# Worker
for i in range(100):
    # ... work ...
    self.progress.emit(i, status_text)

# Main
def on_progress(self, value, text):
    self.progress_bar.setValue(value)
    self.status_label.setText(text)
```

### Error Handling

```python
# Worker
class Worker(QObject):
    error = pyqtSignal(str)
    
    def run(self):
        try:
            # ... work ...
        except Exception as e:
            self.error.emit(str(e))

# Main
worker.error.connect(self.show_error)

def show_error(self, message):
    QMessageBox.critical(self, "Error", message)
```

### Long-Running Operations

```python
# Worker with periodic updates
def run(self):
    for batch in data_batches:
        if self.should_stop:
            return
        
        result = process_batch(batch)
        self.batch_complete.emit(result)
        
        # Allow thread to be interrupted
        QThread.msleep(10)
```

## Debugging Tips

### Check Thread Identity

```python
from PyQt6.QtCore import QThread

def some_method(self):
    current = QThread.currentThread()
    main = QApplication.instance().thread()
    
    if current == main:
        print("Running on MAIN thread ✓")
    else:
        print("Running on WORKER thread")
        print("WARNING: Don't call GUI methods here!")
```

### Signal Connection

```python
# Direct connection (same thread)
signal.connect(slot)

# Queued connection (cross-thread, thread-safe)
signal.connect(slot, Qt.ConnectionType.QueuedConnection)

# Auto (Qt decides based on threads)
signal.connect(slot, Qt.ConnectionType.AutoConnection)  # Default
```

### Common Mistakes

**❌ WRONG:**
```python
# Creating GUI objects in worker thread
def run(self):
    dialog = QDialog()  # ERROR: Not on main thread!
    dialog.exec()
```

**✅ CORRECT:**
```python
# Emit signal instead
def run(self):
    self.show_dialog_requested.emit()

# In main thread
def on_show_dialog_requested(self):
    dialog = QDialog(self)  # Safe: on main thread
    dialog.exec()
```

## Summary

**Golden Rule:** All GUI operations on main thread only!

**Use QThread + Signals for:**
- Long-running computations
- Training loops
- File I/O
- Network requests
- Any blocking operation

**Pattern:**
1. Create Worker(QObject) with signals
2. Create QThread
3. Move worker to thread
4. Connect signals to main thread slots
5. Start thread
6. Emit signals from worker
7. Handle in main thread

This ensures thread-safe, responsive, crash-free Qt applications!
