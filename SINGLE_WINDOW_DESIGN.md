# Single-Window GUI Design

## Overview

The Tower of Hanoi RL Trainer now uses a **single-window design** where all functionality happens in one unified window. No more switching between multiple windows or hiding/showing!

## Key Design Change

### Before (Multi-Window)
```
Menu Window → Hide → Training Window Opens → Close → Menu Window Shows
```
**Problems:**
- Window management complexity
- Event loop conflicts
- User loses context
- Confusing navigation

### Now (Single-Window)
```
Menu Page ↔️ Visualization Page (in same window)
```
**Benefits:**
- ✅ Single window stays open
- ✅ No event loop conflicts
- ✅ Clean back button navigation
- ✅ Embedded visualization
- ✅ Better user experience

## Architecture

### QStackedWidget Layout

The main window uses `QStackedWidget` to switch between two pages:

1. **Menu Page** (Index 0)
   - Main menu with all buttons
   - Always available via "Back to Menu" button

2. **Visualization Page** (Index 1)
   - Contains the Tower of Hanoi visualization
   - Back button at top
   - Visualization widget embedded below

```
┌─ MainLauncher Window ─────────────────────┐
│  QStackedWidget                            │
│  ┌──────────────────────────────────────┐ │
│  │ [Menu Page]                          │ │
│  │  - Title                             │ │
│  │  - Buttons (Demo, Train, etc.)      │ │
│  │  - Exit                              │ │
│  └──────────────────────────────────────┘ │
│  ┌──────────────────────────────────────┐ │
│  │ [Visualization Page]                 │ │
│  │  ⬅️ Back to Menu                    │ │
│  │  ┌────────────────────────────────┐ │ │
│  │  │  TowerOfHanoiVisualizer        │ │ │
│  │  │  - Canvas                       │ │ │
│  │  │  - Controls                     │ │ │
│  │  │  - Metrics                      │ │ │
│  │  └────────────────────────────────┘ │ │
│  └──────────────────────────────────────┘ │
└────────────────────────────────────────────┘
```

## User Flow

### Demo
1. Click "🎬 Demo" on menu page
2. Window switches to visualization page
3. Watch optimal solution animation
4. Click "⬅️ Back to Menu" anytime
5. Returns to menu page

### Train
1. Click "🏋️ Train Model"
2. Configure in dialog (episodes, discs, batch size)
3. Click OK
4. Window switches to visualization page with training
5. Watch training progress with animations
6. Click "⬅️ Back to Menu" when done
7. Returns to menu page

### Test
1. Click "🧪 Test Model"
2. Select model from dialog
3. Window switches to visualization page
4. Watch model solving puzzle
5. Click "⬅️ Back to Menu"
6. Returns to menu page

### Quick Train
1. Click "⚡ Quick Train"
2. Confirm in dialog
3. Window switches to visualization page
4. Training runs with visualization hidden (fast mode)
5. Progress shown in metrics
6. Click "⬅️ Back to Menu" when done

## Implementation Details

### Page Switching

```python
# Switch to visualization page
def show_visualization_page(self, visualizer_widget):
    # Clear old content
    # Add back button
    # Add visualizer widget
    self.stacked_widget.setCurrentWidget(self.viz_container)

# Switch to menu page
def show_menu_page(self):
    self.stacked_widget.setCurrentWidget(self.menu_page)
```

### Embedded Training

Training now happens **inside** the main window:

```python
def on_train(self):
    # Get configuration
    config = dialog.get_config()
    
    # Create visualizer
    visualizer = TowerOfHanoiVisualizer(env, num_discs=config['num_discs'])
    
    # Show it in same window
    self.show_visualization_page(visualizer)
    
    # Run training in background thread
    training_thread = Thread(target=training_loop, daemon=True)
    training_thread.start()
```

**Key Points:**
- Visualizer created as widget (not separate window)
- Embedded into main window's visualization page
- Training runs in background thread
- Main window event loop handles everything
- No `sys.exit(app.exec())` calls

### Back Button

Always visible at top of visualization page:

```python
back_btn = QPushButton("⬅️  Back to Menu")
back_btn.clicked.connect(self.show_menu_page)
```

**Safe to click anytime:**
- Training thread is daemon (auto-cleanup)
- Visualizer has `should_stop` flag
- Clean transition back to menu

## Benefits

### 1. No Event Loop Conflicts
**Before:**
```python
# Training window
sys.exit(app.exec())  # ERROR: Loop already running!
```

**Now:**
```python
# Everything in one event loop
QApplication.processEvents()  # Works perfectly
```

### 2. Better Context
- User always knows where they are
- One window title bar
- Consistent window size
- No "where did my window go?"

### 3. Simpler Code
- No window hiding/showing logic
- No return_on_complete flags needed
- Single event loop management
- Cleaner navigation flow

### 4. Professional UX
- Modern single-page app feel
- Smooth transitions
- Always accessible back button
- Consistent navigation pattern

## Window Sizing

```python
self.setMinimumSize(800, 700)
```

**Large enough for:**
- Menu buttons with descriptions
- Full visualization canvas
- Metrics display
- Control buttons
- Back button

## Styling

### Menu Page
- 40px margins for breathing room
- 20px spacing between sections
- Large buttons (50px height)
- Clear typography hierarchy

### Visualization Page
- No margins (full-width back button)
- Visualizer takes remaining space
- Clean separation with back button

### Back Button
- Gray color (#6c757d) - neutral, not primary action
- Full width for easy clicking
- Left-aligned arrow + text
- Hover effect for feedback

## Future Enhancements

Possible additions:

1. **Navigation Breadcrumbs**
   ```
   Menu > Train Model > Episode 245/1000
   ```

2. **Quick Actions on Viz Page**
   - Pause/Resume button
   - Save checkpoint button
   - Speed slider

3. **Split View**
   ```
   ┌─ Menu (left sidebar) ─┬─ Visualization ─┐
   │ - Demo                 │                  │
   │ - Train                │   [Viz Here]     │
   │ - Test                 │                  │
   └────────────────────────┴──────────────────┘
   ```

4. **Tabs Instead of Stack**
   ```
   [Menu] [Training] [Results]
   ```

5. **Floating Controls**
   - Back button floats over visualization
   - Doesn't take vertical space

## Testing

To test the single-window design:

```bash
./start_gui.sh
```

**Test each flow:**
1. Demo → Back → Train → Back → Test → Back
2. Verify window never closes/hides
3. Verify smooth transitions
4. Verify back button always works
5. Verify training can be interrupted

## Troubleshooting

### "Window disappears after training"
- **Should not happen** in single-window design
- If it does, check for stray `self.hide()` calls

### "Back button doesn't work"
- Check `show_menu_page()` is connected
- Verify stacked widget indices are correct

### "Training freezes window"
- Ensure `QApplication.processEvents()` is called
- Check training runs in daemon thread
- Verify no blocking operations in main thread

## Conclusion

The single-window design provides a much better user experience:
- ✅ **One window** - No confusion
- ✅ **Embedded viz** - Seamless integration  
- ✅ **Easy navigation** - Always one click back to menu
- ✅ **No conflicts** - Single event loop
- ✅ **Professional** - Modern UX pattern

This matches patterns from modern applications like VS Code, Spotify, and other professional desktop apps.
