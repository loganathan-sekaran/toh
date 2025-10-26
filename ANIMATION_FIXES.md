# Animation System Improvements

## Problem
The visualization was too fast and showing duplicate discs because:
1. QTimer was being used from a background thread (training thread)
2. Qt components must only be accessed from the main thread
3. Animation synchronization was failing

## Solution
Completely redesigned the animation system to be thread-safe:

### Changes Made:

1. **Removed QTimer-based animation** - Eliminated the async timer approach that was causing thread conflicts

2. **Implemented frame-by-frame rendering** - The `animate_move()` function now directly renders each frame:
   ```python
   for frame in range(self.animation_frames):
       progress = frame / self.animation_frames
       self.canvas.animation_progress = progress
       self.canvas.update()
       QApplication.processEvents()
       time.sleep(self.animation_speed / 1000.0)
   ```

3. **Added Visualization Toggle Button** - NEW FEATURE!
   - **Green button**: "Hide Visualization (Fast Training)" - Currently showing animations
   - **Orange button**: "Show Visualization (Currently Fast)" - Currently training fast without animation
   - Click to toggle between modes at any time during training
   - When hidden: Training runs much faster (skips all animations)
   - When shown: See smooth lift/move/drop animations for each disc
   - Metrics always update regardless of visualization state

4. **Adjusted animation parameters**:
   - `animation_frames`: 30 frames per move (was 60)
   - `animation_speed`: 20ms per frame (was 30ms)
   - Total animation time per move: 600ms (30 frames × 20ms)
   - For 7-move optimal solution: ~4.2 seconds total

5. **Proper thread synchronization**:
   - `QApplication.processEvents()` ensures Qt events are processed
   - Each frame is explicitly rendered before moving to the next
   - No more blocking while loops or timer conflicts

6. **Show every episode**: Changed `show_every_n` from 10 to 1 in `train_with_gui.py` so you can see every training episode (when visualization is enabled)

7. **Initial state pause**: Added 0.5s delay after environment reset to clearly show starting position with all discs on rod A

## How to Use

### Run Demo (Optimal 7-move solution):
```bash
source venv/bin/activate
python main.py
# Select option 1 (Demo)
```

### Run Training with Visualization:
```bash
source venv/bin/activate
python main.py
# Select option 2 (Train)
```

### Toggle Visualization During Training:
- **To train FAST in background**: Click "Hide Visualization (Fast Training)" button
  - Training runs at maximum speed without animation delays
  - Metrics still update in the UI
  - Perfect for training many episodes quickly
  
- **To watch training**: Click "Show Visualization (Currently Fast)" button
  - Animations resume for each move
  - See the lift/move/drop sequence clearly
  - Perfect for monitoring agent behavior

### Adjust Animation Speed:
- Use the slider in the GUI: range 10-200ms per frame
- Lower values = faster animation
- Higher values = slower, more visible movement

## Animation Phases
Each move shows 3 phases:
1. **Lift** (0-33%): Disc rises from source rod
2. **Across** (33-67%): Disc moves horizontally 
3. **Drop** (67-100%): Disc descends onto target rod

## Current Settings
- **Frames per move**: 30
- **Speed per frame**: 20ms (adjustable via slider: 10-200ms)
- **Total time per move**: ~600ms at default speed
- **Visualization frequency**: Every episode when enabled (show_every_n=1)
- **Initial pause**: 0.5s to show starting state
- **Fast mode**: Skips all animations and most UI updates for maximum training speed

## Performance
- Thread-safe: Works correctly with PyQt6 event loop
- No crashes: Eliminated QBasicTimer threading errors
- Smooth animation: Proper frame-by-frame rendering
- Responsive GUI: Pause/stop/toggle buttons work during training
- **Fast mode**: Training speed increases dramatically when visualization is hidden

## UI Layout
```
┌─────────────────────────────────────┐
│  Tower of Hanoi - RL Training       │
│                                     │
│  [Canvas with animated discs]       │
│                                     │
│  Episode: 42   Step: 7              │
│  Reward: 93    Epsilon: 0.452       │
│  Success Rate: 65.2%                │
│                                     │
│  Animation Speed: [====] 20 ms/frame│
│                                     │
│  [Pause] [Stop] [Toggle Viz] [Test] │
└─────────────────────────────────────┘
```

## Next Steps
If animation is still too fast or too slow, adjust:
1. Use the speed slider in the GUI (10-200ms range)
2. Modify `animation_speed` in `visualizer.py` __init__ (line 197)
3. Modify `animation_frames` in `visualizer.py` (line 285)

Formula: `total_time = animation_frames × animation_speed`

## Training Strategy
1. **Start with visualization ON** to verify agent is learning correctly
2. **Toggle to fast mode** once you're confident training is working
3. **Toggle back to visualization** periodically to check agent progress
4. This gives you the best of both worlds: fast training + visual monitoring
