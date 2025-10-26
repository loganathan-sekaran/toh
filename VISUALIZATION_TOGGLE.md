# Visualization Toggle Feature

## Quick Start

During training, you'll see a button that lets you toggle between slow animated training and fast background training:

### When Visualization is ON (Default):
```
┌──────────────────────────────────────────────────┐
│ [Hide Visualization (Fast Training)]             │  ← GREEN BUTTON
└──────────────────────────────────────────────────┘
```
- Button is **GREEN**
- Shows animated disc movements (lift → across → drop)
- Training is slower but you can see what's happening
- Good for: Monitoring agent behavior, debugging, demonstrations

### When Visualization is OFF (Fast Mode):
```
┌──────────────────────────────────────────────────┐
│ [Show Visualization (Currently Fast)]            │  ← ORANGE BUTTON
└──────────────────────────────────────────────────┘
```
- Button is **ORANGE**
- Skips all animations
- Training runs at maximum speed
- Metrics still update in the background
- Good for: Training many episodes quickly, reaching target performance faster

## Use Cases

### Example Training Session:

1. **Start training** (Visualization ON by default)
   - Watch first 10-20 episodes to verify agent is learning
   - See if it's making good moves or struggling
   
2. **Switch to Fast Mode** (Click button → turns ORANGE)
   - Let it train 500-1000 episodes quickly
   - Monitor success rate and epsilon in the metrics panel
   
3. **Check Progress** (Click button → turns GREEN again)
   - See how well the agent performs now
   - Watch it solve Tower of Hanoi efficiently
   
4. **Continue Fast Training** if needed
   - Switch back to fast mode for more episodes

## Keyboard Shortcut Idea (Future Enhancement)
Could add:
- **Spacebar**: Toggle visualization
- **P**: Pause/Resume
- **S**: Stop training

## Performance Comparison

### With Visualization ON:
- ~600ms per move (default speed)
- ~7-10 episodes per minute

### With Visualization OFF:
- ~1-2ms per move (minimal overhead)
- ~500-1000 episodes per minute
- **50-100x faster training!**

## Tips

1. **Always start with visualization ON** to make sure training is working correctly
2. **Use fast mode for bulk training** when you want to reach 1000+ episodes
3. **Toggle back periodically** to check on the agent's progress
4. **Success rate metric** is your friend - watch it climb even in fast mode!
5. **The button works instantly** - you can toggle mid-episode

## Technical Details

When visualization is OFF:
- `animate_move()` returns immediately without rendering frames
- `QApplication.processEvents()` called only every 10 episodes (instead of every frame)
- State updates are skipped
- Only metrics labels are updated
- Result: Training loop runs at near-native Python speed

When visualization is ON:
- Full 30-frame animation for each move
- State canvas updated after each move
- All UI elements refresh in real-time
- Smooth, observable training process
