# Architecture Preview Feature

## ✨ New Feature: Visual Architecture Preview

You can now **preview the neural network architecture** before training or when selecting saved models!

## 📍 Where to Find It

### 1. Training Dialog - Preview Before Training

When configuring a new training session:

```
./start_gui.sh → Click "🏋️ Train Model"
```

You'll see:
- **Model Architecture** dropdown with all available architectures
- **👁️ Preview** button next to the dropdown
- Click **Preview** to see the visual diagram!

**What it shows:**
- Complete neural network diagram (layers, nodes, connections)
- Architecture name and description
- Total parameter count
- Layer-by-layer details (units, activation functions)
- Input/output sizes based on your disc configuration

### 2. Model Selection Dialog - Preview Trained Models

When selecting a model for testing:

```
./start_gui.sh → Click "🧪 Test Model"
```

You'll see a table with all trained models, including:
- Model name
- **Architecture** column showing which architecture was used
- Training metrics (episodes, success rate, avg steps)
- **👁️ Preview Architecture** button

Click **Preview Architecture** to see:
- The neural network structure of the trained model
- Complete training history and performance metrics
- Layer-by-layer breakdown
- Visual diagram showing the exact architecture

## 🎯 Use Cases

### Before Training

**Compare architectures visually:**

1. Select "Small (24-24)" → Click Preview
   - See: 2 hidden layers, 990 parameters
   
2. Select "Large (128-64-32)" → Click Preview
   - See: 3 hidden layers, 11,814 parameters
   
3. Select "Extra Large (256-128-64)" → Click Preview
   - See: 3 large layers + dropout, 44,102 parameters

**Decision making:**
- "Do I need the complexity of Extra Large?"
- "Is Small sufficient for my 3-disc problem?"
- Visual comparison helps you choose!

### After Training

**Analyze trained models:**

1. Open model selection dialog
2. See which architecture each model used
3. Click preview to visualize the exact structure
4. Compare different architectures' performance

**Example workflow:**
```
Model A: Small (24-24) - 85% success, 12 avg steps
Model B: Large (128-64-32) - 95% success, 8 avg steps
Model C: Extra Large - 96% success, 7.5 avg steps

→ Preview each to understand why performance differs
→ See layer sizes and connections visually
→ Make informed decisions about future training
```

## 📊 Preview Window Features

When you click preview, you get:

### Header Section
- **Architecture name**: e.g., "Large (128-64-32)"
- **Description**: What this architecture is designed for
- **Complexity rating**: Low/Medium/High/Very High
- **Total parameters**: Exact count (e.g., "11,814 parameters")
- **Recommended episodes**: Suggested training duration

### Visual Diagram
- **Color-coded layers**:
  - 🟢 Green: Input layer
  - 🔵 Blue: Hidden layers
  - 🟠 Orange: Output layer
- **Nodes**: Circles representing neurons
- **Connections**: Lines showing layer connectivity
- **Labels**: Layer names and neuron counts

### Layer Details Section
Detailed breakdown:
```
• Layer 1: Dense - 128 units (relu)
• Layer 2: Dense - 64 units (relu)
• Layer 3: Dense - 32 units (relu)
• Layer 4: Dense - 6 units (linear)
```

For models with dropout:
```
• Layer 1: Dense - 256 units (relu)
• Layer 2: Dropout - rate 0.2
• Layer 3: Dense - 128 units (relu)
...
```

## 🚀 Quick Demo

Try this to see all architectures:

1. Start the GUI: `./start_gui.sh`
2. Click "🏋️ Train Model"
3. Go through each architecture in the dropdown:
   - Select "Small (24-24)" → Preview
   - Select "Medium (64-32)" → Preview
   - Select "Large (128-64-32)" → Preview
   - Select "Extra Large (256-128-64)" → Preview
4. Compare them side-by-side!

## 💡 Pro Tips

### Choosing the Right Architecture

Use preview to help decide:

**For Quick Experiments:**
- Preview Small → See simple 2-layer network
- Fast training, good for testing ideas

**For Production:**
- Preview Large → See 3-layer funnel
- Best balance of performance and speed

**For Research:**
- Preview Extra Large → See deep network with dropout
- Maximum capacity, worth the training time

### Understanding Your Model

After training:
1. Open test dialog
2. Look at success rates in table
3. Preview high-performing models
4. Study their architectures
5. Understand what made them successful

### Debugging

If model isn't learning:
- Preview the architecture
- Check if it's too small (Small architecture for 5 discs?)
- Or too large (Extra Large for 3 discs?)
- Visual feedback helps diagnose issues

## 🎨 Visual Learning

The preview helps you:
- **Understand** what "128-64-32" actually means visually
- **Compare** different architectures at a glance
- **Learn** neural network architecture concepts
- **Debug** by seeing the actual structure
- **Make informed choices** based on visual comparison

## Example Comparison

### Small (24-24)
```
Preview shows:
Input(9) → 24 nodes → 24 nodes → Output(6)
Total: 990 parameters
Simple, fast, minimal
```

### Large (128-64-32)
```
Preview shows:
Input(9) → 128 nodes → 64 nodes → 32 nodes → Output(6)
Total: 11,814 parameters
Complex, powerful, funnel design
```

### Extra Large (256-128-64)
```
Preview shows:
Input(9) → 256 nodes → Dropout → 128 nodes → Dropout → 64 nodes → Output(6)
Total: 44,102 parameters
Very complex, with regularization
```

## Summary

✅ **Preview before training** - See what you're about to train  
✅ **Preview trained models** - Understand what worked  
✅ **Visual comparison** - Make informed architecture choices  
✅ **Educational** - Learn neural network structures  
✅ **No guessing** - See exact layer configurations  

**Start using it now:**
```bash
./start_gui.sh
# → Train Model → Click 👁️ Preview
```

Enjoy exploring your neural networks visually! 🎨🧠
