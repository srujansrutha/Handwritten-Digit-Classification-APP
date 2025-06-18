
---
## 🧠 Deep Learning Basics with MNIST – Summary Outline

### 1. **What Is Deep Learning?**
- Works best for large, complex, or unstructured data (e.g., images, text)
- Inspired by the brain — built from layers of *artificial neurons*
- Used when classical ML reaches its limits

---

### 2. **Neural Network Structure**
- **Input Layer** → Receives data (e.g. image pixels)
- **Hidden Layers** → Extract patterns with weights, activations
- **Output Layer** → Final prediction (e.g., class scores)

---

### 3. **Training Process Overview**
- Forward pass → Model predicts
- Loss computed → Measures error
- Backward pass → Gradients calculated
- Optimizer step → Weights updated

**Formula:**  
`W_new = W_old - η * ∂E/∂W`

---

### 4. **Training Variants**
- **SGD**: One sample at a time
- **Batch GD**: All data in one go
- **Mini-Batch GD**: Small groups (most practical)

---

### 5. **Optimizers**
- **Vanilla GD**: Basic updates
- **Momentum**: Adds velocity
- **RMSProp**: Adapts per parameter
- **Adam**: Most popular; combines Momentum + RMSProp

---

### 6. **Loss Functions**
- **MSE / MAE** → Regression
- **Binary CrossEntropy** → Binary classification
- **Categorical CrossEntropy** → Multi-class classification (used for MNIST)

---

### 7. **Normalization**
- Input pixel values scaled:
  - `ToTensor()` → [0, 1]
  - `Normalize((0.5,), (0.5,))` → [-1, 1]
- Keeps data centered and stable during training

---

### 8. **PyTorch Workflow Recap**
- `Dataset` + `DataLoader` → Handle batching/shuffling
- `nn.Module` → Define model architecture
- `CrossEntropyLoss()` → Calculates loss
- `Adam(model.parameters())` → Updates weights
- `model.eval() + torch.no_grad()` → Turns off training features for testing

---

### 9. **Evaluation Metrics**
- **Accuracy** → % of correct predictions
- **Confusion Matrix** → Breakdown of true vs predicted per class
- **Classification Report**:
  - **Precision**: Of predicted class X, how many were correct
  - **Recall**: Of actual class X, how many were found
  - **F1-score**: Balance of precision & recall

---

### 10. **Highlights You’ve Explored**
- `Flatten()` reshapes input to 1D for linear layers
- MNIST is grayscale → 1 channel (`(1, 28, 28)`)
- Output of model = logits → Softmax applied inside loss
- Evaluation collects predictions → compares with true labels

---

