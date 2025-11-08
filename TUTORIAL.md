# Complete Tutorial: Building an Emotion Classifier with ModernBERT

## Table of Contents

1. [Introduction](#introduction)
2. [Prerequisites](#prerequisites)
3. [Understanding the Problem](#understanding-the-problem)
4. [Dataset Overview](#dataset-overview)
5. [Step-by-Step Implementation](#step-by-step-implementation)
6. [Understanding the Results](#understanding-the-results)
7. [Deployment Guide](#deployment-guide)
8. [Troubleshooting](#troubleshooting)
9. [FAQ](#faq)

---

## Introduction

This tutorial will guide you through building a **state-of-the-art emotion classifier** using ModernBERT, a cutting-edge transformer model. By the end of this tutorial, you'll have a working model that can classify text into 6 different emotions with over 94% accuracy.

### What You'll Build

An AI model that can analyze text and determine if it expresses:
- 😢 Sadness
- 😊 Joy
- ❤️ Love
- 😠 Anger
- 😨 Fear
- 😲 Surprise

### Why This Matters

Emotion classification has real-world applications in:
- **Mental Health**: Detecting emotional distress in text
- **Customer Service**: Understanding customer sentiment
- **Social Media**: Analyzing public opinion
- **Content Moderation**: Identifying emotionally charged content

---

## Prerequisites

### Required Knowledge

- **Basic Python**: Understanding of functions, loops, and data structures
- **Machine Learning Basics**: Familiarity with training/validation concepts
- **Optional**: Understanding of neural networks (helpful but not required)

### Required Software

```bash
# Python 3.8 or higher
python --version

# GPU recommended but not required
# Training time: ~60 minutes on GPU, ~4 hours on CPU
```

### Required Libraries

```bash
pip install torch>=2.0.0
pip install transformers>=4.30.0
pip install datasets>=2.12.0
pip install matplotlib>=3.5.0
pip install tqdm>=4.65.0
pip install numpy>=1.23.0
```

---

## Understanding the Problem

### What is Emotion Classification?

Emotion classification is a **supervised learning** problem where:
- **Input**: A piece of text (e.g., "I'm so happy!")
- **Output**: An emotion label (e.g., "joy")

### Why Use ModernBERT?

**ModernBERT** is a transformer-based model that:
1. **Pre-trained on massive text data**: Already understands language
2. **Context-aware**: Considers word relationships and context
3. **Transfer learning**: Can be fine-tuned for specific tasks
4. **State-of-the-art**: Achieves excellent results on NLP tasks

### Traditional ML vs. Deep Learning

| Aspect | Traditional ML | ModernBERT |
|--------|---------------|-----------|
| Features | Manual feature engineering | Automatic feature learning |
| Context | Limited context understanding | Full contextual understanding |
| Training Data | Can work with less data | Better with more data |
| Accuracy | 70-80% typical | 90%+ achievable |
| Setup | Complex preprocessing | Simpler with tokenizers |

---

## Dataset Overview

### DAIR-AI Emotion Dataset

We use the [DAIR-AI emotion dataset](https://huggingface.co/datasets/dair-ai/emotion) which contains:

| Split | Examples | Purpose |
|-------|----------|---------|
| Train | 16,000 | Model training |
| Validation | 2,000 | Hyperparameter tuning |
| Test | 2,000 | Final evaluation |

### Data Format

Each example contains:
```python
{
    "text": "I'm so happy about my promotion!",
    "label": 1  # 1 = joy
}
```

### Label Distribution

Understanding class balance:
```python
# Approximate distribution
Sadness:  ~4,600 examples (29%)
Joy:      ~5,400 examples (34%)
Love:     ~1,300 examples (8%)
Anger:    ~2,200 examples (14%)
Fear:     ~1,900 examples (12%)
Surprise: ~600 examples (4%)
```

**Note**: Some imbalance exists, but our model handles this well.

---

## Step-by-Step Implementation

### Step 1: Environment Setup

```python
# Install required packages
!pip install -q datasets transformers

# Import libraries
import torch
from transformers import AutoTokenizer, ModernBertForSequenceClassification
from datasets import load_dataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
```

**Explanation**: We install HuggingFace libraries for loading models and datasets.

---

### Step 2: Load Tokenizer and Dataset

```python
# Load ModernBERT tokenizer
model_id = "answerdotai/ModernBERT-base"
tokenizer = AutoTokenizer.from_pretrained(model_id)

# Load emotion dataset
ds = load_dataset("dair-ai/emotion")
train_ds = ds["train"]
val_ds = ds["validation"]
test_ds = ds["test"]

print(f"Training examples: {len(train_ds)}")
print(f"Validation examples: {len(val_ds)}")
```

**What's happening**:
- `AutoTokenizer`: Converts text to numerical tokens
- `load_dataset`: Downloads and caches the dataset

**Sample output**:
```
Training examples: 16000
Validation examples: 2000
```

---

### Step 3: Tokenization

**Why tokenize?**
- Neural networks only understand numbers, not text
- Tokenization converts "Hello world" → [2045, 2088]

```python
def tokenization(item):
    """
    Convert text to token IDs

    Parameters:
    - padding='max_length': Ensure all sequences are same length
    - truncation=True: Cut off text longer than max_length
    - max_length=300: Maximum tokens (chosen based on data analysis)
    """
    return tokenizer(
        item['text'],
        padding="max_length",
        truncation=True,
        max_length=300
    )

# Apply tokenization to datasets
train_ds = train_ds.map(tokenization, batched=True)
val_ds = val_ds.map(tokenization, batched=True)
```

**Understanding the output**:
```python
# Example tokenized output
{
    'input_ids': [101, 1045, 1005, 1049, 2061, 3407, 999, 102, 0, 0, ...],  # Token IDs
    'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, ...]  # 1=real token, 0=padding
}
```

**Key parameters**:
- `batched=True`: Process multiple examples at once (faster)
- `max_length=300`: Based on analysis showing longest text is ~300 chars

---

### Step 4: Prepare Data for Training

```python
# Convert to PyTorch format
train_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
val_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

# Create DataLoader for batching
train_dataloader = DataLoader(train_ds, batch_size=32, shuffle=True)

# Define label mapping
label_mapping = {
    0: "sadness", 1: "joy", 2: "love",
    3: "anger", 4: "fear", 5: "surprise"
}
```

**Why DataLoader?**
- Automatically batches data
- Shuffles for better training
- Handles parallel loading

---

### Step 5: Load and Configure Model

```python
# Set device (GPU if available)
cuda_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {cuda_device}")

# Load pre-trained model with classification head
model = ModernBertForSequenceClassification.from_pretrained(
    "answerdotai/ModernBERT-base",
    num_labels=6  # 6 emotion categories
).to(cuda_device)
```

**Model Architecture**:
```
ModernBERT-base:
├── Embedding Layer (vocabulary → vectors)
├── 12 Transformer Layers (context understanding)
├── Pooling Layer (sequence → single vector)
└── Classification Head (vector → 6 probabilities)
```

**Parameters**:
- Total parameters: ~110 million
- Trainable parameters: ~110 million
- Model size: ~440 MB

---

### Step 6: Configure Training

```python
import torch.optim as optim
from torch import nn

# Hyperparameters
learning_rate = 1e-4      # How fast the model learns
weight_decay = 1e-2       # Regularization strength
num_epochs = 5            # Training iterations

# Optimizer (updates model weights)
optimizer = optim.AdamW(
    model.parameters(),
    lr=learning_rate,
    weight_decay=weight_decay
)

# Loss function (measures prediction error)
criterion = nn.CrossEntropyLoss()
```

**Understanding Hyperparameters**:

| Parameter | Value | Why This Value? |
|-----------|-------|----------------|
| Learning Rate | 1e-4 | Small enough for stable training, large enough to converge |
| Weight Decay | 1e-2 | Prevents overfitting by penalizing large weights |
| Batch Size | 32 | Balance between memory usage and training stability |
| Epochs | 5 | Sufficient for convergence without overfitting |

---

### Step 7: Training Loop

```python
from tqdm import tqdm

# Set model to training mode
model.train()

# Training loop
for epoch in range(num_epochs):
    total_loss = 0

    # Progress bar for batches
    for batch in tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}"):
        # Move data to GPU
        input_ids = batch['input_ids'].to(cuda_device)
        attention_mask = batch['attention_mask'].to(cuda_device)
        labels = batch['label'].to(cuda_device)

        # Reset gradients
        optimizer.zero_grad()

        # Forward pass: compute predictions
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )

        # Get loss
        loss = outputs.loss
        total_loss += loss.item()

        # Backward pass: compute gradients
        loss.backward()

        # Update weights
        optimizer.step()

    # Print epoch statistics
    avg_loss = total_loss / len(train_dataloader)
    print(f"Epoch {epoch+1}, Average Loss: {avg_loss:.4f}")
```

**What's happening in each step**:

1. **Forward Pass**: Model makes predictions
2. **Loss Calculation**: Compare predictions to true labels
3. **Backward Pass**: Calculate gradients (how to improve)
4. **Weight Update**: Adjust model parameters

**Expected Output**:
```
Epoch 1/5: 100%|██████| 500/500 [18:00<00:00]
Epoch 1, Average Loss: 0.4261

Epoch 2/5: 100%|██████| 500/500 [18:00<00:00]
Epoch 2, Average Loss: 0.1211

Epoch 3/5: 100%|██████| 500/500 [18:00<00:00]
Epoch 3, Average Loss: 0.1039
```

**Interpreting Loss**:
- **Decreasing loss** = model is learning ✅
- **Increasing loss** = something is wrong ❌
- **Plateauing loss** = model has converged 📊

---

### Step 8: Evaluation - Confusion Matrix

```python
import matplotlib.pyplot as plt
import numpy as np

# Create validation DataLoader
val_dataloader = DataLoader(val_ds, batch_size=32)

# Initialize confusion matrix
confusion = torch.zeros(6, 6)

# Evaluation mode (disables dropout)
model.eval()

# Evaluate on validation set
with torch.no_grad():
    for batch in val_dataloader:
        input_ids = batch['input_ids'].to(cuda_device)
        attention_mask = batch['attention_mask'].to(cuda_device)
        labels = batch['label'].to(cuda_device)

        # Get predictions
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        preds = torch.argmax(outputs.logits, dim=1)

        # Update confusion matrix
        for t, p in zip(labels, preds):
            confusion[t, p] += 1

# Normalize by row (convert to percentages)
for i in range(6):
    confusion[i] = confusion[i] / confusion[i].sum()

# Visualize
plt.figure(figsize=(10, 8))
plt.imshow(confusion.cpu().numpy(), cmap='Blues')
plt.colorbar()
plt.xticks(range(6), label_mapping.values(), rotation=90)
plt.yticks(range(6), label_mapping.values())
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()
```

**Reading the Confusion Matrix**:
- **Diagonal (top-left to bottom-right)**: Correct predictions
- **Bright diagonal** = good model
- **Off-diagonal values** = misclassifications

Example interpretation:
```
         sadness  joy  love  anger  fear  surprise
sadness    0.92   0.01  0.00   0.02  0.05    0.00
joy        0.01   0.96  0.02   0.00  0.00    0.01
...
```
- Row 1: 92% of sadness texts were correctly identified
- 5% of sadness texts were misclassified as fear (makes sense!)

---

### Step 9: Calculate Accuracy

```python
correct = 0
total = 0

model.eval()
with torch.no_grad():
    for batch in val_dataloader:
        input_ids = batch['input_ids'].to(cuda_device)
        attention_mask = batch['attention_mask'].to(cuda_device)
        labels = batch['label'].to(cuda_device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        preds = torch.argmax(outputs.logits, dim=1)

        correct += (preds == labels).sum().item()
        total += labels.size(0)

accuracy = correct / total
print(f"Validation Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
```

**Expected Output**:
```
Validation Accuracy: 0.9435 (94.35%)
```

**Is 94% good?**
- **Random guessing**: ~16.7% (1 in 6)
- **Simple ML**: ~70-80%
- **Our model**: 94.35% ✅ Excellent!

---

### Step 10: Save the Model

```python
# Save model weights
torch.save(model.state_dict(), 'emotion_classifier_model.pth')
print("Model saved successfully!")

# To load later:
# model.load_state_dict(torch.load('emotion_classifier_model.pth'))
```

**What gets saved?**
- All model weights and biases
- File size: ~440 MB

**What's NOT saved?**
- Tokenizer (save separately if needed)
- Training history
- Optimizer state

---

### Step 11: Inference (Using the Model)

```python
def predict_emotion(text, model, tokenizer, label_mapping):
    """Predict emotion for a given text"""
    # Tokenize
    inputs = tokenizer(
        text,
        padding="max_length",
        truncation=True,
        max_length=300,
        return_tensors="pt"
    )

    # Move to device
    inputs = {k: v.to(cuda_device) for k, v in inputs.items()}

    # Predict
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)[0]
        pred_class = torch.argmax(probs).item()
        confidence = probs[pred_class].item()

    return label_mapping[pred_class], confidence

# Test it out!
test_texts = [
    "I'm so excited about my vacation!",
    "This is the worst day of my life.",
    "I absolutely love this movie!",
]

for text in test_texts:
    emotion, conf = predict_emotion(text, model, tokenizer, label_mapping)
    print(f"'{text}'")
    print(f"  → Emotion: {emotion} (Confidence: {conf*100:.1f}%)\n")
```

**Expected Output**:
```
'I'm so excited about my vacation!'
  → Emotion: joy (Confidence: 97.3%)

'This is the worst day of my life.'
  → Emotion: sadness (Confidence: 94.8%)

'I absolutely love this movie!'
  → Emotion: love (Confidence: 96.2%)
```

---

## Understanding the Results

### Training Dynamics

**Loss Curve Analysis**:
```
Epoch 1: 0.4261  ← Model is learning basic patterns
Epoch 2: 0.1211  ← Significant improvement (71% reduction)
Epoch 3: 0.1039  ← Convergence (14% reduction)
```

**What this tells us**:
- ✅ Model is learning effectively
- ✅ No overfitting (loss keeps decreasing)
- ✅ Could potentially train a bit longer

### Error Analysis

**Common mistakes the model makes**:
1. **Sadness ↔ Fear**: Both are negative emotions
2. **Joy ↔ Love**: Both are positive emotions
3. **Anger ↔ Sadness**: Can overlap in expression

**Why these confusions occur**:
- Linguistic similarity in expression
- Overlapping vocabulary
- Subjective nature of emotions

### Model Strengths

✅ **What the model does well**:
- Clear, unambiguous emotions (joy, anger)
- Short, direct expressions
- Common emotional phrases

❌ **What challenges the model**:
- Sarcasm and irony
- Mixed emotions
- Context-dependent emotions
- Cultural/dialectal variations

---

## Deployment Guide

### Option 1: Python Script

Create `inference.py`:
```python
import torch
from transformers import AutoTokenizer, ModernBertForSequenceClassification

class EmotionClassifier:
    def __init__(self, model_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base")
        self.model = ModernBertForSequenceClassification.from_pretrained(
            "answerdotai/ModernBERT-base",
            num_labels=6
        )
        self.model.load_state_dict(torch.load(model_path))
        self.model.to(self.device)
        self.model.eval()

        self.labels = ["sadness", "joy", "love", "anger", "fear", "surprise"]

    def predict(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=300)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.softmax(outputs.logits, dim=1)[0]
            pred = torch.argmax(probs).item()

        return self.labels[pred], probs[pred].item()

# Usage
classifier = EmotionClassifier('emotion_classifier_model.pth')
emotion, confidence = classifier.predict("I'm so happy!")
print(f"{emotion}: {confidence:.2%}")
```

### Option 2: REST API with Flask

Create `app.py`:
```python
from flask import Flask, request, jsonify
from inference import EmotionClassifier

app = Flask(__name__)
classifier = EmotionClassifier('emotion_classifier_model.pth')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    text = data.get('text', '')

    if not text:
        return jsonify({'error': 'No text provided'}), 400

    emotion, confidence = classifier.predict(text)
    return jsonify({
        'text': text,
        'emotion': emotion,
        'confidence': float(confidence)
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
```

Run it:
```bash
python app.py

# Test it:
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "I am so happy today!"}'
```

### Option 3: Gradio Web Interface

```python
import gradio as gr
from inference import EmotionClassifier

classifier = EmotionClassifier('emotion_classifier_model.pth')

def predict_emotion(text):
    emotion, confidence = classifier.predict(text)
    return f"{emotion} ({confidence:.1%} confidence)"

iface = gr.Interface(
    fn=predict_emotion,
    inputs=gr.Textbox(lines=3, placeholder="Enter text here..."),
    outputs="text",
    title="Emotion Classifier",
    description="Enter text to predict its emotion"
)

iface.launch()
```

---

## Troubleshooting

### Common Issues

**1. CUDA Out of Memory**
```python
# Solution: Reduce batch size
train_dataloader = DataLoader(train_ds, batch_size=16)  # Was 32
```

**2. Model Not Learning (Loss Not Decreasing)**
```python
# Check learning rate
optimizer = optim.AdamW(model.parameters(), lr=1e-5)  # Try smaller LR

# Check data
print(train_ds[0])  # Verify data format
```

**3. Low Accuracy**
```python
# Train longer
num_epochs = 10  # Increase from 5

# Check for class imbalance
from collections import Counter
print(Counter(train_ds['label']))
```

**4. Import Errors**
```bash
# Reinstall packages
pip install --upgrade transformers datasets torch
```

---

## FAQ

**Q: How long does training take?**
A: ~60 minutes on GPU (T4/P100), ~4 hours on CPU

**Q: Can I use this for other languages?**
A: Yes, but you'll need a multilingual model and translated dataset

**Q: How much data do I need for my own emotion classifier?**
A: Minimum 1000 examples per class, ideally 5000+

**Q: Can I add more emotion categories?**
A: Yes! Just change `num_labels` and update `label_mapping`

**Q: How do I improve accuracy?**
A: Train longer, use more data, try ModernBERT-large, ensemble models

**Q: Can this detect multiple emotions?**
A: Not currently. You'd need to modify it for multi-label classification

**Q: How do I deploy to production?**
A: Use the Flask API example above, or deploy with Docker/Kubernetes

**Q: Is the model biased?**
A: Potentially yes, as it inherits biases from training data. Test thoroughly!

---

## Conclusion

Congratulations! 🎉 You've built a state-of-the-art emotion classifier.

**What you learned**:
- Transformer architectures (ModernBERT)
- Transfer learning and fine-tuning
- Text classification pipelines
- Model evaluation and interpretation
- Production deployment strategies

**Next steps**:
- Try on your own data
- Experiment with different models
- Deploy to production
- Share your results!

---

## Additional Resources

- [ModernBERT Paper](https://huggingface.co/answerdotai/ModernBERT-base)
- [HuggingFace Transformers Docs](https://huggingface.co/docs/transformers)
- [PyTorch Tutorials](https://pytorch.org/tutorials/)
- [Understanding Transformers](https://jalammar.github.io/illustrated-transformer/)

---

**Happy Classifying! 🚀**
