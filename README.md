# ModernBERT Emotion Classifier 🎭

A fine-tuned **ModernBERT** model for classifying text into 6 emotion categories: sadness, joy, love, anger, fear, and surprise.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![HuggingFace](https://img.shields.io/badge/🤗-Transformers-yellow.svg)](https://huggingface.co/)

## 📊 Model Performance

- **Validation Accuracy**: 94.35%
- **Base Model**: [answerdotai/ModernBERT-base](https://huggingface.co/answerdotai/ModernBERT-base)
- **Dataset**: [DAIR-AI Emotion Dataset](https://huggingface.co/datasets/dair-ai/emotion)
- **Training Examples**: 16,000
- **Validation Examples**: 2,000

## 🎯 Emotion Categories

| Emotion | Label | Example |
|---------|-------|---------|
| 😢 Sadness | 0 | "I feel so lonely and depressed" |
| 😊 Joy | 1 | "I'm so happy and excited!" |
| ❤️ Love | 2 | "I absolutely adore spending time with you" |
| 😠 Anger | 3 | "I can't believe you did that, I'm furious!" |
| 😨 Fear | 4 | "I'm really scared about the exam" |
| 😲 Surprise | 5 | "Wow, I didn't expect that at all!" |

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/ModernBERT-Classifier.git
cd ModernBERT-Classifier

# Install required packages
pip install torch transformers datasets matplotlib tqdm
```

### Using the Model

```python
from transformers import AutoTokenizer, ModernBertForSequenceClassification
import torch
import torch.nn.functional as F

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base")
model = ModernBertForSequenceClassification.from_pretrained("answerdotai/ModernBERT-base", num_labels=6)

# Load trained weights
model.load_state_dict(torch.load('emotion_classifier_model.pth'))
model.eval()

# Emotion labels
label_mapping = {0: "sadness", 1: "joy", 2: "love", 3: "anger", 4: "fear", 5: "surprise"}

# Predict emotion
def predict_emotion(text):
    inputs = tokenizer(text, padding="max_length", truncation=True, max_length=300, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)
        probabilities = F.softmax(outputs.logits, dim=1)[0]
        confidence, predicted_class = torch.max(probabilities, dim=0)

    return label_mapping[predicted_class.item()], confidence.item()

# Example usage
text = "I'm so happy about my new job!"
emotion, confidence = predict_emotion(text)
print(f"Emotion: {emotion} (Confidence: {confidence*100:.2f}%)")
```

## 📓 Tutorial

The main tutorial is available in the Jupyter notebook: [`ModernBERT_Classifier.ipynb`](ModernBERT_Classifier.ipynb)

### What's Covered:

1. **Installation** - Setting up the environment
2. **Data Loading** - Loading and exploring the DAIR-AI emotion dataset
3. **Tokenization** - Converting text to model-ready format
4. **Model Training** - Fine-tuning ModernBERT for emotion classification
5. **Evaluation** - Confusion matrix and accuracy metrics
6. **Model Saving** - Exporting the trained model
7. **Inference** - Using the model to predict emotions in new text

## 🏗️ Architecture

```
Input Text
    ↓
Tokenizer (WordPiece)
    ↓
ModernBERT Encoder (12 layers, 768 hidden dimensions)
    ↓
Classification Head (Linear layer, 6 outputs)
    ↓
Softmax
    ↓
Emotion Prediction
```

## 🎓 Training Details

### Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| Learning Rate | 1e-4 | Step size for gradient descent |
| Weight Decay | 1e-2 | L2 regularization strength |
| Batch Size | 32 | Number of examples per batch |
| Epochs | 5 | Complete passes through training data |
| Max Length | 300 | Maximum sequence length (tokens) |
| Optimizer | AdamW | Adam with weight decay |
| Loss Function | CrossEntropyLoss | Multi-class classification loss |

### Training Results

| Epoch | Average Loss | Change |
|-------|-------------|---------|
| 1 | 0.4261 | - |
| 2 | 0.1211 | ↓ 71.6% |
| 3 | 0.1039 | ↓ 14.2% |

The model shows strong convergence with decreasing loss across epochs.

## 📈 Evaluation

### Confusion Matrix

The confusion matrix shows the model's prediction accuracy for each emotion category. Diagonal values represent correct predictions, while off-diagonal values show misclassifications.

**Key Insights:**
- High diagonal values (>0.9) across all emotions
- Minor confusion between similar emotions (e.g., sadness/fear, joy/love)
- Consistent performance across all 6 categories

### Performance Metrics

- **Overall Accuracy**: 94.35%
- **Training Loss**: Decreased from 0.4261 to 0.1039
- **Convergence**: Stable after 3 epochs

## 🔧 Advanced Usage

### Fine-tuning on Your Own Data

```python
from datasets import load_dataset
from torch.utils.data import DataLoader

# Load your custom dataset
# Format: {"text": "...", "label": 0-5}
custom_ds = load_dataset("your_dataset")

# Tokenize
def tokenization(item):
    return tokenizer(item['text'], padding="max_length", truncation=True, max_length=300)

custom_ds = custom_ds.map(tokenization, batched=True)
custom_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

# Create DataLoader
train_loader = DataLoader(custom_ds, batch_size=32, shuffle=True)

# Train the model (see notebook for full training loop)
```

### Batch Prediction

```python
def predict_batch(texts):
    """Predict emotions for multiple texts efficiently"""
    inputs = tokenizer(texts, padding="max_length", truncation=True, max_length=300, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)
        probabilities = F.softmax(outputs.logits, dim=1)
        predictions = torch.argmax(probabilities, dim=1)

    return [label_mapping[p.item()] for p in predictions]

# Example
texts = ["I'm so happy!", "This is terrible.", "I love this!"]
emotions = predict_batch(texts)
print(emotions)  # ['joy', 'sadness', 'love']
```

## 📦 Model Files

- **`ModernBERT_Classifier.ipynb`**: Complete training and evaluation notebook
- **`emotion_classifier_model.pth`**: Trained model weights (generated after training)
- **`README.md`**: This file
- **`TUTORIAL.md`**: Detailed step-by-step tutorial

## 🎯 Use Cases

- **Social Media Monitoring**: Analyze sentiment and emotions in tweets, posts, comments
- **Customer Feedback**: Understand emotional tone in reviews and support tickets
- **Mental Health**: Detect emotional states in therapy or counseling contexts
- **Content Moderation**: Flag emotionally charged content
- **Market Research**: Analyze emotional responses to products/campaigns
- **Chatbots**: Enable emotion-aware conversational AI

## ⚠️ Limitations

1. **Context Length**: Limited to 300 tokens, may miss longer context
2. **Sarcasm/Irony**: May struggle with non-literal language
3. **Mixed Emotions**: Designed for single-emotion classification
4. **Cultural Bias**: Training data may reflect specific cultural contexts
5. **Domain Specificity**: Best performance on text similar to training data (tweets/short text)

## 🔮 Future Improvements

- [ ] Train on larger ModernBERT-large model
- [ ] Multi-label classification for mixed emotions
- [ ] Increase context length to 512 tokens
- [ ] Add attention visualization
- [ ] Deploy as REST API
- [ ] Create web demo interface
- [ ] Export to ONNX for faster inference
- [ ] Fine-tune on domain-specific data

## 📚 References

- **ModernBERT**: [Answer.AI ModernBERT](https://huggingface.co/answerdotai/ModernBERT-base)
- **Dataset**: [DAIR-AI Emotion Dataset](https://huggingface.co/datasets/dair-ai/emotion)
- **Paper**: Transformer-based models for emotion classification
- **HuggingFace**: [Transformers Documentation](https://huggingface.co/docs/transformers/index)

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. Areas for contribution:

- Improve model performance
- Add more evaluation metrics
- Create deployment examples
- Add data augmentation
- Improve documentation

## 📝 License

This project is open source and available under the MIT License.

## 🙏 Acknowledgments

- Answer.AI for the ModernBERT model
- DAIR-AI for the emotion dataset
- HuggingFace for the Transformers library
- PyTorch team for the deep learning framework

## 📧 Contact

For questions, issues, or suggestions, please open an issue on GitHub.

---

**Built with ❤️ using ModernBERT and PyTorch**
