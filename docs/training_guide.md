# 🎯 Model Training Guide

Complete guide for training network intrusion detection models in the arx-nid project.

## 📋 Table of Contents

- [Quick Start](#quick-start)
- [Baseline Models](#baseline-models)
- [Deep Learning Models](#deep-learning-models)
- [Hyperparameter Optimization](#hyperparameter-optimization)
- [Adversarial Robustness](#adversarial-robustness)
- [Model Evaluation](#model-evaluation)
- [MLflow Tracking](#mlflow-tracking)

---

## 🚀 Quick Start

### Prerequisites

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Ensure tensor data is generated:
```bash
python scripts/create_tensors.py
```

### Train All Models

```bash
# 1. Train baseline models
python scripts/train_baseline.py

# 2. Train BiLSTM model
python scripts/train_lstm.py --epochs 50

# 3. Evaluate robustness (when compatible)
python scripts/evaluate_robustness.py --model models/lstm/best_lstm_model.pt
```

---

## 📊 Baseline Models

### Logistic Regression & Random Forest

Train simple, interpretable models for comparison:

```bash
python scripts/train_baseline.py \
  --models logistic random_forest \
  --data data/processed/tensor_v0.npy \
  --output-dir models/baselines \
  --test-size 0.2
```

**Parameters:**
- `--models`: Which models to train (`logistic`, `random_forest`)
- `--data`: Path to tensor data
- `--output-dir`: Where to save trained models
- `--test-size`: Proportion for test set (default: 0.2)
- `--rf-n-estimators`: Number of trees for Random Forest (default: 100)
- `--rf-max-depth`: Max depth for Random Forest (default: 20)

**Output:**
- `models/baselines/logistic_model.model.pkl` - Trained logistic regression
- `models/baselines/random_forest_model.model.pkl` - Trained random forest
- MLflow experiment: `arx-nid-baselines`

**Example Results:**
```
LOGISTIC:
  accuracy: 0.9048
  precision: 0.6667
  recall: 0.4000
  f1_score: 0.5000
  roc_auc: 0.8432

RANDOM_FOREST:
  accuracy: 0.8810
  roc_auc: 0.8865
```

---

## 🧠 Deep Learning Models

### Bi-LSTM Classifier

Train a bidirectional LSTM for sequence-based intrusion detection:

```bash
python scripts/train_lstm.py \
  --data data/processed/tensor_v0.npy \
  --output-dir models/lstm \
  --hidden-size 64 \
  --num-layers 2 \
  --dropout 0.3 \
  --batch-size 32 \
  --epochs 50 \
  --lr 0.001 \
  --patience 10
```

**Parameters:**
- `--hidden-size`: Number of LSTM hidden units (default: 64)
- `--num-layers`: Number of stacked LSTM layers (default: 2)
- `--dropout`: Dropout probability (default: 0.3)
- `--batch-size`: Training batch size (default: 32)
- `--epochs`: Maximum training epochs (default: 50)
- `--lr`: Learning rate (default: 0.001)
- `--patience`: Early stopping patience (default: 10)
- `--test-size`: Test set proportion (default: 0.2)
- `--val-size`: Validation set proportion from training (default: 0.1)

**Output:**
- `models/lstm/best_lstm_model.pt` - Best model checkpoint
- MLflow experiment: `arx-nid-deep-learning`

**Example Results:**
```
BiLSTM Classifier Summary
======================================================================
Input size:        34
Hidden size:       64
Num layers:        2
Total parameters:  158,977
======================================================================

Training Results:
  accuracy: 0.8333
  precision: 0.4167
  recall: 1.0000
  f1_score: 0.5882
  roc_auc: 0.9189
```

### Model Architecture

The BiLSTM model (arx_nid/models/lstm.py:1) consists of:
1. **Bi-directional LSTM layers** - Capture temporal dependencies
2. **Dropout** - Regularization to prevent overfitting
3. **Fully connected layers** - Classification head
4. **Batch normalization** - Stabilize training

---

## 🔍 Hyperparameter Optimization

Use Optuna for automated hyperparameter search:

```bash
python scripts/hyperparameter_search.py \
  --data data/processed/tensor_v0.npy \
  --output-dir models/optuna \
  --n-trials 50 \
  --timeout 3600
```

**Parameters:**
- `--n-trials`: Number of optimization trials (default: 50)
- `--timeout`: Timeout in seconds
- `--test-size`: Test set proportion (default: 0.2)

**Optimized Hyperparameters:**
- `hidden_size` ∈ [32, 64, 96, 128]
- `num_layers` ∈ [1, 2, 3]
- `dropout` ∈ [0.1, 0.5]
- `lr` ∈ [0.0001, 0.01] (log scale)
- `batch_size` ∈ [16, 32, 64]

**Output:**
- `models/optuna/study.pkl` - Complete Optuna study
- `models/optuna/best_params.txt` - Best hyperparameters
- `models/optuna/optimization_history.html` - Visualization
- `models/optuna/param_importances.html` - Parameter importance plot
- MLflow experiment: `arx-nid-hyperparameter-search`

**Using Best Parameters:**

After optimization, train final model with best parameters:
```bash
python scripts/train_lstm.py \
  --hidden-size 64 \
  --num-layers 2 \
  --dropout 0.3 \
  --lr 0.001 \
  --batch-size 32 \
  --epochs 100
```

---

## 🛡️ Adversarial Robustness

### Adversarial Training

Train models robust to adversarial attacks using IBM ART:

```bash
python scripts/adversarial_train.py \
  --model-path models/lstm/best_lstm_model.pt \
  --output-dir models/adversarial \
  --epochs 30 \
  --batch-size 16 \
  --attack-eps 0.1 \
  --attack-ratio 0.5
```

**Parameters:**
- `--model-path`: Pre-trained model to fine-tune (optional)
- `--attack-eps`: Attack epsilon (perturbation magnitude, default: 0.1)
- `--attack-ratio`: Ratio of adversarial examples in training (default: 0.5)
- `--epochs`: Adversarial training epochs (default: 30)

**Output:**
- `models/adversarial/adversarial_trained_model.pt` - Robust model
- MLflow experiment: `arx-nid-adversarial-training`

**Note:** Current version has some ART compatibility issues with PyTorch float32/float64 that are being resolved.

### Robustness Evaluation

Evaluate model robustness against FGSM and PGD attacks:

```bash
python scripts/evaluate_robustness.py \
  --model models/lstm/best_lstm_model.pt \
  --output-dir reports/robustness
```

**Output:**
- `reports/robustness/robustness_evaluation.csv` - Detailed metrics
- `reports/robustness/robustness_evaluation.json` - JSON results
- `reports/robustness/robustness_vs_epsilon.png` - Accuracy vs attack strength
- `reports/robustness/accuracy_comparison.png` - Clean vs adversarial accuracy
- `reports/robustness/perturbation_analysis.png` - Perturbation magnitude analysis

**Attack Suite:**
- FGSM with ε ∈ {0.01, 0.05, 0.10, 0.20}
- PGD with ε ∈ {0.01, 0.05, 0.10, 0.20} (40 iterations)

**Metrics Tracked:**
- Clean accuracy
- Adversarial accuracy
- Accuracy drop
- Attack success rate
- L2 and L∞ perturbation magnitudes

---

## 📈 Model Evaluation

### Standard Metrics

All training scripts report:
- **Accuracy**: Overall classification accuracy
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **F1 Score**: Harmonic mean of precision and recall
- **ROC AUC**: Area under the ROC curve

### Confusion Matrix

Training scripts output confusion matrices:
```
Confusion Matrix:
[[30  7]
 [ 0  5]]

              precision    recall  f1-score   support
      Benign       1.00      0.81      0.90        37
      Attack       0.42      1.00      0.59         5
```

### Model Comparison

Compare different models using MLflow:
```bash
mlflow ui
# Navigate to http://localhost:5000
```

---

## 🔬 MLflow Tracking

### Experiments

All training runs are logged to MLflow:
- `arx-nid-baselines` - Logistic Regression & Random Forest
- `arx-nid-deep-learning` - BiLSTM models
- `arx-nid-hyperparameter-search` - Optuna optimization
- `arx-nid-adversarial-training` - Adversarially trained models

### Viewing Results

Start MLflow UI:
```bash
mlflow ui
```

Navigate to `http://localhost:5000` to:
- Compare model performance
- View training curves
- Download trained models
- Track hyperparameters

### Logged Artifacts

For each run, MLflow logs:
- Model weights
- Training metrics
- Hyperparameters
- Confusion matrices
- Model architecture

---

## 🎓 Advanced Usage

### Custom Data

To train on custom data:

1. Create tensor data:
```bash
python scripts/create_tensors.py \
  --input data/processed/my_flows.parquet \
  --output data/processed/my_tensor.npy \
  --window-size 20
```

2. Train models:
```bash
python scripts/train_lstm.py --data data/processed/my_tensor.npy
```

### Multi-Class Classification

For multi-class problems, update `num_classes` in arx_nid/models/lstm.py:48:

```python
model = BiLSTMClassifier(
    input_size=input_size,
    num_classes=3,  # For 3-class classification
    ...
)
```

### Transfer Learning

Fine-tune pre-trained models:
```bash
# Train initial model
python scripts/train_lstm.py --epochs 50 --output-dir models/pretrained

# Fine-tune on new data
python scripts/train_lstm.py \
  --model-path models/pretrained/best_lstm_model.pt \
  --data data/processed/new_data.npy \
  --epochs 20 \
  --lr 0.0001  # Lower learning rate for fine-tuning
```

---

## 🐛 Troubleshooting

### Out of Memory

Reduce batch size:
```bash
python scripts/train_lstm.py --batch-size 8
```

### Slow Training

Use GPU if available:
- PyTorch automatically detects CUDA
- Verify with: `python -c "import torch; print(torch.cuda.is_available())"`

### Poor Performance

1. Check data quality
2. Try hyperparameter optimization
3. Increase model capacity (hidden_size, num_layers)
4. Add more training data
5. Adjust class imbalance

### MLflow Issues

Reset MLflow database:
```bash
rm -rf mlruns/
mlflow server
```

---

## 📚 References

- **BiLSTM Architecture**: arx_nid/models/lstm.py:1
- **Baseline Models**: arx_nid/models/baselines.py:1
- **Training Scripts**: scripts/train_*.py
- **Adversarial Module**: arx_nid/security/

---

*Last updated: January 2026*
