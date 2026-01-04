# arx-nid

**Adversarial-Robust, Explainable Network-Intrusion Detector**

A production-ready deep-learning IDS trained on 2024-era traffic datasets, hardened with adversarial training using IBM's ART library, and equipped with state-of-the-art explainability (SHAP, Integrated Gradients).

## 🎯 Features

- **Modern Datasets**: Trained on 2024 network traffic (HiTar, CIC IoV, LSNM)
- **Multiple Models**: Baseline (Logistic Regression, Random Forest) + Deep Learning (Bi-LSTM)
- **Adversarial Robustness**: FGSM and PGD attack generation with adversarial training
- **Explainability**: SHAP force plots and Integrated Gradients for interpretable predictions
- **MLOps Ready**: MLflow experiment tracking, DVC data versioning, automated CI/CD
- **Production Code**: Clean architecture, comprehensive tests, pre-commit hooks

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/arx-nid.git
cd arx-nid

# Install dependencies
pip install -r requirements.txt

# Set up pre-commit hooks
pre-commit install
```

### Training Models

```bash
# Generate tensor data
python scripts/create_tensors.py

# Train baseline models
python scripts/train_baseline.py

# Train BiLSTM model
python scripts/train_lstm.py --epochs 50

# View results in MLflow
mlflow ui
```

### Explainability

```bash
# Generate SHAP explanations
python scripts/make_shap.py --limit 10

# Create Integrated Gradients visualizations
python scripts/make_ig.py --samples 5
python scripts/ig_to_html.py --sample 0

# Create synthetic attacks for testing
python scripts/make_synthetic.py --attack-type all
```

### Hyperparameter Optimization

```bash
python scripts/hyperparameter_search.py --n-trials 50
```

## 📊 Model Performance

**Baseline Models:**
- Logistic Regression: 90.5% accuracy, 0.84 ROC AUC
- Random Forest: 88.1% accuracy, 0.89 ROC AUC

**Deep Learning:**
- Bi-LSTM: 83.3% accuracy, 0.92 ROC AUC, 100% recall on attacks

## 📂 Project Structure

```
arx-nid/
├── arx_nid/              # Main package
│   ├── features/         # Feature transformers
│   ├── models/           # Model architectures (BiLSTM, baselines)
│   ├── explain/          # Explainability module
│   └── security/         # Adversarial robustness (IBM ART)
├── scripts/              # Training and analysis scripts
│   ├── train_baseline.py
│   ├── train_lstm.py
│   ├── adversarial_train.py
│   ├── evaluate_robustness.py
│   ├── make_shap.py
│   └── make_ig.py
├── data/                 # Data directory (DVC tracked)
├── models/               # Saved models
├── reports/              # Explainability reports
├── tests/                # Unit tests
└── docs/                 # Documentation
```

## 📖 Documentation

- **Training Guide**: [docs/training_guide.md](docs/training_guide.md)
- **Explainability**: [docs/explainability.md](docs/explainability.md)
- **Data Catalogue**: [data/README.md](data/README.md)

## 🛡️ Adversarial Robustness

Evaluate model robustness against adversarial attacks:

```bash
python scripts/evaluate_robustness.py --model models/lstm/best_lstm_model.pt
```

Generates comprehensive robustness reports with:
- FGSM and PGD attack evaluations
- Accuracy vs epsilon plots
- Perturbation analysis
- Attack success rates

## 🔍 Explainability Examples

The project includes production-ready explainability:

- **SHAP Force Plots**: Feature-level importance for individual predictions
- **Integrated Gradients**: Temporal attribution showing which packets matter most
- **Synthetic Attacks**: DDoS, port scan, and data exfiltration simulations

See [docs/explainability.md](docs/explainability.md) for details.

## 🧪 Development

```bash
# Run tests
pytest tests/

# Format code
black arx_nid/ scripts/ tests/
ruff check arx_nid/ scripts/ tests/ --fix

# Generate test data
python scripts/create_test_data.py --num-flows 1000
```

## 📦 Dependencies

- **ML/DL**: PyTorch, scikit-learn, MLflow, Optuna
- **Data**: pandas, pyarrow, DVC
- **Explainability**: SHAP, Captum, ONNX Runtime
- **Security**: adversarial-robustness-toolbox
- **DevOps**: pre-commit, pytest, Black, Ruff

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Datasets**: HiTar-2024, CIC IoV-2024, LSNM-2024, CTAP
- **Libraries**: IBM ART, SHAP, Captum, MLflow, PyTorch
- **Community**: Network security and ML research communities

## 📧 Contact

For questions or collaborations, please open an issue on GitHub.

---

**arx-nid** - Adversarial-Robust, Explainable Network Intrusion Detection
