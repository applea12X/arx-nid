# 🔍 Explainability Module

The arx-nid explainability module provides tools to understand and interpret model predictions using state-of-the-art explainable AI techniques. This module is designed for SOC analysts, security researchers, and compliance teams who need to understand why the network intrusion detection system flagged specific traffic as malicious.

## 🎯 Overview

Our explainability module implements two complementary approaches:

- **SHAP (Shapley Additive Explanations)**: Provides feature-level importance scores showing which network flow characteristics contributed most to a prediction
- **Integrated Gradients**: Shows temporal attribution highlighting which packets or time-steps in a flow sequence were most critical for the decision

## 🚀 Quick Start

### Generate SHAP Explanations

```bash
# Analyze 10 samples from your dataset
python scripts/make_shap.py --limit 10

# Analyze synthetic attack flows
python scripts/make_synthetic.py --attack-type ddos
python scripts/make_shap.py --sample synthetic --limit 1
```

### Generate Integrated Gradients

```bash
# Create temporal attribution maps
python scripts/make_ig.py --samples 5 --steps 50

# Generate interactive heatmap visualization  
python scripts/ig_to_html.py --sample 0
```

### Create Synthetic Attack Flows

```bash
# Generate different types of synthetic attacks
python scripts/make_synthetic.py --attack-type all
```

## 📊 Understanding the Results

### SHAP Force Plots

SHAP force plots show:
- **Base Value**: Expected model output (baseline probability)
- **Red Features**: Push prediction toward "attack" (increase probability)
- **Blue Features**: Push prediction toward "benign" (decrease probability)
- **Feature Names**: Mapped to time step and feature index (e.g., Feature 112 = time step 3, feature 10)

Example interpretation:
```
Feature 112 (t=3, f=10): -0.0030
```
This means at time step 3, feature 10 contributed -0.003 to the prediction, pushing it toward "benign".

### Integrated Gradients Heatmaps

IG heatmaps reveal:
- **Temporal Importance**: Which packets in the sequence were most critical
- **Feature Attribution**: Which flow characteristics at each time step mattered most
- **Attack Patterns**: Visual signatures of different attack types

Color coding:
- 🔴 **Red**: Features that increase attack probability
- 🔵 **Blue**: Features that decrease attack probability  
- ⚪ **White**: Features with minimal influence

## 📁 Output Files

The explainability module generates several types of outputs:

### Reports Directory (`reports/`)

| File | Description | Tracked By |
|------|-------------|------------|
| `shap_N.html` | Interactive SHAP visualization | Git (small) |
| `shap_N.json` | Raw SHAP values and metadata | DVC (large) |
| `ig_heatmap_N.html` | IG temporal heatmap visualization | Git (small) |
| `ig_attr.npy` | Raw attribution tensors | DVC (large) |
| `ig_attr_metadata.json` | IG analysis metadata | Git (small) |

### Synthetic Data (`data/processed/`)

| File | Description | Use Case |
|------|-------------|----------|
| `synthetic_ddos.npy` | DDoS attack simulation | Testing explainability |
| `synthetic_port_scan.npy` | Port scan simulation | Edge case analysis |
| `synthetic_exfiltration.npy` | Data exfiltration pattern | Advanced threat detection |

## 🔧 Configuration & Customization

### SHAP Parameters

```python
# In scripts/make_shap.py
python scripts/make_shap.py \
    --limit 50 \           # Number of samples to explain
    --nsamples 200 \       # Background samples for explanation
    --output-dir reports   # Output directory
```

### Integrated Gradients Parameters

```python  
# In scripts/make_ig.py
python scripts/make_ig.py \
    --samples 10 \         # Number of samples to analyze
    --steps 50 \           # Integration steps (higher = more accurate)
    --output-dir reports   # Output directory  
```

## 🏛️ Architecture

```
arx_nid/
├── explain/
│   ├── __init__.py           # Explainability module
│   └── load_model.py         # Model loading utilities
└── scripts/
    ├── make_shap.py          # SHAP explanation generation
    ├── make_ig.py            # Integrated Gradients computation
    ├── ig_to_html.py         # Heatmap visualization 
    └── make_synthetic.py     # Synthetic attack generation
```

### Model Support

The explainability module supports:
- **ONNX models** for SHAP analysis (faster inference)
- **PyTorch models** for Integrated Gradients (gradient computation)
- **Automatic fallback** to placeholder models for testing

## 🎨 Visualization Examples

### SHAP Force Plot
![SHAP Example](../reports/shap_5.html)
*Interactive force plot showing feature contributions*

### Integrated Gradients Heatmap  
![IG Example](../reports/ig_heatmap_0.html)
*Temporal attribution heatmap highlighting important time steps*

## 📈 Use Cases

### SOC Analysis Workflow

1. **Alert Investigation**: When IDS flags suspicious traffic
   ```bash
   python scripts/make_shap.py --limit 1 --data-path suspicious_flow.npy
   ```

2. **Temporal Analysis**: Understanding attack progression
   ```bash
   python scripts/make_ig.py --samples 1 --data-path attack_sequence.npy
   python scripts/ig_to_html.py --sample 0
   ```

3. **Pattern Recognition**: Compare explanations across attack types
   ```bash
   python scripts/make_synthetic.py --attack-type all
   # Generate explanations for each synthetic attack type
   ```

### Compliance & Auditing

- **Regulatory Requirements**: Document model decision-making process
- **Bias Detection**: Identify unexpected feature dependencies  
- **Model Validation**: Verify predictions align with security knowledge

### Research & Development

- **Feature Engineering**: Identify most predictive network characteristics
- **Model Debugging**: Understand failure modes and edge cases
- **Attack Evolution**: Analyze new attack patterns

## 🔬 Technical Details

### SHAP Implementation

- **Algorithm**: Kernel SHAP for model-agnostic explanations
- **Background**: 100 samples from training data
- **Approximation**: 200 evaluations per sample (configurable)
- **Output**: Additive feature attributions summing to prediction

### Integrated Gradients Implementation

- **Baseline**: Zero tensor (neutral network state)
- **Integration**: 50 steps along straight-line path (configurable)  
- **Targets**: Regression output (no target specification needed)
- **Output**: Per-feature, per-timestep attribution scores

## 🚨 Troubleshooting

### Common Issues

**SHAP fails with NumPy compatibility error:**
```bash
pip install "numpy==1.26.4" --force-reinstall
```

**IG fails with target specification error:**
- Remove target parameter for regression models
- Check model output shape matches expected format

**Model not found errors:**
- Scripts automatically create placeholder models for testing
- Replace with actual trained models for production use

### Performance Optimization

```bash
# Faster SHAP (fewer background samples)
python scripts/make_shap.py --limit 10 --nsamples 50

# Faster IG (fewer integration steps)
python scripts/make_ig.py --samples 5 --steps 20
```

## 🤝 Contributing

To extend the explainability module:

1. **Add new explanation methods**: Create scripts following the pattern in `scripts/`
2. **Enhance visualizations**: Modify HTML templates in visualization scripts
3. **Support new model formats**: Extend `load_model.py` with new model types
4. **Add attack patterns**: Expand `make_synthetic.py` with new synthetic attacks

## 📚 References

- [SHAP: A Unified Approach to Explaining the Output of Any Machine Learning Model](https://proceedings.neurips.cc/paper/2017/file/8a20a8621978632d76c43dfd28b67767-Paper.pdf)
- [Axiomatic Attribution for Deep Networks](https://proceedings.mlr.press/v70/sundararajan17a.html)
- [Captum: A unified and generic model interpretability library for PyTorch](https://arxiv.org/abs/2009.07896)

---

*Generated by arx-nid explainability module • Last updated: September 2025*
