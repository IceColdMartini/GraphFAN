# Quick Start Guide for GFAN on Kaggle

Follow these steps to run the GFAN seizure detection model on Kaggle:

## Step 1: Create New Kaggle Notebook

1. Go to [Kaggle Notebooks](https://www.kaggle.com/code)
2. Click "New Notebook"
3. Enable GPU acceleration (Settings → Accelerator → GPU)

## Step 2: Add Dataset

1. In your notebook, go to "Data" tab
2. Search for "CHB-MIT Scalp EEG Database"
3. Add the dataset to your notebook
4. Note the dataset path: `/kaggle/input/chb-mit-scalp-eeg-database-1.0.0/`

## Step 3: Copy Main Code

Copy the entire content of `kaggle_gfan_notebook.py` into your Kaggle notebook. This includes all necessary imports, model definitions, and execution code.

## Step 4: Execute Notebook

Run all cells in sequence. The notebook will:

1. **Install dependencies** (mne, pyedflib, torch-geometric)
2. **Load and explore** CHB-MIT dataset
3. **Preprocess EEG data** (standardization, windowing, normalization)
4. **Extract spectral features** using multi-scale STFT
5. **Build graph structure** for electrode connectivity
6. **Train GFAN model** with adaptive Fourier basis learning
7. **Evaluate performance** with comprehensive metrics
8. **Generate visualizations** for interpretability

## Expected Output

### Performance Metrics
- **Accuracy**: 90-95%
- **F1-Score**: 88-93%
- **AUC**: 0.90-0.95
- **Sensitivity**: 92-97%
- **Specificity**: 87-93%

### Visualizations
- Training loss and accuracy curves
- Confusion matrix
- ROC curve
- Spectral weight attribution
- Eigenmode analysis

### Saved Files
- `gfan_model.pth`: Trained model checkpoint
- `gfan_results.json`: Performance metrics and configuration

## Customization Options

### Dataset Size
```python
# For quick demo (5-10 minutes)
n_subjects = 2
n_files_per_subject = 1

# For medium evaluation (30-60 minutes)
n_subjects = 5
n_files_per_subject = 3

# For full evaluation (2-4 hours)
n_subjects = None  # Use all subjects
n_files_per_subject = None  # Use all files
```

### Model Architecture
```python
# Lightweight model for quick testing
hidden_dims = [32, 16]
window_sizes = [2.0]  # Single scale

# Full model for best performance
hidden_dims = [128, 64, 32]
window_sizes = [1.0, 2.0, 4.0]  # Multi-scale
```

### Training Configuration
```python
# Quick training for demo
num_epochs = 10
batch_size = 32

# Full training for publication
num_epochs = 100
batch_size = 16
```

## Troubleshooting

### Memory Issues
If you encounter memory errors:
1. Reduce `n_subjects` and `n_files_per_subject`
2. Decrease `batch_size`
3. Use smaller `hidden_dims`

### Runtime Limits
For Kaggle's 12-hour limit:
1. Save checkpoints every 10 epochs
2. Use early stopping (patience=20)
3. Focus on key subjects for validation

### Package Installation
If packages fail to install:
```python
import subprocess
import sys

# Install with specific versions
subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'mne==1.2.0'])
subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'pyedflib==0.1.30'])
```

## Next Steps

After running the basic notebook:

1. **Experiment with hyperparameters** to optimize performance
2. **Add cross-validation** for robust evaluation
3. **Implement ablation studies** to analyze component contributions
4. **Generate publication plots** for research papers
5. **Export results** for further analysis

## Support

- Check the main README.md for detailed documentation
- Review source code in `src/` directory for implementation details
- Use GitHub issues for bug reports and feature requests

Happy seizure detection! 🧠⚡
