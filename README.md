# Parkinson's Disease Signal Classification with Multi-Scale and Frequency-Aware CNNs

This project provides a comprehensive framework for time series classification, specifically designed for Parkinson's disease diagnosis using wearable sensor data. The framework supports various model architectures, including multi-scale convolution, frequency-aware modules, channel attention, and their combinations. It also includes robust cross-validation, visualization, and evaluation tools.

---

## Features

- **Flexible Model Selection**: Easily switch between baseline, multi-scale, frequency-aware, attention, and combined models via command-line arguments.
- **Nested Cross-Validation**: Supports robust performance evaluation with nested K-Fold cross-validation.
- **Feature Extractor Options**: Enable or disable advanced feature extractors (multi-scale, frequency, attention) as needed.
- **Comprehensive Evaluation**: Outputs detailed metrics, confusion matrices, ROC and PR curves, and boxplots for each fold.
- **Visualization**: Includes t-SNE and UMAP scripts for feature space visualization.

---

## Directory Structure

```
|-- main.py                    # Main training and evaluation script (nested CV)
|-- model.py                   # Combined model definitions (multi-scale, frequency, attention)
|-- model_base.py              # Baseline CNN model
|-- model_multiscale.py        # Multi-scale only model
|-- model_frequency.py         # Frequency-aware only model
|-- model_attention.py         # Channel attention only model
|-- train.py                   # Training utilities
|-- evaluate.py                # Evaluation utilities
|-- data_loader.py             # Data loading utilities
|-- tsne_visualization.py      # t-SNE feature visualization
|-- run_hand_evaluation.py     # Hand-specific evaluation script
|-- run_task_evaluation.py     # Task-specific evaluation script
|-- requirements.txt           # Python dependencies
|-- results.csv                # Example output metrics
|-- ... (other scripts and result folders)
```

---

## Installation

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd <your-project-directory>
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

---

## Usage

### 1. Main Training & Evaluation

Run nested cross-validation with your chosen model type:

```bash
python main.py --model_type full --use_feature_extractor
```

**Model type options:**
- `base`         : Baseline CNN
- `multiscale`   : Multi-scale convolution only
- `frequency`    : Frequency-aware only
- `attention`    : Channel attention only
- `full`         : Multi-scale + frequency + attention
- `full_noattn`  : Multi-scale + frequency (no attention)

**Other useful arguments:**
- `--data_path`           : Path to preprocessed data (default: ../PADS/preprocessed)
- `--output_dir`          : Output directory for results
- `--epochs`              : Number of training epochs
- `--batch_size`          : Batch size
- `--tremor_freq_band`    : Tremor frequency band, e.g. "3,7"
- `--sampling_rate`       : Signal sampling rate (Hz)
- `--use_feature_extractor` : Enable advanced feature extractor

### 2. Hand or Task-Specific Evaluation

```bash
python run_hand_evaluation.py --model_type full --use_feature_extractor
python run_task_evaluation.py --model_type full --use_feature_extractor
```

### 3. Feature Visualization

- **t-SNE**:
  ```bash
  python tsne_visualization.py --model_path <path_to_model.pth> --use_feature_extractor
  ```
- **UMAP**:
  ```bash
  python umap_visualization.py --model_path <path_to_model.pth> --use_feature_extractor
  ```

---

## Model Architecture

- **Multi-Scale Feature Extractor**: Parallel 1D convolutions with different kernel sizes to capture features at multiple temporal scales.
- **Frequency-Aware Module**: Focuses on tremor-related frequency bands (e.g., 3-7 Hz) using frequency-domain operations.
- **Channel Attention (optional)**: Learns the importance of each sensor channel.
- **Classifier**: Stacked convolutional layers, global pooling, and fully connected layers.

You can flexibly combine these modules using the `--model_type` and `--use_feature_extractor` arguments.

---

## Results & Outputs

- **results.csv**: Summary of metrics for each fold and the overall average.
- **Boxplots**: Visualize metric distributions across folds.
- **ROC/PR Curves**: For each fold and overall.
- **Confusion Matrices**: For each test fold.
- **Feature Visualizations**: t-SNE/UMAP plots for raw and learned features.

---

## Requirements

- numpy==1.24.3
- pandas==2.0.3
- torch==2.0.1
- scikit-learn==1.3.0
- matplotlib==3.7.2
- tqdm==4.65.0

Install with:
```bash
pip install -r requirements.txt
```

---

## Citation

If you use this codebase for your research, please cite the original authors and this repository.

---

## License

This project is for academic research and educational purposes.

---

**For more details, please refer to the code comments and docstrings in each script. If you have questions, feel free to open an issue or contact the maintainer.** 
