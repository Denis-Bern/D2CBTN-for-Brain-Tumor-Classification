# Brain Tumor Classification using Deep Learning Models

This repository contains a comprehensive implementation of brain tumor classification using multiple state-of-the-art deep learning models. The project implements 10-fold cross-validation for robust evaluation and includes six different architectures for comparative analysis.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Dataset Structure](#dataset-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Models](#models)
- [Results](#results)
- [File Structure](#file-structure)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)

## Overview

This project implements brain tumor classification using multiple deep learning architectures to classify brain MRI images into four categories:
- **Glioma Tumor**
- **Meningioma Tumor** 
- **No Tumor**
- **Pituitary Tumor**

The implementation includes comprehensive evaluation metrics, data augmentation, and 10-fold cross-validation for robust performance assessment.

## Features

- **Multiple Model Architectures**: 6 different deep learning models
- **10-Fold Cross-Validation**: Robust evaluation methodology
- **Data Augmentation**: Enhanced training with image transformations
- **Comprehensive Metrics**: Accuracy, Precision, Recall, F1-Score, AUC-ROC
- **GPU Support**: Multi-GPU training with TensorFlow
- **Visualization**: Confusion matrices, learning curves, and ROC plots
- **Model Persistence**: Automatic model weight saving

## Dataset Structure

Organize your dataset in the following structure:

```
Dataset/
├── Training/
│   ├── glioma_tumor/
│   │   ├── image1.jpg
│   │   ├── image2.jpg
│   │   └── ...
│   ├── meningioma_tumor/
│   │   ├── image1.jpg
│   │   ├── image2.jpg
│   │   └── ...
│   ├── no_tumor/
│   │   ├── image1.jpg
│   │   ├── image2.jpg
│   │   └── ...
│   └── pituitary_tumor/
│       ├── image1.jpg
│       ├── image2.jpg
│       └── ...
└── Testing/
    ├── glioma_tumor/
    ├── meningioma_tumor/
    ├── no_tumor/
    └── pituitary_tumor/
```

## Installation

### Prerequisites

- Python 3.7 or higher
- CUDA-compatible GPU (recommended)
- CUDA Toolkit 11.0 or higher
- cuDNN 8.0 or higher

### Step 1: Clone the Repository

```bash
git clone <repository-url>
cd D2CBTN
```

### Step 2: Create Virtual Environment

```bash
# Using conda
conda create -n brain_tumor_classification python=3.8
conda activate brain_tumor_classification

# Or using venv
python -m venv brain_tumor_env
source brain_tumor_env/bin/activate  # On Windows: brain_tumor_env\Scripts\activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Install Additional Dependencies

```bash
pip install tensorflow==2.10.0
pip install opencv-python==4.6.0.66
pip install scikit-learn==1.1.2
pip install matplotlib==3.5.3
pip install seaborn==0.11.2
pip install pandas==1.5.0
pip install numpy==1.23.3
pip install tqdm==4.64.1
pip install ipywidgets==7.7.0
pip install Pillow==9.3.0
```

### Step 5: Verify Installation

```python
import tensorflow as tf
print(f"TensorFlow version: {tf.__version__}")
print(f"GPU available: {tf.config.list_physical_devices('GPU')}")
```

## Usage

### 1. Prepare Your Dataset

Place your brain MRI images in the appropriate folders according to the dataset structure above.

### 2. Run the Models

Each model can be run independently. Choose the model you want to train:

#### EfficientNetB0 Model
```bash
python 10_FOLD_EfficientNetB0.py
```

#### DenseNet121 Model
```bash
python 10_FOLD_DenseNet121.py
```

#### ResNet152 Model
```bash
python 10_FOLD_ResNet152.py
```

#### Ensemble Model (EfficientNet + DenseNet)
```bash
python 10_FOLD_EDNet.py
```

#### Swin Transformer Model
```bash
python 10_FOLD_Swin_Transformer.py
```

#### Vision Transformer Model
```bash
python 10_FOLD_Vision_Transformer.py
```

### 3. Monitor Training

The training process will:
- Display progress bars for data loading
- Show training and validation metrics
- Generate confusion matrices for each fold
- Plot learning curves and ROC curves
- Save model weights automatically

### 4. View Results

Results are automatically saved in model-specific directories:
- `10_FOLD_EfficientNetB0_Model/`
- `10_FOLD_DenseNet121_Model/`
- `10_FOLD_ResNet152_FINAL_Model/`
- `10_FOLD_Ensemble_EfficientNet_DenseNet121_Model/`
- `10_FOLD_Swin_Transformer_Model/`
- `10_FOLD_Vision_Transformer_Model/`

## Models

### 1. EfficientNetB0
- **Architecture**: EfficientNetB0 with custom classification head
- **Input Size**: 224×224×3
- **Features**: Efficient CNN architecture with compound scaling
- **File**: `10_FOLD_EfficientNetB0.py`

### 2. DenseNet121
- **Architecture**: DenseNet121 with custom classification head
- **Input Size**: 224×224×3
- **Features**: Dense connections for better gradient flow
- **File**: `10_FOLD_DenseNet121.py`

### 3. ResNet152
- **Architecture**: ResNet152V2 with custom classification head
- **Input Size**: 224×224×3
- **Features**: Residual connections for deep networks
- **File**: `10_FOLD_ResNet152.py`

### 4. Ensemble Model (EDNet)
- **Architecture**: Ensemble of EfficientNetB0 and DenseNet121
- **Input Size**: 224×224×3
- **Features**: Combines predictions from two models
- **File**: `10_FOLD_EDNet.py`

### 5. Swin Transformer
- **Architecture**: Swin Transformer with hierarchical design
- **Input Size**: 224×224×3
- **Features**: Window-based self-attention mechanism
- **File**: `10_FOLD_Swin_Transformer.py`

### 6. Vision Transformer (ViT)
- **Architecture**: Vision Transformer with patch-based attention
- **Input Size**: 224×224×3
- **Features**: Pure transformer architecture for vision
- **File**: `10_FOLD_Vision_Transformer.py`

## Results

Each model generates comprehensive evaluation metrics:

### Metrics Calculated
- **Accuracy (ACC)**: Overall classification accuracy
- **Precision (PPV)**: Positive predictive value
- **Sensitivity (TPR)**: True positive rate
- **Specificity (TNR)**: True negative rate
- **F1-Score**: Harmonic mean of precision and recall
- **AUC-ROC**: Area under the ROC curve

### Output Files
- Confusion matrices (PDF format)
- Learning curves (PDF format)
- ROC curves (PDF format)
- Model weights (.h5 format)
- Aggregated metrics across all folds

## File Structure

```
D2CBTN/
├── README.md
├── requirements.txt
├── 10_FOLD_EfficientNetB0.py
├── 10_FOLD_DenseNet121.py
├── 10_FOLD_ResNet152.py
├── 10_FOLD_EDNet.py
├── 10_FOLD_Swin_Transformer.py
├── 10_FOLD_Vision_Transformer.py
├── 10_FOLD_EfficientNetB0.h5
├── 10_FOLD_DenseNet121.h5
├── 10_FOLD_ResNet152.h5
├── 10_FOLD_EDNet.h5
├── 10_FOLD_Vision_Transformer.h5
└── Dataset/
    ├── Training/
    └── Testing/
```

## ⚙️ Configuration

### Model Parameters

Each model can be configured by modifying these parameters:

```python
# Training Parameters
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 30-50 (varies by model)
LEARNING_RATE = 0.0001

# Data Augmentation
rotation_range = 10
width_shift_range = 0.1
height_shift_range = 0.1
shear_range = 0.1
zoom_range = 0.1
horizontal_flip = True

# Cross-Validation
N_FOLDS = 10
RANDOM_SEED = 33
```

### GPU Configuration

The code automatically detects and configures GPUs:

```python
# Multi-GPU support
strategy = tf.distribute.MirroredStrategy(devices=[f"/gpu:{i}" for i in range(len(gpus))])
```

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Reduce batch size
   - Use smaller image size
   - Enable memory growth

2. **Import Errors**
   - Ensure all dependencies are installed
   - Check Python version compatibility

3. **Dataset Loading Issues**
   - Verify dataset structure
   - Check file permissions
   - Ensure image formats are supported

4. **Training Convergence**
   - Adjust learning rate
   - Modify data augmentation parameters
   - Increase/decrease epochs

### Performance Optimization

1. **GPU Memory Management**
   ```python
   # Enable memory growth
   tf.config.experimental.set_memory_growth(gpu, True)
   ```

2. **Data Loading Optimization**
   ```python
   # Use prefetch for better performance
   dataset = dataset.prefetch(tf.data.AUTOTUNE)
   ```

3. **Mixed Precision Training**
   ```python
   # Enable mixed precision
   policy = tf.keras.mixed_precision.Policy('mixed_float16')
   tf.keras.mixed_precision.set_global_policy(policy)
   ```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- TensorFlow team for the deep learning framework
- Original authors of the pre-trained models
- Research community for the brain tumor datasets
- Open-source contributors for various libraries

## Support

For questions, issues, or contributions:
- Open an issue on GitHub
- Contact the maintainers
- Check the troubleshooting section

---

**Note**: This implementation is for research purposes. For clinical applications, additional validation and regulatory compliance may be required. 
