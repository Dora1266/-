# Video Anomaly Detection System

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.7+-blue.svg" alt="Python">
  <img src="https://img.shields.io/badge/PyTorch-1.7+-orange.svg" alt="PyTorch">
  <img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License">
  <img src="https://img.shields.io/badge/Stars-7-yellow.svg" alt="Stars">
</p>

<p align="center">
  <b>Deep Learning-based Video Anomaly Detection with 3D CNN + Bidirectional LSTM</b>
</p>

---

## 🎯 Overview

This project implements an **end-to-end unsupervised video anomaly detection system** designed for:

- 🏢 **Security & Surveillance** - Detect suspicious activities in CCTV footage
- 🏭 **Industrial Safety** - Identify unsafe behaviors in workplaces
- 🏥 **Healthcare Monitoring** - Recognize unusual patient activities
- 🛍️ **Retail Analytics** - Analyze customer behavior patterns

### ✨ Key Features

| Feature | Description |
|---------|-------------|
| 🧠 **Unsupervised Learning** | Train only on normal behavior - no anomaly annotations needed |
| 🎬 **3D CNN (R3D-18)** | Extract spatiotemporal features from video clips |
| 🔄 **Bidirectional LSTM** | Capture long-term temporal dependencies |
| 🎯 **Attention Mechanism** | Focus on critical moments for detection |
| 📊 **Auto Threshold** | Automatic optimal threshold selection based on F1-score |
| 📈 **Visualization** | Anomaly score curves with highlighted abnormal segments |

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Input Video                         │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│           Data Preprocessing                             │
│  • Frame Extraction  • Resize  • Normalization         │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│              3D CNN Feature Extractor                  │
│                  (R3D-18 Pre-trained)                    │
│     Spatiotemporal Feature: [batch, seq, 512]          │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│           Bidirectional LSTM Encoder                   │
│              Capture Long-term Dependencies              │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│              Attention Mechanism                       │
│         Focus on Critical Temporal Features             │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│           MLP Classifier + Sigmoid                       │
│              Anomaly Score [0, 1]                       │
└─────────────────────────────────────────────────────────┘
```

---

## 🚀 Quick Start

### Prerequisites

- Python 3.7+
- CUDA-capable GPU (recommended)
- 8GB+ RAM

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/Dora1266/-.git
cd -

# 2. Create virtual environment (recommended)
conda create -n anomaly python=3.8
conda activate anomaly

# 3. Install dependencies
pip install torch torchvision opencv-python numpy scikit-learn matplotlib tqdm
```

### Dataset Preparation

Organize your dataset as follows:

```
dataset/
├── train/
│   └── normal/              # Training: normal videos only
│       ├── video1.mp4
│       └── ...
└── test/
    ├── normal/              # Testing: normal videos
    │   └── ...
    └── abnormal/            # Testing: abnormal videos
        └── ...
```

### Training

```python
# Edit Config in anomaly_detection.py, then run:
python anomaly_detection.py
```

### Inference Example

```python
import torch
from anomaly_detection import AnomalyDetector, visualize_anomalies

# Load model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = AnomalyDetector().to(device)
checkpoint = torch.load('saved_models/best_model.pth')
model.load_state_dict(checkpoint['model_state_dict'])
threshold = checkpoint['threshold']

# Detect anomalies
scores, time_axis = visualize_anomalies(
    model, 
    'path/to/video.mp4', 
    threshold, 
    device
)
```

---

## ⚙️ Configuration

All hyperparameters can be adjusted in the `Config` class:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `CLIP_LENGTH` | 16 | Frames per video clip |
| `FRAME_SIZE` | 112 | Frame resize dimension |
| `BATCH_SIZE` | 8 | Mini-batch size |
| `HIDDEN_SIZE` | 512 | LSTM hidden units |
| `NUM_LAYERS` | 2 | LSTM layers |
| `DROPOUT` | 0.5 | Dropout rate |
| `LEARNING_RATE` | 1e-4 | Initial learning rate |
| `WEIGHT_DECAY` | 1e-5 | L2 regularization |
| `EPOCHS` | 50 | Training epochs |

---

## 📊 Performance Metrics

The system automatically evaluates using:

- **AUC (Area Under Curve)** - Overall discrimination ability
- **F1-Score** - Precision-Recall balance at optimal threshold
- **Threshold** - Auto-selected for maximum F1-score

---

## 🗺️ Roadmap

- [ ] Real-time processing optimization
- [ ] Multi-GPU training support
- [ ] ONNX export for deployment
- [ ] Web-based demo interface
- [ ] Pre-trained model zoo
- [ ] Multi-modal fusion (RGB + Optical Flow)

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- [PyTorch](https://pytorch.org/) - Deep learning framework
- [OpenCV](https://opencv.org/) - Computer vision library
- [torchvision](https://pytorch.org/vision/stable/index.html) - Pre-trained models

---

<p align="center">
  <b>⭐ Star this repository if you find it helpful! ⭐</b>
</p>

<p align="center">
  Made with ❤️ by <a href="https://github.com/Dora1266">Dora1266</a>
</p>
