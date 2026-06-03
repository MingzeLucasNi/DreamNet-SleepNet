# SleepNet and DreamNet: Enriching and Reconstructing Representations for Consolidated Visual Classification

An effective integration of rich feature representations with robust classification mechanisms remains a key challenge in visual understanding tasks. This study introduces two novel deep learning models, SleepNet and DreamNet, which are designed to improve representation utilization through feature enrichment and reconstruction strategies. SleepNet integrates supervised learning with representations obtained from pre-trained encoders, leading to stronger and more robust feature learning. Building on this foundation, DreamNet incorporates pre-trained encoder–decoder frameworks to reconstruct hidden states, allowing deeper consolidation and refinement of visual representations. Our experiments show that our models consistently achieve superior performance compared with existing state-of-the-art methods, demonstrating the effectiveness of the proposed enrichment and reconstruction approaches.

---

## Table of Contents

- [Features](#features)
- [Requirements](#requirements)
- [Usage](#usage)
- [Anonymous Review](#anonymous-review)
- [License](#license)

---

## Features

- **Vision Transformer (ViT) Integration**  
  Incorporates the ViT architecture to capture global contextual features efficiently for image classification tasks.

- **ResNet18 Backbone**  
  Utilizes ResNet18 for robust hierarchical feature extraction and strong baseline performance.

- **Advanced Data Processing**  
  Includes flexible data augmentation and normalization pipelines to enhance generalization and robustness across visual datasets.

- **Efficient GPU Memory Management**  
  Periodically clears GPU cache to optimize resource utilization during long training runs.

- **Adaptive Learning Rate Scheduling**  
  Employs a dynamic scheduler for smooth and stable convergence.

---

## Requirements

- Python 3.8 or later  
- PyTorch  
- torchvision  
- Hugging Face’s `transformers` library (for Vision Transformer support)  
- CUDA-compatible GPU (recommended)

---

## Usage

### 1. Setup & Installation

Install all dependencies:

```bash
pip install -r requirements.txt
```

### 2. Training the Vision Classifier

Train the **SleepNet** model for image classification with default parameters:

```bash
python train_vision.py
```

For customized settings (e.g., specific dataset, epochs, number of classes):

```bash
python train_vision.py --dataset 'cifar100' --epochs 50 --num_classes 100
```

## Anonymous Review

This repository is prepared for double-blind review. Author names, affiliations, and citation metadata are intentionally omitted during the review period.

An anonymized repository link can be added here:

```text
https://anonymous.4open.science/r/DreamNet-SleepNet-3EC3/
```

---

## License

This project is released under the [MIT License](LICENSE).
