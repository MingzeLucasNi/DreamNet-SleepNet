# Dreaming Is All You Need

In classification tasks, achieving a harmonious balance between exploration and precision is of paramount importance. To this end, this research introduces two novel deep learning models, SleepNet and DreamNet, to strike this balance. SleepNet seamlessly integrates supervised learning with unsupervised "sleep" stages using pre-trained encoder models. Dedicated neurons within SleepNet are embedded in these unsupervised features, forming intermittent "sleep" blocks that facilitate exploratory learning. Building upon the foundation of SleepNet, DreamNet employs full encoder-decoder frameworks to reconstruct the hidden states, mimicking the human "dreaming" process. This reconstruction process enables further exploration and refinement of the learned representations. Moreover, the principle ideas of our SleepNet and DreamNet are generic and can be applied to both computer vision and natural language processing downstream tasks. Through extensive empirical evaluations on diverse image and text datasets, SleepNet and DreanNet have demonstrated superior performance compared to state-of-the-art models, showcasing the strengths of unsupervised exploration and supervised precision afforded by our innovative approaches.

## Table of Contents

- [Features](#features)
- [Requirements](#requirements)
- [Usage](#usage)

## Features

- **Vision Transformer (ViT) Integration**: Enhances feature extraction capabilities by leveraging the Vision Transformer (ViT) model, providing state-of-the-art performance in image classification.
- **ResNet18 Architecture**: Utilizes the proven ResNet18 architecture to achieve high accuracy in image classification tasks.
- **Advanced Data Processing**: Employs data augmentation and normalization techniques to improve generalization and robustness across diverse datasets.
- **Efficient GPU Memory Management**: Periodically clears GPU memory to optimize resource utilization, especially during extended training sessions.
- **Adaptive Learning Rate Scheduling**: Includes a dynamic learning rate scheduler to facilitate smoother and more effective model convergence.

## Requirements

- Python 3.8 or later
- PyTorch
- torchvision
- Hugging Face's `transformers` library
- CUDA-compatible GPU (recommended for faster training)

## Usage

1. **Setup & Installation**:

   Install all necessary dependencies with:
   ```bash
   pip install -r requirements.txt
   ```

2. **Training the Text Classifier**:

   To train the SleepNet model with default parameters:
   ```bash
   python trainer.py
   ```

   For customized training settings, use:
   ```bash
   python trainer.py --dataset 'ag_news' --epochs 50 --num_classes 10
   ```

3. **Training the Vision Classifier**:

   To train the SleepNet model for image classification:
   ```bash
   python train_vision.py --dataset 'cifar100' --epochs 50 --num_classes 10
   ```
