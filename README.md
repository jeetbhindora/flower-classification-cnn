# Flower Species Classification — CNN & Transfer Learning

## Overview
A deep learning project that classifies flower species using two approaches:
1. A custom Convolutional Neural Network (CNN) built from scratch to classify 5 flower species from the TF Flowers dataset
2. Transfer Learning using MobileNetV2 (pretrained on ImageNet) to classify 102 flower species from the Oxford Flowers 102 dataset

## Tech Stack
- Python
- TensorFlow / Keras
- Scikit-learn
- NumPy
- Matplotlib
- Google Colab

## Features
- Custom CNN architecture with 3 Conv2D + MaxPooling layers
- Transfer Learning with MobileNetV2 — two-phase fine-tuning
- Data augmentation (random flip, rotation, zoom)
- Image preprocessing and pipeline optimization
- Model evaluation using Scikit-learn classification reports
- Training accuracy and loss curve visualization

## How to Run
1. Open the `.ipynb` file in Google Colab or Jupyter Notebook
2. Install dependencies:
pip install tensorflow scikit-learn matplotlib numpy
3. Run all cells in order

## Results
- CNN trained on TF Flowers dataset (5 classes)
- MobileNetV2 fine-tuned on Oxford Flowers 102 dataset (102 classes)
- Classification report generated using Scikit-learn
- Training history plots (accuracy and loss) visualized

## Project Structure
flower-classification-cnn/
│
├── flower_classification_cnn.ipynb       # CNN from scratch
├── flower_classification_transfer.ipynb  # MobileNetV2 Transfer Learning
└── README.md

## Author
**Jeet Bhindora**
- GitHub: https://github.com/jeetbhindora
- Email: jeetbhindora.jb@gmail.com
