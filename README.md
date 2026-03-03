# SSL-ViT-Deepfake-Detection
Self-Supervised Vision Transformer-based Deepfake Detection with Robustness Evaluation and Streamlit Deployment.
Overview

This repository presents an end-to-end deep learning system for deepfake image detection using contrastive self-supervised learning and Vision Transformers.

The model first learns robust visual representations using contrastive self-supervised learning and is then fine-tuned for binary classification (Real vs Fake). The system includes robustness evaluation under Gaussian noise and an interactive Streamlit deployment for real-time inference.

This project demonstrates a complete machine learning lifecycle from data exploration to deployment.

Key Features

Vision Transformer (ViT Tiny) backbone

Self-Supervised Contrastive Pretraining (NT-Xent Loss)

Supervised Fine-Tuning for Binary Classification

Clean Validation Metrics

Robustness Testing under Gaussian Noise

Streamlit-based Deployment Interface

Modular Notebook-based Architecture

System Architecture
Input Image
    ↓
Preprocessing (Resize 224×224 + Normalize)
    ↓
Vision Transformer Encoder (SSL Pretrained)
    ↓
Linear Classification Head
    ↓
Sigmoid Activation
    ↓
Prediction (Real / Fake) + Confidence Score
Model Configuration

Backbone: vit_tiny_patch16_224

Embedding Dimension: 192

Transformer Blocks: 12

Activation Function: GELU

SSL Loss: NT-Xent Contrastive Loss

Fine-Tuning Loss: Binary Cross Entropy

Model Performance
Evaluation Setting	Accuracy	F1 Score	ROC-AUC
Clean Validation	0.738	0.711	0.828
Gaussian Noise	0.703	0.645	0.800
Observations

Strong separability on clean validation data (AUC = 0.828).

Limited degradation under Gaussian perturbations.

Self-supervised pretraining improves robustness of learned representations.

Transformer-based embeddings remain stable under noisy inputs.

Repository Structure
SSL-ViT-Deepfake-Detection/
│
├── 1_Overview_and_EDA.ipynb
├── 2_SSL_Pretraining.ipynb
├── 3_Fine_Tuning.ipynb
├── 4_Evaluation_and_Robustness.ipynb
│
├── app.py
├── ssl_encoder.pth
├── classifier.pth
│
└── README.md
Development Workflow
1. Data Understanding

Dataset inspection

Class distribution analysis

Domain validation

2. Self-Supervised Pretraining

Dual augmentations

Contrastive learning objective

Vision Transformer encoder training

Encoder saved as ssl_encoder.pth

3. Supervised Fine-Tuning

Load pretrained encoder

Add binary classification head

Optimize using Binary Cross Entropy

Model saved as classifier.pth

4. Evaluation and Robustness

Clean validation evaluation

Gaussian noise perturbation testing

Final performance comparison

5. Deployment

Streamlit-based interactive interface

Real-time image inference

Confidence score visualization

Running the Project Locally
Create Environment
conda create -n deepfake python=3.9
conda activate deepfake
pip install torch torchvision timm streamlit scikit-learn albumentations opencv-python pillow
Run Streamlit Application
python -m streamlit run app.py
Technology Stack

Python

PyTorch

timm (Vision Transformers)

Albumentations

scikit-learn

Streamlit

OpenCV

NumPy

What This Project Demonstrates

Representation learning with self-supervised learning

Transformer-based image modeling

End-to-end ML pipeline design

Robustness evaluation methodology

Deployment-ready inference system

Clean modular project structuring
