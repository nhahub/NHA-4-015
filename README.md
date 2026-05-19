# NeuroScope – Brain Tumor Classification System

## Overview
NeuroScope is an AI-powered brain tumor classification system developed as a graduation project for the Faculty of Computing and Data Science. The project aims to assist in the early detection and classification of brain tumors from MRI scans using deep learning techniques.

The system analyzes MRI images and classifies them into four categories:
- Glioma Tumor
- Meningioma Tumor
- Pituitary Tumor
- No Tumor

NeuroScope was designed to provide fast, accurate, and intelligent predictions through an interactive and user-friendly interface.

---

## Features
- MRI image upload and prediction
- Deep learning-based tumor classification
- Multiple model experiments and comparisons
- Interactive Streamlit web application
- Confidence score display
- SQLite database integration
- Model evaluation and performance analysis
- Image preprocessing and augmentation pipeline

---

## Dataset
The project uses a publicly available Brain MRI dataset obtained from Kaggle containing approximately 14,000 MRI images divided into four classes:
- Glioma Tumor
- Meningioma Tumor
- Pituitary Tumor
- No Tumor

The dataset was split into training, validation, and testing sets for model development and evaluation.

---

## Deep Learning Models Used
Several deep learning architectures were implemented and evaluated during the development process, including:

- Custom CNN
- ResNet50
- EfficientNetB3
- Vision Transformer (ViT Base)
- InceptionResNetV2

The final deployed model used in NeuroScope is:
### InceptionResNetV2
This model achieved the best overall performance and classification accuracy among the tested architectures.

---

## Preprocessing Techniques
The MRI images underwent several preprocessing steps before training, including:
- Image resizing
- Normalization
- Data augmentation
- Noise reduction
- Class balancing techniques

These preprocessing techniques helped improve model generalization and reduce overfitting.

---

## Technologies Used
- Python
- TensorFlow / Keras
- OpenCV
- Streamlit
- NumPy
- Pandas
- Scikit-learn
- SQLite
- Matplotlib

---

## Project Structure

```text
NeuroScope/
│
├── app/
│   └── mri_app.py
│
├── models/
│
├── notebooks/
│   ├── preprocessing.ipynb
│   ├── originalmodel.ipynb
│   ├── efficientNetb3.ipynb
│   ├── vitbase.ipynb
│   ├── inceptionresnetv2.ipynb
│   ├── final_ensemble.ipynb
│   └── model_evaluation.ipynb
│
├── database/
│
├── screenshots/
│
├── requirements.txt
├── README.md
└── .gitignore
