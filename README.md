# NeuroScope – AI-Powered Brain Tumor Classification System

## Overview
NeuroScope is an AI-powered web application developed for brain tumor classification using MRI images and deep learning techniques. The project was developed as a graduation project for the Faculty of Computing and Data Science and aims to assist in the early detection and classification of brain tumors.

The system analyzes MRI scans and classifies them into one of four categories:
- Glioma Tumor
- Meningioma Tumor
- Pituitary Tumor
- No Tumor

NeuroScope combines multiple deep learning architectures, preprocessing pipelines, and evaluation techniques to achieve high classification accuracy while providing an interactive and user-friendly interface.

---

# Features
- MRI image upload and classification
- Deep learning-based prediction system
- Multiple model experiments and comparisons
- Interactive Streamlit web application
- Confidence score display
- MRI preprocessing pipeline
- Ensemble model experimentation
- Model evaluation and performance visualization
- SQLite database integration
- Retraining experiments using feedback data

---

# Dataset
The project uses a publicly available Brain MRI dataset obtained from Kaggle containing approximately 14,000 MRI images divided into four classes:
- Glioma Tumor
- Meningioma Tumor
- Pituitary Tumor
- No Tumor

The dataset was split into training, validation, and testing sets for model development and evaluation.

---

# Deep Learning Models Used
Several deep learning architectures were implemented and evaluated during the development process, including:

- Custom CNN
- ResNet50
- EfficientNetB3
- Vision Transformer (ViT Base)
- InceptionResNetV2

The final deployed model used in NeuroScope is:

## InceptionResNetV2
This model achieved the best overall performance and classification accuracy among all tested architectures.

---

# Preprocessing Techniques
The MRI images underwent several preprocessing techniques before training, including:
- Image resizing
- Normalization
- Data augmentation
- Noise reduction
- Class balancing

These preprocessing steps helped improve model generalization and reduce overfitting.

---

# Technologies Used
- Python
- TensorFlow / Keras
- PyTorch
- OpenCV
- Streamlit
- NumPy
- Pandas
- Scikit-learn
- SQLite
- Matplotlib

---

# Project Structure

```text
NeuroScope/
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
├── app/
│   └── mri_app.py
│
├── README.md
├── requirements.txt
└── .gitignore
```

---

# Installation

## Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/NNA-4-015.git
```

## Navigate to the project directory

```bash
cd NNA-4-015
```

## Install dependencies

```bash
pip install -r requirements.txt
```

## Run the application

```bash
streamlit run mri_app.py
```

---

# Trained Models
Due to GitHub file size limitations, trained model files are hosted externally.

Download trained models here:

[:contentReference[oaicite:0]{index=0}](https://drive.google.com/drive/folders/1tz4fksJ8X4Q1QJ637jATctXT13CZfA6I?usp=drive_link)

---

# Results
Multiple experiments were conducted to compare model performance using:
- Accuracy
- Precision
- Recall
- F1-score
- Loss curves
- Confusion matrices

The final selected model, InceptionResNetV2, achieved the best overall performance and reduced misclassification compared to the other tested architectures.

---

# Future Enhancements
Future improvements planned for NeuroScope include:
- Cloud deployment
- Explainable AI visualizations (Grad-CAM)
- Real-time clinical integration
- Tumor segmentation
- Expanded MRI datasets
- Improved model generalization
- Multilingual support
- Enhanced retraining pipelines

---

# Screenshots
<img width="2517" height="1120" alt="image" src="https://github.com/user-attachments/assets/b55b3340-85ef-4d6b-8784-cb6684f6c4aa" />
<img width="2991" height="1969" alt="image" src="https://github.com/user-attachments/assets/ba4bf4ad-6fbe-4e57-8c6e-5485ea10e7f2" />
<img width="2423" height="1012" alt="image" src="https://github.com/user-attachments/assets/bf8bfe5d-15e5-424f-853d-a5834d09e71f" />


---

# License
This project is intended for educational and research purposes only.
