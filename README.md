# Plant Disease Detection System using Deep Learning

## Overview

The Plant Disease Detection System is a Deep Learning-based application developed using Python, PyTorch, CNN, and Streamlit. The system detects plant leaf diseases from uploaded images and predicts the disease category with confidence score.

This project helps farmers, researchers, and agricultural industries identify plant diseases quickly and accurately using Artificial Intelligence.

---

# Features

* Plant leaf disease classification
* Deep Learning-based CNN model
* Real-time prediction
* Streamlit web application
* Confidence score display
* Automatic image preprocessing
* Multi-class classification
* User-friendly interface

---

# Technologies Used

## Programming Language

* Python

## Libraries & Frameworks

* PyTorch
* Torchvision
* NumPy
* Scikit-learn
* PIL
* Streamlit
* tqdm

---

# Dataset

Dataset used:

New Plant Diseases Dataset (Augmented)

The dataset contains:

* Multiple plant species
* Healthy and diseased leaf images
* Augmented training images

Dataset structure:

```text
train/
    Tomato___Healthy/
    Tomato___Early_blight/
    Potato___Late_blight/
    ...
```

Each folder name represents a class label.

---

# Project Workflow

```text
Image Dataset
      ↓
Image Preprocessing
      ↓
Data Augmentation
      ↓
CNN Model Training
      ↓
Model Evaluation
      ↓
Save Best Model
      ↓
Streamlit Deployment
      ↓
Disease Prediction
```

---

# Image Preprocessing

The following preprocessing techniques are used:

* Resize images to 128×128
* Random horizontal flip
* Tensor conversion
* Image normalization

These techniques improve model performance and generalization.

---

# CNN Architecture

The model uses a custom Convolutional Neural Network (CNN) consisting of:

* Convolution layers
* ReLU activation
* MaxPooling layers
* Adaptive Average Pooling
* Fully Connected layers
* Dropout regularization

---

# Model Training

## Training Parameters

| Parameter     | Value            |
| ------------- | ---------------- |
| Image Size    | 128×128          |
| Batch Size    | 32               |
| Epochs        | 15               |
| Learning Rate | 0.001            |
| Optimizer     | Adam             |
| Loss Function | CrossEntropyLoss |

---

# Evaluation Metrics

The model performance is evaluated using:

* Accuracy
* Precision
* Recall
* F1-Score

These metrics help measure the effectiveness of the classification model.

---

# Streamlit Application

The Streamlit web application allows users to:

* Upload plant leaf images
* Detect diseases instantly
* View confidence score
* Display prediction results interactively

---

# Installation

## Step 1: Clone Project

```bash
git clone <repository_link>
```

---

## Step 2: Install Dependencies

```bash
pip install torch torchvision streamlit numpy pillow scikit-learn tqdm
```

---

## Step 3: Run Training File

```bash
python train.py
```

---

## Step 4: Run Streamlit Application

```bash
streamlit run mbapp.py
```

---

# Project Structure

```text
plant_disease/
│
├── data/
├── mbapp.py
├── train.py
├── cnn_cplant_classifiers.pth
├── plant_classes.npy
├── README.md
```

---

# Output

The system predicts:

* Plant disease name
* Confidence score

Example:

```text
Prediction:
Tomato Early Blight

Confidence:
95.34%
```

---

# Advantages

* Fast disease detection
* Reduces manual inspection
* Helps farmers take early action
* AI-based automated solution
* Easy-to-use interface

---

# Future Enhancements

* Mobile application integration
* Real-time camera detection
* More plant species support
* Cloud deployment
* Treatment recommendation system

---

# Conclusion

The Plant Disease Detection System successfully classifies plant diseases using Deep Learning techniques. The project demonstrates the practical application of Artificial Intelligence in agriculture and provides an efficient, accurate, and user-friendly disease prediction solution.

---

# Author

Komal

Artificial Intelligence & Machine Learning Project

---
