# Plant-Disease-Detection
Plant Disease Detection from Images 
🌿 Plant Disease Detection
An AI-powered Streamlit web application that allows users to upload plant leaf images and instantly detect plant diseases using a Custom CNN model.
This project is designed to assist farmers, gardeners, and researchers in quickly diagnosing plant health issues, enabling timely interventions and improved crop management

📋 Project Overview
This application uses a Convolutional Neural Network (CNN) trained on the New Plant Diseases Dataset to classify plant leaf images into one of 37 categories — covering various crops and diseases.
The tool provides:

Fast, accurate predictions from uploaded images

Easy-to-use Streamlit interface

Insights for agricultural decision-making

📂 Dataset
We used the New Plant Diseases Dataset (Augmented) from Kaggle, organized into training, validation, and test sets.
It contains images for crops like:

🍅 Tomato – Late Blight, Early Blight, Healthy

🍇 Grape – Black Rot, Healthy

🍊 Orange – Huanglongbing

🥔 Potato – Late Blight, Healthy

🌽 Corn (Maize) – Northern Leaf Blight

🍓 Strawberry – Leaf Scorch

Plus Apple, Peach, Squash, Soybean, Blueberry, Cherry, and more.

📥 Dataset Download:
Download Datase 📥 **Dataset Download**  

[New Plant Diseases Dataset (Augmented) — Kaggle](https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset)

🎯 Objectives

Develop a user-friendly image upload interface with Streamlit.

Train a Custom CNN model for plant disease classification.

Compare with at least three pretrained models to ensure superior performance.

Deliver a fully functional web app for real-world agricultural use.

🚀 Approach

1. Image Preprocessing
Resizing all images to a uniform resolution.
Normalization for consistent pixel value ranges.
Data augmentation (flip, rotation, etc.) to improve generalization.

2. Disease Classification
Custom CNN model trained on the dataset.

Performance comparison with pretrained models 

Optimized training using early stopping, learning rate scheduling, and dropout.

3. Performance Metrics

Accuracy – Overall prediction correctness.

Precision – Correct positive predictions ratio.

Recall – Ability to find all relevant cases.

F1 Score – Balanced measure for all classes

🚧 Challenges
Class imbalance – Some diseases had fewer samples, leading to bias.

Real-time inference optimization – Reducing prediction time without sacrificing accuracy.

🔮 Future Enhancements
Add a user feedback system to improve predictions over time.

Expand dataset to include more plant species and diseases.

Deploy as a mobile application for field use.

🌱 Results

Delivered a Streamlit web app for plant disease detection.

Achieved high accuracy on validation and test datasets.

Generated a detailed classification report for model evaluation.

Included a user guide to help farmers and gardeners interpret prediction

📸 Example
streamlit run app.py
(Upload a plant leaf image → Receive instant disease prediction → Take preventive action.)

🛠 Tech Stack

Python

PyTorch – Model building and training

Streamlit – Web app interface

scikit-learn – Metrics & evaluation

Pillow (PIL) – Image processing

📜 License
This project is open-source and available under the MIT License.
