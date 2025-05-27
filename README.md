# Fake news detection using machine learning and deep learning
# Overview

This project presents a fake news classification system that integrates traditional machine learning, deep learning, and transformer-based models. It also incorporates SHAP-based explainability and a real-time ensemble voting mechanism to enhance prediction transparency and user trust.
Key Features

Comparison of 8 models: LR, RF, SVM, NB, LSTM, BiLSTM, CNN, BERT

Hyperparameter tuning using Bayesian Optimization

Majority voting ensemble system with Real / Caution / Fake outcomes

SHAP-based interpretability for decision explanation

Real-time prediction with conditional visual explanation

Designed for trust-building and user transparency

# File Overview

LR, RFC, SVM, Navie bayes.py: Implements classical models (Logistic Regression, Random Forest, SVM, Naive Bayes) using TF-IDF.

LSTM_Bilstm_cnn_model.py: mplements deep learning models (LSTM, BiLSTM, CNN) using Keras/TensorFlow.

BERT.py: Fine-tuned BERT model for binary classification of fake vs real news.

real_time_system.py: real-time predictions using all models + SHAP explanations.

# Dataset

ISOT Fake News Dataset

Link: https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset

Required Libraries

numpy

pandas

scikit-learn

matplotlib

seaborn

torch

tensorflow

transformers

shap

joblib

scikit-optimize

# How to Run the Code

Step 1. Clone the Repository

Step 2. Prepare the Dataset

Place True.csv and Fake.csv inside the data/ folder.

Step 3. Run Traditional or Deep Learning Models

LR, RFC, SVM, Navie bayes.py

LSTM_Bilstm_cnn_model.py

BERT.py

Step 5. Run Real-Time System

real_time_system.py

For deep learning models (LSTM, BiLSTM, CNN), after performing Bayesian Optimization, you must manually apply the best hyperparameter values (e.g., learning rate, batch size, number of epochs) into the 
corresponding model definitions in real_time_system.py before running the real-time ensemble system.

These optimized parameters are not loaded automatically and must be transferred manually to ensure consistent performance.
