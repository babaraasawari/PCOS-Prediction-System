<div align="center">

# 🩺 PCOS Prediction System

### Hybrid Machine Learning-Based PCOS Prediction with OCR-Powered Medical Report Analysis

<p>
  <img src="https://img.shields.io/badge/Python-3.x-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/Scikit--learn-ML-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white"/>
  <img src="https://img.shields.io/badge/XGBoost-ML-006400?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/OCR-Pytesseract-8A2BE2?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/GUI-CustomTkinter-2575FC?style=for-the-badge"/>
</p>

<p>
  <img src="https://img.shields.io/badge/Model-Random%20Forest%20%2B%20XGBoost-00C853?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Accuracy-~92%25-FF6B35?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Domain-Healthcare-A020F0?style=for-the-badge"/>
</p>

<br>

<img src="https://readme-typing-svg.demolab.com?font=Fira+Code&weight=600&size=20&duration=3000&pause=1000&color=8A2BE2&center=true&vCenter=true&width=800&lines=Hybrid+Machine+Learning+%F0%9F%A4%96;Random+Forest+%2B+XGBoost;OCR-Powered+Medical+Report+Analysis+%F0%9F%93%84;Data-Driven+PCOS+Prediction+%F0%9F%93%8A" alt="Typing SVG"/>

</div>

---

## 📌 Overview

The **PCOS Prediction System** is a machine learning-based application designed to predict the likelihood of **Polycystic Ovary Syndrome (PCOS)** using patient-related data.

The system combines **Random Forest and XGBoost** through a hybrid machine learning approach and integrates **OCR-based medical report analysis** to extract relevant information from uploaded reports.

The application provides a user-friendly graphical interface where users can enter patient parameters, upload medical reports, obtain a prediction probability, and visualize the prediction result.

> **Project Goal:** Develop an accessible, data-driven system that can assist with early risk assessment and demonstrate the application of machine learning and OCR in healthcare.

---

## ✨ Key Features

| Feature | Description |
|---|---|
| 🤖 **Hybrid ML Model** | Combines Random Forest and XGBoost |
| 📄 **OCR Analysis** | Extracts information from uploaded medical reports |
| 🔍 **Automatic Extraction** | Processes extracted report information for prediction |
| 🖥️ **Graphical Interface** | User-friendly CustomTkinter-based application |
| 📊 **Probability Prediction** | Displays the predicted PCOS probability |
| 📈 **Visualization** | Presents prediction results through graphs |
| 🧬 **Feature Engineering** | Uses relevant clinical and lifestyle parameters |
| 📋 **Report-Based Input** | Supports medical report information as an input source |

---

# 🧠 Machine Learning Approach

The system uses a **hybrid ensemble approach** combining two machine learning algorithms:

### 🌲 Random Forest

Random Forest combines multiple decision trees to improve classification performance and robustness.

### ⚡ XGBoost

XGBoost is a gradient boosting algorithm that builds sequential decision trees to improve predictive performance.

### 🔗 Hybrid Approach

The project combines the predictions of:

```text
                ┌──────────────────┐
                │   Patient Data   │
                └────────┬─────────┘
                         │
                         ▼
              ┌─────────────────────┐
              │ Data Preprocessing  │
              └──────────┬──────────┘
                         │
                         ▼
              ┌─────────────────────┐
              │ Feature Engineering│
              └──────────┬──────────┘
                         │
              ┌──────────┴──────────┐
              ▼                     ▼
      ┌───────────────┐     ┌───────────────┐
      │ Random Forest │     │    XGBoost    │
      └───────┬───────┘     └───────┬───────┘
              │                     │
              └──────────┬──────────┘
                         ▼
                ┌─────────────────┐
                │ Hybrid Model    │
                │   Prediction    │
                └────────┬────────┘
                         │
                         ▼
                ┌─────────────────┐
                │ Probability +   │
                │ Prediction      │
                └─────────────────┘
