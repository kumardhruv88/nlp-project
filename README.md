# 📊 Sentiment Analysis on Amazon Alexa Reviews

An **end-to-end NLP mini project** that performs **sentiment analysis** on Amazon Alexa product reviews using classical machine learning techniques.  
The project covers **data collection, exploratory data analysis (EDA), feature engineering, model comparison, hyperparameter tuning**, and **deployment using Flask on Render**.

🔗 **Live Demo**: *(https://nlp-project-1a6s.onrender.com)*

---

## 🚀 Project Overview

Customer reviews contain valuable insights into user satisfaction and product quality.  
This project builds a **binary sentiment classification system** to predict whether a review expresses **positive or negative sentiment**.

### Key Highlights
- Real-world dataset sourced from Kaggle
- Strong emphasis on **EDA and data visualization**
- Multiple ML models implemented and compared
- Best model selected using **GridSearchCV**
- Simple **Flask-based web UI**
- Successfully deployed on **Render**

---

## 🗂 Dataset

- **Source**: Kaggle – Amazon Alexa Reviews Dataset  
- **Format**: `.tsv`
- **Data includes**:
  - Review text
  - Ratings
  - Feedback labels

### Data Preprocessing
- Text cleaning and normalization
- Stopword removal
- Feature extraction using vectorization
- Handling missing values

---

## 🛠 Tech Stack

### Programming Language
- Python

### Data Analysis & Visualization
- Numpy
- Pandas
- Matplotlib
- Seaborn

### NLP & Machine Learning
- Scikit-learn
- XGBoost
- Random Forest
- Decision Tree

### Model Optimization
- GridSearchCV (Hyperparameter tuning)

### Development & Experimentation
- Jupyter Notebook

### Web Development
- Flask (Backend)
- HTML & CSS (Frontend)

### Deployment
- Render
- Gunicorn

---

## 📈 Exploratory Data Analysis (EDA)

EDA was performed to:
- Analyze sentiment distribution
- Study rating vs sentiment relationship
- Understand review length patterns
- Identify important features influencing sentiment

Visualizations helped in:
- Detecting class imbalance
- Improving feature selection
- Choosing appropriate models

---

## 🤖 Models Implemented

The following machine learning models were trained and evaluated:

| Model | Description |
|------|------------|
| Decision Tree | Baseline classifier |
| Random Forest | Ensemble learning approach |
| XGBoost | Gradient boosting model |

### ✅ Best Performing Model
- **XGBoost**
- Achieved the **highest accuracy**
- Tuned using **GridSearchCV** to obtain optimal hyperparameters

---

## ⚙️ Hyperparameter Tuning

- Used **GridSearchCV**
- Tuned parameters such as:
  - Number of estimators
  - Maximum depth
  - Learning rate
- Improved generalization and reduced overfitting

---

## 🌐 Web Application

A simple and user-friendly web interface where users can:
- Enter a product review
- Get instant sentiment prediction (**Positive / Negative**)

### Tech Used
- Flask for backend routing and inference
- HTML & CSS for frontend
- Trained XGBoost model loaded using `.pkl` files

---

## 🧩 Project Structure

```bash
sentiment-analysis/
│
├── Data/
│   └── amazon_alexa.tsv
│
├── Models/
│   ├── model_xgb.pkl
│   ├── model_rf.pkl
│   └── vectorizer.pkl
│
├── templates/
│   ├── index.html
│   └── landing.html
│
├── static/
│   └── css/
│       └── style.css
│
├── app.py
├── requirements.txt
├── Procfile
├── runtime.txt
└── README.md
