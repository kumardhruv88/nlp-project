📊 Sentiment Analysis on Amazon Alexa Reviews

An end-to-end NLP mini project that performs sentiment analysis on Amazon Alexa product reviews using classical machine learning techniques.
The project covers data collection, exploratory data analysis (EDA), feature engineering, model comparison, hyperparameter tuning, and deployment with Flask on Render.

🔗 Live Demo: (https://nlp-project-cyng.onrender.com/)

🚀 Project Overview

Customer reviews contain valuable insights into user satisfaction and product quality.
This project builds a binary sentiment classifier to predict whether a review expresses positive or negative sentiment.

Key Highlights

Real-world dataset from Kaggle
Strong focus on EDA and visualization
Comparison of multiple ML models
Best model selection using GridSearchCV
Deployed as a Flask web application
Hosted on Render

🗂 Dataset

Source: Kaggle – Amazon Alexa Reviews Dataset
Format: .tsv
Features include: 
Review text
Rating
Feedback label

Data Preprocessing

Text cleaning & normalization
Stopword removal
Feature extraction using vectorization
Handling missing values

🛠 Tech Stack
🔹 Programming Language:python
🔹 Data Analysis & Visualization :Numpy,Pandas,Seaborn,Matplotlib
🔹 NLP & Machine Learning:Scikit-learn ,XGBoost,Random Forest,Decision Tree
🔹 Model Optimization: GridSearchCV (Hyperparameter tuning)
🔹 Experimentation :Jupyter Notebook (EDA & model experimentation)
🔹 Web Development:Flask (Backend),HTML & CSS (Frontend)
🔹 Deployment:Render,Gunicorn (WSGI server)

📈 Exploratory Data Analysis (EDA)
EDA was performed to understand:
Distribution of sentiments

Relationship between ratings and sentiment
Review length patterns
Feature importance indicators
Visualizations helped in:
Identifying class imbalance
Choosing suitable algorithms
Improving model performance

🤖 Models Implemented

The following machine learning models were trained and evaluated:

Model	Description
Decision Tree	Baseline classifier
Random Forest	Ensemble-based approach
XGBoost	Gradient boosting model
✅ Best Performing Model

XGBoost Achieved the highest accuracy..Tuned using GridSearchCV for optimal hyperparameters

⚙️ Hyperparameter Tuning

Used GridSearchCV,Tuned parameters such as:
Number of estimators
Max depth
Learning rate
Improved generalization and reduced overfitting

🌐 Web Application
A simple and clean web interface where users can:Enter a product review..Get real-time sentiment prediction (Positive / Negative)
Tech Used
Flask for routing and inference
HTML & CSS for UI
Pre-trained XGBoost model loaded using .pkl

🧩 Project Structure
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

🚀 Deployment
The application is deployed on Render.Uses Gunicorn as the production WSGI server.Automatically redeploys on GitHub push

📊 Results & Insights

XGBoost outperformed other models
Hyperparameter tuning significantly improved accuracy
The deployed model provides fast and reliable predictions

🎯 Learning Outcomes

End-to-end NLP project implementation
Hands-on EDA for text data
Model comparison and selection
Hyperparameter tuning using GridSearchCV
Deploying ML models with Flask and Render
Writing production-ready ML code

🔮 Future Improvements
Use deep learning models (LSTM / Transformers)
Add confidence score to predictions
Deploy using Docker
Improve UI/UX
Extend to multi-class sentiment analysis
