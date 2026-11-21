# 💰 AI-Powered Financial Risk Analyzer

An AI-powered financial risk assessment application that predicts loan default risk using Machine Learning and Deep Learning. It provides AI-generated explanations using Gemini API and retrieves relevant financial policy information using RAG.

## 🚀 Features

- Data cleaning and preprocessing
- Exploratory Data Analysis (EDA)
- Loan default prediction
- Random Forest and PyTorch models
- Default probability estimation
- Low Risk / High Risk classification
- Django REST API for model predictions
- PostgreSQL prediction history
- RAG-based financial policy retrieval
- Gemini AI-generated risk explanations
- Interactive Streamlit dashboard
- Model probability comparison

## 🛠️ Technologies Used

- **Language:** Python
- **Frontend:** Streamlit
- **Backend:** Django, Django REST Framework
- **Machine Learning:** Scikit-learn, Random Forest
- **Deep Learning:** PyTorch
- **Data Processing:** Pandas, NumPy, SciPy
- **Database:** PostgreSQL
- **RAG:** TF-IDF, Cosine Similarity
- **Generative AI:** Google Gemini API
- **Visualization:** Matplotlib, Seaborn
- **Other Tools:** Joblib, Git, Jupyter Notebook

## 🤖 Machine Learning Models

The project uses:

- Logistic Regression
- Random Forest Classifier
- PyTorch Neural Network

**Random Forest Performance:**
- Accuracy: 93.55%
- ROC-AUC: 0.9331

Random Forest is the primary prediction model, while PyTorch provides a secondary prediction for comparison.

## 🧠 AI-Powered Explanation

- Retrieves relevant financial policy sections using TF-IDF and cosine similarity.
- Uses Google Gemini API to generate simple explanations.
- Compares predictions from Random Forest and PyTorch.
- Displays retrieved policy information alongside the explanation.

## 📊 Application Output

- Risk classification
- Estimated default probability
- Comparison of model predictions
- AI-generated risk explanation
- Retrieved financial policy sections
- Prediction history

## ⚠️ Disclaimer

This application provides educational machine-learning estimates only. It is not financial advice and should not be used as the sole basis for lending or financial decisions.