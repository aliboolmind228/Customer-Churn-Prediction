# 📉 Customer Churn Prediction Dashboard

An **end-to-end machine learning project** to predict telecom customer churn using **Python** and **Streamlit**.

This project includes:
- Data preprocessing & cleaning
- Model training (Random Forest & Logistic Regression)
- Hyperparameter tuning
- Model saving & loading
- Interactive dashboard for churn prediction

---

## 📂 Project Structure
```plaintext
customer-churn-prediction-dashboard/
│
├── Dataset/
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv
│
├── Notebook/
│   └── churn_model_training.ipynb
│
├── App/
│   ├── app.py
│   └── churn_model.pkl
│
│── LICENSE
│
├── requirements.txt
│
└── README.md
```
## 🚀 How to Run the Project

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/customer-churn-prediction-dashboard.git
cd customer-churn-prediction-dashboard
```
### 2️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```
### 3️⃣ Run the Streamlit App
```bash
cd app
streamlit run app.py
```
## 📊 Dashboard Features

- Churn Prediction: Predict if a customer will churn based on input features.
- Visual Analysis: Distribution plots for churn, monthly charges, contract type, and more.
- Interactive Controls: Filter and explore churn patterns.
- Model Integration: Predicts using a tuned Random Forest model
  
## 🛠 Tech Stack

- Python
- Pandas,NumPy
- Matplotlib, Seaborn
- Scikit-learn
- Streamlit









