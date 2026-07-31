# 🏠 House Price Prediction using Simple Linear Regression

> A complete end-to-end Machine Learning project that predicts house prices based on house area using **Simple Linear Regression**. This project demonstrates the complete ML workflow—from business understanding and data analysis to model training, evaluation, persistence, and prediction.

---

## 📌 Project Overview

House price estimation is one of the most common regression problems in Machine Learning.

In this project, we build a Simple Linear Regression model that learns the relationship between **House Area (sq.ft.)** and **House Price**.

The project follows a structured, industry-style workflow instead of simply training a model, making it suitable for learning, interviews, and GitHub portfolios.

---

## 🎯 Business Problem

Real estate companies often need to estimate house prices quickly and consistently.

Instead of manually estimating prices, we train a Machine Learning model that predicts the selling price of a house based on its area.

---

## 🎯 Objectives

* Understand the Machine Learning workflow.
* Perform professional Exploratory Data Analysis (EDA).
* Build a Simple Linear Regression model.
* Evaluate model performance using regression metrics.
* Save the trained model for future use.
* Predict house prices without retraining the model.

---

# 📂 Project Structure

```text
House_Price_Prediction/
│
├── data/
│   └── house_price.csv
│
├── notebooks/
│   └── analysis.ipynb
│
├── outputs/
│   └── actual_vs_predicted.csv
│
├── src/
│   ├── train.py
│   ├── predict.py
│   └── utils.py
│
├── model.pkl
├── requirements.txt
└── README.md
```

---

# 🚀 Machine Learning Workflow

```text
Business Understanding
        │
        ▼
Dataset Collection
        │
        ▼
Data Loading
        │
        ▼
Exploratory Data Analysis (EDA)
        │
        ▼
Feature & Target Selection
        │
        ▼
Train-Test Split
        │
        ▼
Model Training
        │
        ▼
Prediction
        │
        ▼
Model Evaluation
        │
        ▼
Save Model
        │
        ▼
Production Prediction
```

---

# 📊 Dataset

The dataset contains:

| Feature | Description              |
| ------- | ------------------------ |
| Area    | House Area (Square Feet) |
| Price   | House Price              |

Example:

| Area |   Price |
| ---: | ------: |
|  800 | 2500000 |
| 1200 | 3900000 |
| 1800 | 6100000 |
| 2500 | 8600000 |

---

# 🤖 Machine Learning Algorithm

**Algorithm Used**

* Simple Linear Regression

**Problem Type**

* Supervised Learning
* Regression

Mathematical Equation:

[
\hat{y}=mx+c
]

Where:

* **x** → House Area
* **ŷ** → Predicted House Price
* **m** → Slope
* **c** → Intercept

---

# 📈 Exploratory Data Analysis (EDA)

The notebook (`analysis.ipynb`) includes:

* Dataset Preview
* Shape & Dimensions
* Data Types
* Missing Value Analysis
* Duplicate Analysis
* Statistical Summary
* Unique Value Analysis
* Distribution Analysis
* Scatter Plot
* Correlation Analysis
* Box Plot
* IQR Outlier Detection
* Data Quality Report
* Final EDA Summary

---

# 🛠️ Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Scikit-learn
* Joblib
* Jupyter Notebook

---

# 📦 Installation

Clone the repository:

```bash
git clone https://github.com/your-username/House_Price_Prediction.git
```

Move into the project directory:

```bash
cd House_Price_Prediction
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# ▶️ Run the Project

## Train the Model

```bash
python src/train.py
```

This will:

* Load the dataset
* Train the Linear Regression model
* Evaluate model performance
* Save the trained model as `model.pkl`

---

## Predict House Price

```bash
python src/predict.py
```

Example:

```text
Enter House Area (sq.ft.): 2500
```

Output:

```text
Predicted House Price:
₹8,685,000.00
```

---

# 📊 Model Evaluation Metrics

The model is evaluated using:

* Mean Absolute Error (MAE)
* Mean Squared Error (MSE)
* Root Mean Squared Error (RMSE)
* R² Score (Coefficient of Determination)

Additional validation includes:

* Residual Analysis
* Actual vs Predicted Comparison
* Residual Plot

---

# 📁 Output Files

After training:

```text
outputs/
│
└── actual_vs_predicted.csv
```

Saved model:

```text
model.pkl
```

---

# ✅ Features

* Professional project structure
* Modular Python code
* Reusable utility functions
* Exception handling
* Production-ready prediction script
* Model persistence with Joblib
* Clean EDA notebook
* Regression evaluation metrics
* Actual vs Predicted comparison
* Beginner-friendly implementation

---

# 📚 Key Learning Outcomes

By completing this project, you will understand:

* Complete Machine Learning workflow
* Business understanding
* Data loading
* Exploratory Data Analysis (EDA)
* Feature selection
* Train-Test Split
* Simple Linear Regression
* Model prediction
* Model evaluation
* Model persistence
* Production-ready project organization

---

# 🔮 Future Improvements

* Multiple Linear Regression
* Feature Engineering
* Polynomial Regression
* Cross Validation
* Hyperparameter Optimization
* Streamlit Web Application
* Flask/FastAPI REST API
* Docker Containerization
* CI/CD Pipeline
* Cloud Deployment

---

# 📖 References

* Scikit-learn Documentation
* Pandas Documentation
* NumPy Documentation
* Matplotlib Documentation

---

# 📜 License

This project is intended for educational and learning purposes.

---

# 👨‍💻 Author

**Sriram Boddu**

* GitHub: https://github.com/sriram806
* LinkedIn: https://www.linkedin.com/in/sriram-boddu-655ba8310/

---

## ⭐ If you found this project useful, consider giving it a star on GitHub!
