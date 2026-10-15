# 🎓 Student Performance Predictor

The Student Performance Predictor is a machine learning project that predicts students' exam scores based on factors such as parental involvement, access to resources, motivation, family income, and study-related conditions.

The project was built to practice end-to-end machine learning workflows, including data preprocessing, categorical encoding, feature scaling, model training, and evaluation using Python and Scikit-learn.

---

## ✨ Features

**Data Preprocessing**  
Handles missing values in categorical features and prepares the dataset for machine learning.

**Categorical Encoding**  
Converts categorical data into numerical representations:
- Ordinal Encoding for features with an ordered relationship
- One-Hot Encoding for nominal categorical features

**Machine Learning Model**  
Implements Linear Regression to predict students' exam scores.

**Feature Scaling**  
Uses StandardScaler to standardize features before training the model.

**Model Evaluation**  
Evaluates performance using:
- R² Score
- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)

**Data Visualization**  
Visualizes actual versus predicted exam scores to examine the model's prediction accuracy.

---

## 📊 Model Performance

The model was evaluated using a test set containing 20% of the dataset.

- **R² Score:** ~0.77
- **MAE:** ~0.44
- **MSE:** ~3.24
- **RMSE:** ~1.80

The model explains approximately 77% of the variation in exam scores on the test set. The error metrics provide additional insight into the differences between actual and predicted scores.

---

## ⚙️ Technologies Used

**Python** – Core programming language  
**Pandas** – Data manipulation and preprocessing  
**Scikit-learn** – Data preprocessing, model training, and evaluation  
**Linear Regression** – Predictive machine learning model  
**Matplotlib** – Data visualization  

---

## 🚀 How to Run

1. Clone the repository:
```bash
git clone <repo-url>
cd Student-Performance-Predictor
```

2. Create and activate a virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Run the model:
```bash
python main.py
```

---

## 📬 Contact

**LinkedIn**: https://www.linkedin.com/in/mariam-buzaladze-aa22b6297/

**Email**: mariambuzaladze10@gmail.com
