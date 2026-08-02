## Price-Driven Demand Prediction in Retail using Machine Learning

## 📌 Overview

This project focuses on predicting product demand in the retail sector based on pricing and other influencing factors using machine learning techniques. Accurate demand forecasting helps businesses optimize pricing strategies, manage inventory efficiently, and maximize profits.

---

## 🎯 Objectives

* Analyze how pricing impacts product demand
* Build and compare multiple machine learning models
* Identify the best-performing model for demand prediction
* Provide insights to support data-driven retail decisions

---

## 📊 Dataset

The dataset includes key retail features such as:

* Product price
* Historical sales/demand
* Promotions/discounts
* Time-based features (seasonality, trends)

---

## ⚙️ Models Used

The following machine learning models were implemented and evaluated:

* Linear Regression
* Random Forest Regressor
* CatBoost Regressor
* **XGBoost Regressor (Best Model)**

---

## 🏆 Best Model: XGBoost

After evaluating all models based on performance metrics (e.g., RMSE, MAE, R² score), **XGBoost outperformed other models** due to:

* Better handling of non-linear relationships
* Strong regularization to prevent overfitting
* High predictive accuracy
* Efficient computation

---

## 📈 Results

* XGBoost achieved the highest accuracy among all models
* Improved demand prediction compared to baseline models
* Demonstrated strong capability in capturing price-demand relationships

---

## 🛠️ Technologies Used

* Python
* Pandas, NumPy
* Scikit-learn
* XGBoost
* Matplotlib / Seaborn

---

## 🚀 How to Run

1. Clone the repository:

   ```bash
   git clone https://github.com/muhammdumairnad/Price-Driven-Demand-Prediction-in-Retail-using-Machine-Learning.git
   ```
2. Navigate to the project directory
3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```
4. Run the notebook or script

## 🌐 Live Demo 🚀 

**Try the App Here:

** 👉 https://price-driven-demand-prediction-in-retail-using-machine-learnin.streamlit.app/

---

## 📌 Future Improvements

* Hyperparameter tuning for further optimization
* Incorporating external factors (weather, economic indicators)
* Deploying the model as an API or web app

---

## 🤝 Contributing

Contributions are welcome! Feel free to fork the repo and submit a pull request.

---

## 📜 License

This project is open-source and available under the MIT License.
