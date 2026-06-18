# Smart Supply Chain Shipment Tracker (Delivery App)

A machine learning web application designed to predict delivery risks and optimize supply chain logistics. Built using Python, this project leverages historical shipping data to provide actionable insights into potential shipment delays.

## 🚀 Features
* **Delivery Risk Prediction:** Utilizes a Random Forest machine learning model to predict the likelihood of delivery issues or delays.
* **Interactive Web Interface:** Deployed as a user-friendly web application using Streamlit.
* **Data-Driven Insights:** Trained on the comprehensive DataCo Supply Chain dataset to ensure robust feature engineering and accurate predictions.

## 🏗️ How It Was Built (Methodology)
This project was developed through a complete end-to-end machine learning pipeline:

1. **Data Collection & Cleaning:** The project utilizes the **DataCo Supply Chain dataset**. Initial steps involved cleaning the raw data, handling missing values, and preparing the dataset for analysis.
2. **Feature Engineering & Preprocessing:** Key features impacting delivery times and risks were carefully selected. The data was then standardized and scaled (saved as `scaler.pkl`) to ensure the machine learning model could interpret the inputs effectively.
3. **Model Training:** A **Random Forest Classifier** was trained on the preprocessed historical data. This specific algorithm was chosen for its strong performance and high accuracy in classification tasks. The trained model was serialized and saved as `supply_chain_rf_model.pkl`.
4. **Web App Deployment:** Instead of traditional desktop GUI frameworks, the predictive model was integrated into a modern, dynamic web interface built with **Streamlit** (`app.py`). This allows users to input real-time shipment details and receive instant delivery risk predictions directly in their browser.

## 🛠️ Tech Stack
* **Programming Language:** Python
* **Web Framework:** Streamlit
* **Machine Learning:** Scikit-Learn (Random Forest Classifier)
* **Data Processing:** Pandas, NumPy
* **Serialization:** Pickle (for model and scaler deployment)

## 📁 Repository Structure
* `app.py`: The main Streamlit application file containing the web interface and prediction logic.
* `supply_chain_rf_model.pkl`: The pre-trained Random Forest model.
* `scaler.pkl`: The saved data scaler used to standardize inputs before prediction.
* `DataCoSupplyChainDataset (1).csv`: The primary dataset used for training and testing the model.
* `requirements.txt`: A list of all Python dependencies required to run the application.

## 💻 How to Run Locally

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/shankardaya4170-ux/DELIVERY-APP.git](https://github.com/shankardaya4170-ux/DELIVERY-APP.git)
   cd DELIVERY-APP
