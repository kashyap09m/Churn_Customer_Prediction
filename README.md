# **Customer Churn Prediction Web App**


A user-friendly web application built with Streamlit to predict customer churn. This project leverages a machine learning model trained on customer data to identify which customers are likely to cancel their service. The app provides an interactive interface for making predictions and visualizing key data insights.

## 🚀 Features
* Interactive Popups: Receive prediction results through clear, color-coded messages with emojis (🎉 for staying, 😞 for churning).

* Data Visualization Dashboard: Explore insightful charts and graphs derived from the customer dataset to understand churn patterns. Visualizations include:
  - Churn Rate by Gender
  - Customer Age Distribution
  - Churn Rate by Contract Type
  - Internet Service Distribution

* Clean UI: A simple and intuitive two-page navigation system for a seamless user experience.

## 🛠️ Tech Stack
* Language: Python 3

* Machine Learning: Scikit-learn

* Data Manipulation: Pandas, NumPy

* Web Framework: Streamlit

* Data Visualization: Matplotlib, Seaborn

## 📂 Project Structure
.

├── 📄 app.py                   # Main Streamlit application file

├── 📄 notebook.ipynb            # Jupyter Notebook for data analysis and model training

├── 📄 customer_churn_data.csv   # The dataset used for training

├── 📄 model.pkl                 # The pre-trained machine learning model

├── 📄 scaler.pkl                # The pre-trained data scaler

├── 📄 requirements.txt          # List of Python dependencies

└── 📄 README.md                 # This file

## ⚙️ Setup and Installation
Follow these steps to set up and run the project on your local machine.

1. Clone the Repository
git clone https://github.com/your-username/customer-churn-prediction.git
cd customer-churn-prediction

2. Create a Virtual Environment (Recommended)
It's a good practice to create a virtual environment to manage project dependencies.

- For Windows
python -m venv venv
venv\Scripts\activate

- For macOS/Linux
python3 -m venv venv
source venv/bin/activate

3. Install Dependencies
Create a requirements.txt :

Then, install all the necessary libraries using pip:

pip install -r requirements.txt

## ▶️ How to Run the App
Once the setup is complete, you can launch the Streamlit application with a single command:

streamlit run app.py

Your web browser should automatically open to the application's URL (usually http://localhost:8501).

## 🧠 Model Details
The prediction model was trained using a Support Vector Machine (SVM) classifier (or specify your final model, e.g., Random Forest). The model was optimized and trained on the provided customer dataset, achieving an accuracy of approximately 86% on the test set. The notebook.ipynb file contains all the steps for data preprocessing, exploratory data analysis, feature engineering, and model training.
