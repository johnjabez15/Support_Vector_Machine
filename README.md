### SVM Project – User Purchase Prediction

## 📌 Overview

This project implements a Support Vector Machine (SVM) Classifier to predict whether a website user will make a purchase based on their browsing behavior.The model is trained using a custom dataset and deployed through a Flask web application, allowing users to input their activity details and get an instant prediction.

___

## 📂 Project Structure


```
DataScience/
│
├── SVM/
│   ├── data/
│   │   └── user_purchase_dataset.csv
│   ├── model/
│   │   └── svm_model.pkl
│   ├── static/
│   │   └── style.css
│   ├── templates/
│   │   ├── index.html
│   │   └── result.html
│   ├── train_model.py
│   ├── app.py
│   ├── requirements.txt
│   └── README.md

```


## ⚙️ Installation & Setup

**1️⃣ Clone the repository**

```
git clone <your-repo-url>
cd "DataScience/SVM"
```

**2️⃣ Create a virtual environment (recommended)**

```
python -m venv venv
source venv/bin/activate   # For Linux/Mac
venv\Scripts\activate      # For Windows
```

**3️⃣ Install dependencies**

```
pip install -r requirements.txt
```

## 📊 Dataset

The dataset contains details of website users with the following features:

```
Age (numeric)
Time on Site (numeric)
Pages Viewed (numeric)
Added to Cart (categorical: 0 = No, 1 = Yes)
Purchased (Target: 0 = Unlikely, 1 = Likely)
```

## 🎯 Problem Statement

E-commerce businesses need a way to predict which users are most likely to make a purchase to optimize marketing strategies and improve user experience.This project aims to automate the prediction of user purchase behavior using machine learning to help businesses make data-driven decisions.💡 
Why Support Vector Machine (SVM)?
- Effective in High-Dimensional Spaces: Works well with datasets that have many features.
- Memory Efficient: Uses a subset of training points in the decision function (support vectors), making it memory efficient.
- Versatile Kernels: Can be used with different kernel functions to handle linear and non-linear data relationships.
- Robustness: Provides a good margin of separation, making it robust against small changes in data.



## 🚀 How to Run

**1️⃣ Train the Model**

```
python train_model.py
```

This will create:svm_model.pkl (trained model and scaler)

**2️⃣ Run the Flask App**

```
python app.py
```

Visit http://127.0.0.1:5000/ in your browser.

**🖥️ Frontend Input Example**

Example user input 

form:Age: 30
Time on Site: 12.5
Pages Viewed: 15
Added to Cart: Yes


## 📌 Prediction Goal

The application predicts:Likely to Purchase ✅ — if the user behavior indicates a high probability of a transaction.Unlikely to Purchase ❌ — if the user behavior does not indicate an immediate purchase.

## 🛠 Tech Stack

- Python – Core programming language
- Pandas & NumPy – Data manipulation
- Scikit-learn – Machine learning model training
- Flask – Web framework for deployment
- HTML/CSS – Frontend UI design


## 📌 Future Scope
🔹 Deploy the model on Heroku or Render for public access.
🔹 Improve accuracy using hyperparameter tuning and additional data.
🔹 Add data visualization dashboard for user trends and statistics.
🔹 Integrate with a real-time data stream for
