📦 Supply Chain Analytics & Demand Prediction
🚀 Project Overview

This project focuses on supply chain analytics and demand prediction using the Online Retail transactional dataset.

The objective is to:

Extract meaningful product features using NLP

Perform clustering on mixed-type retail data

Predict product demand (Quantity)

Support inventory planning and stock optimization

The pipeline combines Natural Language Processing, feature engineering, clustering, and machine learning models to analyze real-world ERP-style retail data.

📊 Dataset

Dataset: Online Retail Dataset (UK-based gift shop)

Total records used: 50,000 transactions

Attributes:

InvoiceNo

StockCode

Description

Quantity

InvoiceDate

UnitPrice

CustomerID

Country

🧹 Data Preprocessing
1️⃣ Missing Value Handling

Removed rows with null values in Description and CustomerID.

2️⃣ Text Processing on Product Description

Used NLP techniques:

Tokenization

Stopword removal

Lemmatization

POS tagging

Extracted:

Product Type (nouns from description)

Colour Type (color detection from predefined list)

🏗 Feature Engineering
🟢 Revenue Feature
Revenue = Quantity × UnitPrice
🟢 Date Features Extracted

From InvoiceDate:

Year

Month

Day

DayOfWeek

These features help capture seasonality and demand patterns.

🔀 Encoding

Categorical variables encoded using:

Label Encoding

Mixed data types handled for clustering.

📌 Clustering – K-Prototypes

Since dataset contains both:

Numerical features

Categorical features

We used K-Prototypes clustering to segment products.

Steps:

Determined optimal K using cost curve

Selected K = 3 clusters

Assigned Cluster Number as a new feature

This helps in grouping similar product transactions.

🧠 Cluster Classification

After generating clusters:

Treated Cluster number as target

Trained Linear SVC to classify new records into clusters

Evaluation:

Accuracy score used for validation

📈 Demand Prediction

Objective:
Predict Quantity (product demand)

Models Tested:

Random Forest

KNN

SVC

AdaBoost

Logistic Regression

Naive Bayes

Decision Tree

Gradient Boosting

Best Performing Model:

Random Forest

Evaluation Metrics:

Accuracy

F1 Score

🛠 Technologies Used

Python

Pandas

NumPy

NLTK

Scikit-learn

KPrototypes (kmodes)

Matplotlib

🧩 Project Pipeline

Data Cleaning

NLP-based Feature Extraction

Revenue & Time Feature Engineering

Mixed-type Clustering (K-Prototypes)

Cluster Classification

Demand Prediction using ML models

Model Evaluation

🎯 Business Impact

This project demonstrates how machine learning can support:

Inventory Planning

Product Segmentation

Demand Forecasting

Supply Chain Optimization

Stock Management

The approach can be extended to:

Stock-out risk prediction

Vendor performance analysis

Procurement intelligence

📌 Future Improvements

Convert demand prediction to regression-based forecasting

Implement time-series forecasting (ARIMA, XGBoost, LSTM)

Use business-focused metrics (MAE, RMSE, MAPE)

Deploy model via API for real-time prediction

👨‍💻 Author

Santosh Kuruventi
Machine Learning & Supply Chain Analytics Enthusiast
