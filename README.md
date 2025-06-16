
# **Retail Spending Prediction using Machine Learning**

## **Project Overview**
This project aims to predict customer spending behavior based on **transactional** and **demographic** data using machine learning models. The goal is to classify customers as **High Spenders** or **Low Spenders** based on their **age**, **salary**, and **gender**. The models used include **Random Forest**, **Support Vector Machine (SVM)**, and **K-Nearest Neighbors (KNN)**.

## **Tools Used**
- **Google Cloud Platform (GCP)**: All tasks, including data processing and model training, were performed in **Google Cloud** using **BigQuery**, **AI Platform**, and **Google Cloud Storage**.
- **Python Libraries**: The project uses **Scikit-learn** for machine learning, **joblib** for model saving, **matplotlib** and **seaborn** for data visualization.
- **Google Colab**: For training models and generating results, the project was executed in **Google Colab**, with data uploaded from **Google Cloud Storage**.

---

## **Steps Involved**

### **1. Data Collection & Preparation**
The project uses two main datasets:
1. **Store Transactions Data**: Contains information on customer purchases, including `CustomerID`, `ProductID`, `Amount`, and `Date`.
2. **Customer Demographics Data**: Includes `CustomerID`, `Age`, `Salary`, `Gender`, and `Country`.

Both datasets were stored in **Google BigQuery**, and data was queried and processed directly within **Google Cloud**.

#### **Data Preprocessing**:
- The data from both datasets was **merged** using the common column `CustomerID`.
- **Missing values** were handled by **dropping rows** with `NaN` values.
- A new feature, `High_Spender`, was created based on whether the customer's spending (`Amount`) was above or below the median spending.

#### **Cloud Storage**:
- The **cleaned dataset** was saved to **Google Cloud Storage** as `cleaned_data.csv`.
- **Data was uploaded** directly from **Google Cloud** into **Google Colab** for model training.

### **2. Model Training**

#### **Machine Learning Models**:
Three machine learning models were used to classify customers as **High Spenders** or **Low Spenders**:
1. **Random Forest Classifier**: A robust ensemble model used for classification.
2. **Support Vector Machine (SVM)**: A classifier known for its effectiveness in binary classification tasks.
3. **K-Nearest Neighbors (KNN)**: A simple, intuitive model based on proximity to neighboring data points.

The models were trained using **Google Colab** with data pulled from **Google Cloud Storage**.

#### **Model Evaluation**:
Each model was evaluated using:
- **Accuracy**, **Precision**, **Recall**, **F1-Score**
- **ROC Curve** and **AUC** (Area Under Curve)

The performance metrics were used to determine the best model for the given task.

#### **Model Saving**:
- After training, all models were saved as **`.pkl`** files using **joblib**.
- These models were uploaded back to **Google Cloud Storage**, making them accessible for future predictions.

---

### **3. Model Evaluation & Visualization**

#### **Confusion Matrix**:
- For each model, a **confusion matrix** was plotted to evaluate how well the model classified the **High Spenders** and **Low Spenders**.

#### **ROC Curve**:
- The **ROC curve** was generated for each model, and the **AUC** score was calculated to compare the models' performance.

#### **Feature Importance (Random Forest)**:
- **Feature importance** was visualized to show which features (e.g., `Age`, `Salary`, `Gender`) were most impactful in predicting spending behavior.

---

## **Repository Structure**

```
/retail-spending-prediction
│
├── /data                    # Folder for storing datasets (Google Cloud Storage)
│   ├── store_transactions.csv
│   └── store_customers.csv
│
├── /models                  # Folder for storing trained models (pickled files in Google Cloud)
│   ├── random_forest_model.pkl
│   ├── svm_model.pkl
│   └── knn_model.pkl
│
├── /scripts                 # Folder for Python scripts used in the cloud environment
│   ├── data_preprocessing.py # Script for data cleaning and preprocessing using BigQuery
│   ├── model_training.py     # Script for model training using Google Cloud AI Platform
│   ├── model_evaluation.py   # Script for evaluating models and generating metrics/graphs
│   ├── visualize_results.py  # Script for generating plots (confusion matrix, ROC curve, etc.)
│   └── eda.py               # Script for Exploratory Data Analysis (EDA)
│
├── /requirements            # Folder for required dependencies (if necessary)
│   └── requirements.txt      # A file listing Python libraries for the project
│
├── README.md                # Project description, instructions, and setup guide
└── .gitignore               # Ignore file for excluding unnecessary files (e.g., model outputs, temporary files)
```

---

## **How to Use This Repository**

### **Step 1: Set Up Google Cloud**:
1. **Google Cloud Storage**: Ensure that you have a **Google Cloud Storage bucket** where the data and models will be stored.
2. **BigQuery**: Set up your **BigQuery dataset** with the **store transactions** and **customer data**.

### **Step 2: Upload Data**:
Upload the **store transactions** and **customer data** to **BigQuery** using the Google Cloud Console.

### **Step 3: Train the Models**:
Run the **model_training.py** script after uploading the data. The models will be trained using **Google Cloud AI Platform** and saved as `.pkl` files.

### **Step 4: Evaluate the Models**:
Run the **model_evaluation.py** script to evaluate the models and visualize their performance using confusion matrices, ROC curves, and AUC scores.

### **Step 5: Download the Models**:
After training and evaluation, download the models from **Google Cloud Storage**. The models will be available as **`random_forest_model.pkl`**, **`svm_model.pkl`**, and **`knn_model.pkl`**.

### **Step 6: Visualize the Results**:
Run the **visualize_results.py** script to generate performance visualizations, including confusion matrices and ROC curves.

---

## **Conclusion**

This project provides a detailed approach to predicting customer spending behavior using machine learning. The models trained in this project can be used for targeted marketing, personalized promotions, and customer segmentation.

By following the steps outlined in this repository, users can train and evaluate models using **Google Cloud** tools, with the ability to download trained models for further use.

---

### **Required Libraries**:
To run the code, ensure the following Python libraries are installed:
- `pandas`
- `numpy`
- `scikit-learn`
- `joblib`
- `matplotlib`
- `seaborn`
- `google-cloud-storage`
- `google-cloud-bigquery`

Use the following command to install the dependencies:
```bash
pip install -r requirements.txt
```

---

### **License**:
This project is licensed 
