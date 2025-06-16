from google.cloud import bigquery
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
import joblib

# Authenticate and connect to Google BigQuery
client = bigquery.Client()

# Load preprocessed data from Google Cloud Storage
data = pd.read_csv('gs://your_bucket_id/cleaned_data.csv')

# Feature preparation
X = data[['Age', 'Salary', 'Gender']]  # Features
y = data['High_Spender']  # Target variable

# Encoding categorical variables (Gender)
X['Gender'] = X['Gender'].map({'Male': 1, 'Female': 0})

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Training the Random Forest model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Training the Support Vector Machine (SVM) model
svm_model = SVC(probability=True, random_state=42)
svm_model.fit(X_train, y_train)

# Training the K-Nearest Neighbors (KNN) model
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train, y_train)

# Save the trained models to Google Cloud Storage
joblib.dump(rf_model, 'gs://your_bucket_id/random_forest_model.pkl')
joblib.dump(svm_model, 'gs://your_bucket_id/svm_model.pkl')
joblib.dump(knn_model, 'gs://your_bucket_id/knn_model.pkl')

# Print confirmation
print("Models trained and saved to Google Cloud Storage.")
