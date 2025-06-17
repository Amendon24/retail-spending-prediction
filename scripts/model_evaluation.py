import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
import pandas as pd

# Load the trained models from Google Cloud Storage
rf_model = joblib.load('gs://your_bucket_id/random_forest_model.pkl')
svm_model = joblib.load('gs://your_bucket_id/svm_model.pkl')
knn_model = joblib.load('gs://your_bucket_id/knn_model.pkl')

# Load preprocessed data from Google Cloud Storage
data = pd.read_csv('gs://your_bucket_id/cleaned_data.csv')

# Feature preparation
X = data[['Age', 'Salary', 'Gender']]  # Features
y = data['High_Spender']  # Target variable

# Encoding categorical variables (Gender)
X['Gender'] = X['Gender'].map({'Male': 1, 'Female': 0})

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Function to plot the confusion matrix
def plot_confusion_matrix(cm, model_name):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Low Spender', 'High Spender'], yticklabels=['Low Spender', 'High Spender'])
    plt.title(f'{model_name} Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

# Function to plot the ROC curve
def plot_roc_curve(fpr, tpr, auc, model_name):
    plt.plot(fpr, tpr, color='blue', label=f'{model_name} ROC curve (AUC = {auc:.2f})')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.title(f'{model_name} ROC Curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc='lower right')
    plt.show()

# Evaluation for Random Forest Model
y_pred_rf = rf_model.predict(X_test)
cm_rf = confusion_matrix(y_test, y_pred_rf)
plot_confusion_matrix(cm_rf, 'Random Forest')

y_prob_rf = rf_model.predict_proba(X_test)[:, 1]
roc_auc_rf = roc_auc_score(y_test, y_prob_rf)
fpr_rf, tpr_rf, _ = roc_curve(y_test, y_prob_rf)
plot_roc_curve(fpr_rf, tpr_rf, roc_auc_rf, 'Random Forest')

# Evaluation for SVM Model
y_pred_svm = svm_model.predict(X_test)
cm_svm = confusion_matrix(y_test, y_pred_svm)
plot_confusion_matrix(cm_svm, 'SVM')

y_prob_svm = svm_model.predict_proba(X_test)[:, 1]
roc_auc_svm = roc_auc_score(y_test, y_prob_svm)
fpr_svm, tpr_svm, _ = roc_curve(y_test, y_prob_svm)
plot_roc_curve(fpr_svm, tpr_svm, roc_auc_svm, 'SVM')

# Evaluation for KNN Model
y_pred_knn = knn_model.predict(X_test)
cm_knn = confusion_matrix(y_test, y_pred_knn)
plot_confusion_matrix(cm_knn, 'KNN')

y_prob_knn = knn_model.predict_proba(X_test)[:, 1]
roc_auc_knn = roc_auc_score(y_test, y_prob_knn)
fpr_knn, tpr_knn, _ = roc_curve(y_test, y_prob_knn)
plot_roc_curve(fpr_knn, tpr_knn, roc_auc_knn, 'KNN')

# Print Final AUC Scores for All Models
print(f'Random Forest AUC: {roc_auc_rf:.4f}')
print(f'SVM AUC: {roc_auc_svm:.4f}')
print(f'KNN AUC: {roc_auc_knn:.4f}')
