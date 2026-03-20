import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import joblib 

df = pd.read_csv('features.csv')

X = df.drop('label', axis=1)
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = joblib.load('scaler.pkl')

X_test_scaled = scaler.transform(X_test)

model_files = ['svm_model.pkl', 'random_forest_model.pkl', 'knn_model.pkl']

for model_file in model_files:
    model = joblib.load(model_file)
    y_pred = model.predict(X_test_scaled)

    print(f"\n=== {model_file.replace('_model.pkl','').upper()} ===")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred, pos_label='cancer'):.4f}")
    print(f"Recall: {recall_score(y_test, y_pred, pos_label='cancer'):.4f}")
    print(f"F1-Score: {f1_score(y_test, y_pred, pos_label='cancer'):.4f}")
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

print("\nModels tested and validated on unseen images.")