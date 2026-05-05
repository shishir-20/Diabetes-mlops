import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

print("Training started...")

# Load data
df = pd.read_csv("data/diabetes.csv")

# Clean data
cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
df[cols] = df[cols].replace(0, np.nan)
df.fillna(df.mean(), inplace=True)

# Split features
X = df.drop("Outcome", axis=1)
y = df["Outcome"]

# Scale
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# Train model
model = RandomForestClassifier(random_state=42,class_weight={0:1, 1:2})
model.fit(X_train, y_train)

# Method 2: Threshold tuning
y_prob = model.predict_proba(X_test)[:, 1]
y_pred = (y_prob > 0.35).astype(int)

# Evaluation
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# Save model
pickle.dump(model, open("model/diabetes_model.pkl", "wb"))
pickle.dump(scaler, open("model/scaler.pkl", "wb"))

print("Training completed")
print("Model saved at: model/diabetes_model.pkl")