# Step 1: Import Libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler

# Step 2: Load Dataset
df = pd.read_csv('student-mat.csv', sep=';')
print("Columns:", df.columns.tolist())  # Debug check

# Step 3: Create Binary Target (Pass = 1 if G3 >= 10, else 0)
df['performance'] = df['G3'].apply(lambda x: 1 if x >= 10 else 0)


# Step 4: Drop irrelevant or target columns
X = df.drop(['G1', 'G2', 'G3', 'performance'], axis=1)
y = df['performance']

# Step 5: Encode categorical features
X = pd.get_dummies(X, drop_first=True)

# Step 6: Feature Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 7: Split Dataset
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Step 8: Train Model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Step 9: Predict and Evaluate
y_pred = model.predict(X_test)
 
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Step 10: Save Model (Optional)
import pickle
with open('student_model.pkl', 'wb') as f:
    pickle.dump(model, f)
