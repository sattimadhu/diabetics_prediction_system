# train_models.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import joblib

# Load dataset
df = pd.read_csv("diabetes_prediction_dataset.csv")

# Preprocessing
le_gender = LabelEncoder()
df['gender'] = le_gender.fit_transform(df['gender'])  # Male:1, Female:0

# One-hot encode smoking_history
df = pd.get_dummies(df, columns=['smoking_history'], drop_first=True)

X = df.drop('diabetes', axis=1)
y = df['diabetes']

# Normalize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Save scaler
joblib.dump(scaler, 'models/scaler.pkl')

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train models
models = {
    "Logistic Regression": LogisticRegression(),
    "Random Forest": RandomForestClassifier(),
    "SVM": SVC(probability=True)
}

accuracies = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    accuracies[name] = acc
    joblib.dump(model, f"models/{name.lower().replace(' ', '_')}_model.pkl")

# Save accuracies
joblib.dump(accuracies, "models/accuracies.pkl")
print("Training complete. Models and scaler saved.")
