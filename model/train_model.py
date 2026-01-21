import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# 1. Load datasets
data = pd.read_csv("dataset/Credit_card.csv")
label = pd.read_csv("dataset/Credit_card_label.csv")

# 2. Merge datasets
data = data.merge(label, on="Ind_ID", how="left")

# 3. Handle missing values
data["GENDER"] = data["GENDER"].fillna("Unknown")
data["Annual_income"] = data["Annual_income"].fillna(data["Annual_income"].median())
data["Birthday_count"] = data["Birthday_count"].fillna(data["Birthday_count"].median())
data["Type_Occupation"] = data["Type_Occupation"].fillna("Unknown")

# 4. Keep only the 4 features + target
X = data[["Car_Owner", "Propert_Owner", "Annual_income", "EDUCATION"]]
y = data["label"]

# 5. Encode categorical columns
cat_cols = ["Car_Owner", "Propert_Owner", "EDUCATION"]
le = LabelEncoder()
for col in cat_cols:
    X[col] = le.fit_transform(X[col])

# 6. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 7. Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 8. Train model
model = RandomForestClassifier(random_state=42)
model.fit(X_train_scaled, y_train)

# 9. Evaluate
y_pred = model.predict(X_test_scaled)
acc = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {acc:.2f}")

# 10. Save model & scaler
joblib.dump(model, "model/credit_card_model.pkl")
joblib.dump(scaler, "model/scaler.pkl")
print("✅ Model and scaler saved successfully")
