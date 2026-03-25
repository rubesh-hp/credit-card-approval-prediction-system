import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
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

# 4. Keep only required features + target
X = data[["Car_Owner", "Propert_Owner", "Annual_income", "EDUCATION"]].copy()
y = data["label"]

# 5. Manual Encoding
X["Car_Owner"] = X["Car_Owner"].map({"Y": 1, "N": 0})
X["Propert_Owner"] = X["Propert_Owner"].map({"Y": 1, "N": 0})

education_map = {
    "Secondary / secondary special": 0,
    "Higher education": 1,
    "Incomplete higher": 2,
    "Academic degree": 3
}
X["EDUCATION"] = X["EDUCATION"].map(education_map)

# Drop rows with missing encoded values
X = X.dropna()
y = y[X.index]

# 6. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 7. Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 8. Train model with class balancing
model = RandomForestClassifier(
    random_state=42,
    class_weight="balanced",
    n_estimators=200,
    max_depth=10
)
model.fit(X_train_scaled, y_train)

# 9. Evaluate
y_pred = model.predict(X_test_scaled)
acc = accuracy_score(y_test, y_pred)

print(f"Model Accuracy: {acc:.2f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# 10. Save model & scaler
joblib.dump(model, "model/credit_card_model.pkl")
joblib.dump(scaler, "model/scaler.pkl")
print("✅ Model and scaler saved successfully")