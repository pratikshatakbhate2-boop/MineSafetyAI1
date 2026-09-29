import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

import joblib


# ==========================================
# 1. Load the dataset
# ==========================================

df = pd.read_csv("mine_safety_dataset.csv")

print("Dataset loaded successfully!")
print("Shape:", df.shape)
print(df.head())


# ==========================================
# 2. Convert text columns into numbers
# ==========================================

label_encoders = {}

categorical_columns = [
    "zone",
    "ventilation",
    "equipment_status"
]

for column in categorical_columns:
    encoder = LabelEncoder()
    df[column] = encoder.fit_transform(df[column])
    label_encoders[column] = encoder


# ==========================================
# 3. Encode the target
# ==========================================

target_encoder = LabelEncoder()

df["risk_level"] = target_encoder.fit_transform(
    df["risk_level"]
)


# ==========================================
# 4. Separate features and target
# ==========================================

X = df.drop("risk_level", axis=1)
y = df["risk_level"]


# ==========================================
# 5. Split data into training and testing
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


print("\nTraining records:", len(X_train))
print("Testing records:", len(X_test))


# ==========================================
# 6. Create Random Forest model
# ==========================================

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)


# ==========================================
# 7. Train the model
# ==========================================

model.fit(X_train, y_train)

print("\nModel training completed!")


# ==========================================
# 8. Test the model
# ==========================================

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy:", round(accuracy * 100, 2), "%")

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        target_names=target_encoder.classes_
    )
)


# ==========================================
# 9. Save the trained model
# ==========================================

joblib.dump(model, "mine_safety_model.pkl")

joblib.dump(
    label_encoders,
    "label_encoders.pkl"
)

joblib.dump(
    target_encoder,
    "target_encoder.pkl"
)


print("\nModel saved successfully!")
print("Created files:")
print("- mine_safety_model.pkl")
print("- label_encoders.pkl")
print("- target_encoder.pkl")