import pandas as pd
import joblib


# ==========================================
# Load trained model and encoders
# ==========================================

model = joblib.load("mine_safety_model.pkl")
label_encoders = joblib.load("label_encoders.pkl")
target_encoder = joblib.load("target_encoder.pkl")


# ==========================================
# Reusable risk prediction function
# ==========================================

def predict_risk(
    zone,
    methane,
    co,
    temperature,
    humidity,
    ventilation,
    worker_count,
    equipment_status,
    previous_incidents
):

    # Create input data
    input_data = pd.DataFrame([{
        "zone": zone,
        "methane": methane,
        "co": co,
        "temperature": temperature,
        "humidity": humidity,
        "ventilation": ventilation,
        "worker_count": worker_count,
        "equipment_status": equipment_status,
        "previous_incidents": previous_incidents
    }])

    # Convert text values into numbers
    for column, encoder in label_encoders.items():
        input_data[column] = encoder.transform(
            input_data[column]
        )

    # Predict risk
    prediction = model.predict(input_data)

    predicted_risk = target_encoder.inverse_transform(
        prediction
    )[0]

    # Get probabilities
    probabilities = model.predict_proba(input_data)[0]

    class_names = target_encoder.classes_

    probability_data = dict(
        zip(class_names, probabilities)
    )

    # Return results
    return {
        "risk_level": predicted_risk,
        "probabilities": probability_data
    }


# ==========================================
# Test the function
# ==========================================

result = predict_risk(
    zone="Zone C",
    methane=2.0,
    co=25.0,
    temperature=40.0,
    humidity=78.0,
    ventilation="Poor",
    worker_count=40,
    equipment_status="Fault",
    previous_incidents=3
)


# ==========================================
# Display result
# ==========================================

print("\n===================================")
print("       MINE SAFETY AI")
print("===================================")

print("Predicted Risk:", result["risk_level"].upper())

print("\nPrediction Probabilities:")

for risk, probability in result["probabilities"].items():
    print(f"{risk}: {probability * 100:.2f}%")

print("===================================")