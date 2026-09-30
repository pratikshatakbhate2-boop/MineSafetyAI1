
import pandas as pd
import joblib

model = joblib.load("mine_safety_model.pkl")
label_encoders = joblib.load("label_encoders.pkl")
target_encoder = joblib.load("target_encoder.pkl")

BASELINE = {
    "methane": 0.8,
    "co": 8.0,
    "temperature": 28.0,
    "humidity": 55.0,
    "ventilation": "Good",
    "worker_count": 15,
    "equipment_status": "Normal",
    "previous_incidents": 0
}

def explain_risk(input_data):
    def high_risk_probability(data):
        df = pd.DataFrame([data])

        for column, encoder in label_encoders.items():
            df[column] = encoder.transform(df[column])

        probabilities = model.predict_proba(df)[0]
        probability_map = dict(
            zip(target_encoder.classes_, probabilities)
        )

        return probability_map.get("High", 0) * 100

    current_probability = high_risk_probability(input_data)
    impacts = []

    for feature, baseline_value in BASELINE.items():
        changed_input = input_data.copy()
        changed_input[feature] = baseline_value

        changed_probability = high_risk_probability(
            changed_input
        )

        impacts.append({
            "feature": feature,
            "current_value": input_data[feature],
            "baseline_value": baseline_value,
            "impact": current_probability - changed_probability
        })

    impacts.sort(
        key=lambda item: item["impact"],
        reverse=True
    )

    return {
        "high_risk_probability": current_probability,
        "impacts": impacts
    }