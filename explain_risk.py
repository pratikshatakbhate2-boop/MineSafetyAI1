import pandas as pd
import joblib


# ==========================================
# 1. Load trained model and encoders
# ==========================================

model = joblib.load("mine_safety_model.pkl")
label_encoders = joblib.load("label_encoders.pkl")
target_encoder = joblib.load("target_encoder.pkl")


# ==========================================
# 2. Current mine/zone conditions
# ==========================================

input_data = {
    "zone": "Zone C",
    "methane": 2.0,
    "co": 25.0,
    "temperature": 40.0,
    "humidity": 78.0,
    "ventilation": "Poor",
    "worker_count": 40,
    "equipment_status": "Fault",
    "previous_incidents": 3
}


# ==========================================
# 3. Convert input into DataFrame
# ==========================================

df = pd.DataFrame([input_data])


# ==========================================
# 4. Encode categorical values
# ==========================================

for column, encoder in label_encoders.items():
    df[column] = encoder.transform(df[column])


# ==========================================
# 5. Get current prediction
# ==========================================

current_prediction = model.predict(df)[0]

current_risk = target_encoder.inverse_transform(
    [current_prediction]
)[0]

current_probabilities = model.predict_proba(df)[0]

class_names = target_encoder.classes_

current_probability_data = dict(
    zip(class_names, current_probabilities)
)

current_high_probability = (
    current_probability_data.get("High", 0) * 100
)


# ==========================================
# 6. Baseline values
# ==========================================
# These represent comparatively normal conditions
# for this prototype.
#
# IMPORTANT:
# These are NOT official mine safety limits.
# They are only reference values for explanation.

baseline_values = {
    "methane": 0.8,
    "co": 8.0,
    "temperature": 28.0,
    "humidity": 55.0,
    "ventilation": "Good",
    "worker_count": 15,
    "equipment_status": "Normal",
    "previous_incidents": 0
}


# ==========================================
# 7. Calculate local impact of each factor
# ==========================================

impacts = []


for feature, baseline_value in baseline_values.items():

    # Copy the original input
    changed_input = input_data.copy()

    # Change only ONE feature
    changed_input[feature] = baseline_value

    # Convert to DataFrame
    changed_df = pd.DataFrame([changed_input])

    # Encode categorical values
    for column, encoder in label_encoders.items():
        changed_df[column] = encoder.transform(
            changed_df[column]
        )

    # Predict probability
    changed_probabilities = model.predict_proba(
        changed_df
    )[0]

    changed_probability_data = dict(
        zip(class_names, changed_probabilities)
    )

    changed_high_probability = (
        changed_probability_data.get("High", 0) * 100
    )

    # Calculate impact
    impact = (
        current_high_probability
        - changed_high_probability
    )

    impacts.append({
        "feature": feature,
        "current_value": input_data[feature],
        "baseline_value": baseline_value,
        "impact": impact
    })


# ==========================================
# 8. Sort factors by impact
# ==========================================

impacts.sort(
    key=lambda x: x["impact"],
    reverse=True
)


# ==========================================
# 9. Display result
# ==========================================

print("\n==============================================")
print("          AI RISK EXPLANATION")
print("==============================================")

print("\nZone:", input_data["zone"])

print(
    "Predicted Risk:",
    current_risk.upper()
)

print(
    "Current HIGH-risk probability:",
    f"{current_high_probability:.2f}%"
)

print("\n----------------------------------------------")
print("WHY IS THIS ZONE AT HIGH RISK?")
print("----------------------------------------------")


# Display important contributors

for item in impacts:

    if item["impact"] > 0:

        print(
            f"\n{item['feature']}:"
        )

        print(
            f"  Current value : {item['current_value']}"
        )

        print(
            f"  Normal value  : {item['baseline_value']}"
        )

        print(
            f"  Model impact  : "
            f"+{item['impact']:.2f} percentage points"
        )


# ==========================================
# 10. Human-readable explanation
# ==========================================

print("\n----------------------------------------------")
print("SAFETY FACTORS DETECTED")
print("----------------------------------------------")

if input_data["methane"] > baseline_values["methane"]:
    print(
        "• Methane is elevated compared with the "
        "prototype normal reference."
    )

if input_data["co"] > baseline_values["co"]:
    print(
        "• CO level is elevated compared with the "
        "prototype normal reference."
    )

if input_data["temperature"] > baseline_values["temperature"]:
    print(
        "• Temperature is higher than the "
        "prototype normal reference."
    )

if input_data["humidity"] > baseline_values["humidity"]:
    print(
        "• Humidity is relatively high."
    )

if input_data["ventilation"] != baseline_values["ventilation"]:
    print(
        "• Ventilation is not in the normal "
        "reference condition."
    )

if input_data["worker_count"] > baseline_values["worker_count"]:
    print(
        "• A relatively large number of workers "
        "are present in the zone."
    )

if input_data["equipment_status"] != baseline_values["equipment_status"]:
    print(
        "• Equipment status indicates a fault."
    )

if input_data["previous_incidents"] > 0:
    print(
        "• Previous incidents are recorded for "
        "this zone."
    )


# ==========================================
# 11. Final explanation
# ==========================================

print("\n----------------------------------------------")
print("AI INTERPRETATION")
print("----------------------------------------------")

print(
    "\nThe model predicts HIGH risk because multiple "
    "safety-related conditions in this zone differ "
    "from the prototype's normal reference conditions."
)

print(
    "\nThe listed model impacts show which individual "
    "changes most affected the HIGH-risk probability "
    "for this particular prediction."
)

print(
    "\nNote: These are model-based contributions, "
    "not proof of physical causation or official "
    "mine-safety thresholds."
)

print("\n==============================================")