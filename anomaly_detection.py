import pandas as pd
from sklearn.ensemble import IsolationForest


# ==========================================
# 1. Load mine safety dataset
# ==========================================

df = pd.read_csv("mine_safety_dataset.csv")

print("Dataset loaded successfully!")
print("Total records:", len(df))


# ==========================================
# 2. Select numerical safety parameters
# ==========================================

features = [
    "methane",
    "co",
    "temperature",
    "humidity",
    "worker_count",
    "previous_incidents"
]

X = df[features]


# ==========================================
# 3. Train anomaly detection model
# ==========================================

model = IsolationForest(
    contamination=0.10,
    random_state=42
)

model.fit(X)


# ==========================================
# 4. Detect anomalies
# ==========================================

df["anomaly"] = model.predict(X)


# Isolation Forest:
#  1  = normal
# -1  = anomaly

df["anomaly_status"] = df["anomaly"].map({
    1: "Normal",
    -1: "Anomaly"
})


# ==========================================
# 5. Count results
# ==========================================

normal_count = (
    df["anomaly_status"] == "Normal"
).sum()

anomaly_count = (
    df["anomaly_status"] == "Anomaly"
).sum()


# ==========================================
# 6. Display summary
# ==========================================

print("\n==============================================")
print("        MINE SAFETY ANOMALY DETECTION")
print("==============================================")

print("\nNormal records :", normal_count)
print("Anomalous records:", anomaly_count)

print("\n==============================================")
print("       ANOMALOUS MINE CONDITIONS")
print("==============================================")


# ==========================================
# 7. Display anomalous records
# ==========================================

anomalies = df[
    df["anomaly_status"] == "Anomaly"
]

display_columns = [
    "zone",
    "methane",
    "co",
    "temperature",
    "humidity",
    "worker_count",
    "previous_incidents",
    "risk_level"
]

print(
    anomalies[display_columns].head(10).to_string(
        index=False
    )
)


# ==========================================
# 8. Save results
# ==========================================

df.to_csv(
    "mine_safety_anomaly_results.csv",
    index=False
)


print("\n==============================================")
print("Anomaly detection completed!")
print("Results saved to:")
print("mine_safety_anomaly_results.csv")
print("==============================================")