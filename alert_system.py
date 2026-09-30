import pandas as pd


def generate_alert(row):
    alerts = []
    actions = []

    # Anomaly-based alert
    if row["anomaly_status"] == "Anomaly":
        alerts.append("Abnormal mine condition detected")
        actions.append(
            "Inspect the affected mine zone and investigate the abnormal condition"
        )

    # Risk-based alert
    if row["risk_level"] == "High":
        alerts.append("High-risk condition detected")
        actions.append(
            "Immediately inspect the affected mine zone"
        )

    # Methane condition
    if row["methane"] > 1.0:
        alerts.append("Elevated methane level detected")
        actions.append(
            "Check ventilation and methane control systems"
        )

    # CO condition
    if row["co"] > 10:
        alerts.append("Elevated carbon monoxide detected")
        actions.append(
            "Check air quality and ventilation"
        )

    # Temperature condition
    if row["temperature"] > 35:
        alerts.append("High temperature detected")
        actions.append(
            "Inspect ventilation and cooling conditions"
        )

    # Ventilation condition
    if row["ventilation"] in ["Poor", "Reduced"]:
        alerts.append("Poor ventilation condition detected")
        actions.append(
            "Inspect and improve mine ventilation"
        )

    # Equipment condition
    if row["equipment_status"] == "Fault":
        alerts.append("Equipment fault detected")
        actions.append(
            "Inspect faulty equipment before continued operation"
        )

    # Previous incidents
    if row["previous_incidents"] > 0:
        alerts.append("Previous safety incidents recorded")
        actions.append(
            "Review previous incidents and increase monitoring"
        )

    # Alert status
    if not alerts:
        alert_status = "Normal"
        recommended_action = "No immediate action required"
    else:
        alert_status = "Alert"
        recommended_action = "; ".join(actions)

    return pd.Series([
        alert_status,
        "; ".join(alerts),
        recommended_action
    ])


def create_alerts():

    # Load anomaly detection results
    df = pd.read_csv("mine_safety_anomaly_results.csv")

    # Generate alert information
    df[
        ["alert_status", "alert_message", "recommended_action"]
    ] = df.apply(generate_alert, axis=1)

    # Save alert results
    df.to_csv(
        "mine_safety_alert_results.csv",
        index=False
    )

    return df


# Run alert system only when this file is executed directly
if __name__ == "__main__":

    df = create_alerts()

    print("==============================================")
    print("        MINE SAFETY ALERT SYSTEM")
    print("==============================================")

    print("Total records:", len(df))

    print(
        "Alert records:",
        (df["alert_status"] == "Alert").sum()
    )

    print("\nSample alerts:")

    print(
        df[df["alert_status"] == "Alert"][
            [
                "zone",
                "alert_status",
                "alert_message",
                "recommended_action"
            ]
        ].head(10).to_string(index=False)
    )

    print("\nAlert system completed!")
    print("Results saved to: mine_safety_alert_results.csv")