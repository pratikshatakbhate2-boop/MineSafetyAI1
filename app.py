
import streamlit as st
import pandas as pd
from predict_risk import predict_risk
from risk_explanation import explain_risk
from alert_system import create_alerts

st.set_page_config(page_title="MineGuard AI", layout="wide")
df = pd.read_csv("mine_safety_dataset.csv")
alert_df = create_alerts()

# Professional Dashboard Header
st.markdown("""
<div style="
    background: linear-gradient(90deg, #123524, #28784d);
    padding: 25px;
    border-radius: 12px;
    color: white;
    text-align: center;
">
    <h1 style="color: white; margin: 0;">
        MINEGUARD AI
    </h1>
    <p style="font-size: 18px; margin-top: 10px;">
        AI-Based Coal Mine Safety Monitoring System
    </p>
    <p style="font-size: 14px;">
        Smart Governance | Risk Prediction | Safety Monitoring
    </p>
</div>
""", unsafe_allow_html=True)
# Dashboard Navigation Sidebar
st.sidebar.title("MineGuard AI")
st.sidebar.subheader("Dashboard Navigation")

st.sidebar.info(
    "AI-Based Coal Mine Safety Monitoring System"
)

st.sidebar.markdown("""
### Dashboard Sections

- Mine Sensor Data
- Methane Monitoring
- Mine Zone Monitoring
- Worker Monitoring
- Risk Analysis
- Safety Alerts
- Ventilation Monitoring
- Equipment Monitoring
- AI Risk Prediction
- AI Risk Explanation
- Previous Incidents
- Download Report
""")
st.subheader("Coal Mine Safety Monitoring Dashboard")

# Dashboard Overview
st.subheader("Dashboard Overview")

total_zones = df["zone"].nunique()
total_records = len(df)

high_risk_records = (
    df["risk_level"].astype(str).str.strip().str.lower()
    == "high"
).sum()

equipment_faults = (
    df["equipment_status"].astype(str).str.strip().str.lower()
    == "fault"
).sum()

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Mine Zones", total_zones)

with col2:
    st.metric("Sensor Records", total_records)

with col3:
    st.metric("High-Risk Records", high_risk_records)

with col4:
    st.metric("Equipment Fault Records", equipment_faults)

# Load your team's dataset



st.write("Mine Sensor Data")
st.dataframe(df)

st.subheader("Methane Levels by Mine Zone")

st.bar_chart(
    df,
    x="zone",
    y="methane"
)
# Mine Zone Monitoring
st.subheader("Mine Zone Monitoring")

selected_zone = st.sidebar.selectbox(
    "Select Mine Zone",
    sorted(df["zone"].unique())
)

# Risk Level Filter
selected_risk = st.sidebar.selectbox(
    "Filter by Risk Level",
    ["All", "Low", "Medium", "High"]
)

zone_data = df[df["zone"] == selected_zone]

if selected_risk == "All":
    filtered_data = zone_data
else:
    filtered_data = zone_data[
        zone_data["risk_level"] == selected_risk
    ]

st.subheader("Filtered Sensor Records")
st.dataframe(filtered_data)

# Download Filtered Records
st.download_button(
    label="Download Filtered Records",
    data=filtered_data.to_csv(index=False),
    file_name="filtered_mine_records.csv",
    mime="text/csv"
)

st.write(f"Sensor readings for {selected_zone}")
st.dataframe(zone_data)

st.subheader(f"Sensor Summary - {selected_zone}")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Methane",
        f"{zone_data['methane'].mean():.2f}"
    )

with col2:
    st.metric(
        "Carbon Monoxide",
        f"{zone_data['co'].mean():.2f}"
    )

with col3:
    st.metric(
        "Temperature",
        f"{zone_data['temperature'].mean():.2f} °C"
    )

with col4:
    st.metric(
        "Humidity",
        f"{zone_data['humidity'].mean():.2f}%"
    )
    
# Worker Monitoring

# Worker Monitoring
st.subheader("Worker Monitoring")

if "worker_count" in zone_data.columns:
    latest_record = zone_data.iloc[-1]
    worker_count = int(latest_record["worker_count"])

    st.metric(
        label=f"Workers in {selected_zone}",
        value=worker_count
    )

    st.info(
        f"Latest recorded worker count in "
        f"{selected_zone}: {worker_count}"
    )
else:
    st.warning("Worker count column not found.")
    
# Risk Analysis
st.subheader("Mine Zone Risk Analysis")

risk_counts = zone_data["risk_level"].value_counts()

st.bar_chart(risk_counts)

st.write(f"Risk records for {selected_zone}")

st.dataframe(
    zone_data[
        ["zone", "methane", "co",
         "temperature", "risk_level"]
    ]
)

# High-Risk Alerts
# Mine Safety Alerts
st.subheader("🚨 Mine Safety Alerts")

zone_alerts = alert_df[
    alert_df["zone"] == selected_zone
]

active_alerts = zone_alerts[
    zone_alerts["alert_status"] == "Alert"
]

if not active_alerts.empty:

    st.error(
        f"⚠️ {len(active_alerts)} active alert record(s) found in {selected_zone}"
    )

    # Get unique alert messages
    alert_messages = (
        active_alerts["alert_message"]
        .dropna()
        .str.split("; ")
        .explode()
        .drop_duplicates()
    )

    # Get unique recommended actions
    recommended_actions = (
        active_alerts["recommended_action"]
        .dropna()
        .str.split("; ")
        .explode()
        .drop_duplicates()
    )

    st.markdown("### 🚨 Alert Types")

    for message in alert_messages:
        st.warning(f"🚨 {message}")

    st.markdown("### ✅ Recommended Actions")

    for action in recommended_actions:
        st.info(f"✅ {action}")

else:

    st.success(
        f"✅ No active safety alerts found for {selected_zone}."
    )
    
# Ventilation Monitoring
st.subheader("Ventilation Monitoring")

ventilation_counts = zone_data["ventilation"].value_counts()

st.bar_chart(ventilation_counts)

st.write("Ventilation Status Summary")
st.dataframe(
    ventilation_counts.reset_index()
)

# Equipment Status Monitoring
st.subheader("Equipment Status Monitoring")

equipment_counts = zone_data["equipment_status"].value_counts()

col1, col2 = st.columns(2)

with col1:
    st.bar_chart(equipment_counts)

with col2:
    st.write("Equipment Status Summary")
    st.dataframe(equipment_counts)

    fault_count = (
        zone_data["equipment_status"] == "Fault"
    ).sum()

    if fault_count > 0:
        st.warning(
            f"{fault_count} equipment fault records "
            f"found in {selected_zone}."
        )
    else:
        st.success("No equipment faults recorded.")
        
# AI Risk Prediction
st.subheader("AI Risk Prediction")

# Use the last recorded sensor reading
# from the selected mine zone
latest = zone_data.iloc[-1]

result = predict_risk(
    zone=latest["zone"],
    methane=latest["methane"],
    co=latest["co"],
    temperature=latest["temperature"],
    humidity=latest["humidity"],
    ventilation=latest["ventilation"],
    worker_count=int(latest["worker_count"]),
    equipment_status=latest["equipment_status"],
    previous_incidents=int(latest["previous_incidents"])
)

st.metric(
    "Predicted Risk Level",
    result["risk_level"]
)

st.write("Prediction Probabilities")

probabilities = pd.DataFrame(
    {
        "Risk Level": list(result["probabilities"].keys()),
        "Probability (%)": [
            value * 100
            for value in result["probabilities"].values()
        ]
    }
)

st.bar_chart(
    probabilities,
    x="Risk Level",
    y="Probability (%)"
)

# AI Risk Status Indicator
st.subheader("AI Risk Status")

predicted_risk = str(result["risk_level"]).strip().lower()

if predicted_risk == "high":
    st.error("HIGH RISK - Immediate safety review required.")

elif predicted_risk == "medium":
    st.warning("MEDIUM RISK - Increased monitoring recommended.")

elif predicted_risk == "low":
    st.success("LOW RISK - Continue routine monitoring.")

else:
    st.info(f"Predicted risk: {result['risk_level']}")
    
# Actual vs AI-Predicted Risk
st.subheader("Dataset Risk vs AI Prediction")

col1, col2 = st.columns(2)

with col1:
    st.metric(
        "Dataset Risk Level",
        latest["risk_level"]
    )

with col2:
    st.metric(
        "AI-Predicted Risk Level",
        result["risk_level"]
    )

if str(latest["risk_level"]).lower() == str(result["risk_level"]).lower():
    st.success("AI prediction matches the dataset risk label.")
else:
    st.warning("AI prediction differs from the dataset risk label.")
    # AI Risk Explanation
st.subheader("AI Risk Explanation")

# Prepare the selected zone's sensor readings
input_data = {
    "zone": latest["zone"],
    "methane": latest["methane"],
    "co": latest["co"],
    "temperature": latest["temperature"],
    "humidity": latest["humidity"],
    "ventilation": latest["ventilation"],
    "worker_count": int(latest["worker_count"]),
    "equipment_status": latest["equipment_status"],
    "previous_incidents": int(latest["previous_incidents"])
}

explanation = explain_risk(input_data)

st.metric(
    "Predicted High-Risk Probability",
    f"{explanation['high_risk_probability']:.2f}%"
)

impact_df = pd.DataFrame(explanation["impacts"])

st.subheader("Factors Affecting High-Risk Probability")

st.bar_chart(
    impact_df,
    x="feature",
    y="impact"
)

st.dataframe(impact_df)

# Human-Readable AI Risk Explanation
st.subheader("Why Did the AI Predict This Risk?")

positive_impacts = impact_df[
    impact_df["impact"] > 0
].sort_values("impact", ascending=False)

if not positive_impacts.empty:
    st.write(
        "The following factors increased the model's "
        "predicted High-risk probability compared "
        "with the prototype reference values:"
    )

    for _, item in positive_impacts.iterrows():
        st.write(
            f"**{item['feature'].replace('_', ' ').title()}**"
        )

        st.write(
            f"Current value: {item['current_value']} | "
            f"Reference value: {item['baseline_value']}"
        )

        st.write(
            f"Increase in High-risk probability: "
            f"{item['impact']:.2f} percentage points"
        )
else:
    st.info(
        "No individual factor increased the predicted "
        "High-risk probability relative to its "
        "prototype reference value."
    )

st.caption(
    "These are model-based comparisons, not proof "
    "of causation or official mine safety thresholds."
)

# Previous Incident Monitoring
st.subheader("Previous Incident Monitoring")

latest_incidents = int(latest["previous_incidents"])

col1, col2 = st.columns(2)

with col1:
    st.metric(
        "Previously Recorded Incidents",
        latest_incidents
    )

with col2:
    if latest_incidents > 0:
        st.warning(
            f"The latest record for {selected_zone} "
            f"reports {latest_incidents} previous incidents."
        )
    else:
        st.success(
            "No previous incidents reported "
            "in the latest record."
        )

st.write("Previous Incident Records")

st.bar_chart(
    zone_data["previous_incidents"].value_counts()
    .sort_index()
)

# Download Mine Zone Report
st.subheader("Download Mine Zone Report")

st.write(
    f"Download sensor records and risk information "
    f"for {selected_zone}."
)

report_csv = zone_data.to_csv(index=False).encode("utf-8")

st.download_button(
    label="Download Zone Report",
    data=report_csv,
    file_name=f"{selected_zone.replace(' ', '_')}_report.csv",
    mime="text/csv"
)