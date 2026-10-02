import streamlit as st
import pandas as pd
import altair as alt

from predict_risk import predict_risk
from risk_explanation import explain_risk
from alert_system import create_alerts


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="MineGuard AI",
    layout="wide"
)


# =========================================================
# LOAD DATA
# =========================================================

df = pd.read_csv("mine_safety_dataset.csv")

alert_df = create_alerts()


# =========================================================
# RISK COLOR SETTINGS
# =========================================================

RISK_COLORS = {
    "High": "#FF0000",
    "Medium": "#FFD700",
    "Low": "#00A000"
}


# =========================================================
# MINEGUARD AI HEADER
# =========================================================

st.markdown("""
<div style="
background: linear-gradient(90deg, #123524, #28784d);
padding: 25px;
border-radius: 12px;
color: white;
text-align: center;
">
<h1 style="
color: white;
margin: 0;
font-size: 36px;
">
MINEGUARD AI
</h1>

<p style="
color: white;
font-size: 18px;
margin-top: 10px;
margin-bottom: 5px;
">
AI-Based Coal Mine Safety Monitoring System
</p>

<p style="
color: white;
font-size: 14px;
margin-top: 5px;
">
Smart Governance | Risk Prediction | Safety Monitoring
</p>

</div>
""", unsafe_allow_html=True)


# =========================================================
# SIDEBAR NAVIGATION
# =========================================================

st.sidebar.title("MineGuard AI")

st.sidebar.info(
    "AI-Based Coal Mine Safety Monitoring System"
)

st.sidebar.subheader("Dashboard Sections")


selected_section = st.sidebar.radio(
    "Select Section",
    [
        "Dashboard Overview",
        "Mine Sensor Data",
        "Methane Monitoring",
        "Mine Zone Monitoring",
        "Worker Monitoring",
        "Risk Analysis",
        "Safety Alerts",
        "Ventilation Monitoring",
        "Equipment Monitoring",
        "AI Risk Prediction",
        "AI Risk Explanation",
        "Previous Incidents",
        "Download Report"
    ]
)


# =========================================================
# MINE ZONE SELECTION
# =========================================================

selected_zone = st.sidebar.selectbox(
    "Select Mine Zone",
    sorted(df["zone"].unique())
)


selected_risk = st.sidebar.selectbox(
    "Filter by Risk Level",
    [
        "All",
        "Low",
        "Medium",
        "High"
    ]
)


# =========================================================
# ZONE DATA
# =========================================================

zone_data = df[
    df["zone"] == selected_zone
]


if selected_risk == "All":

    filtered_data = zone_data

else:

    filtered_data = zone_data[
        zone_data["risk_level"] == selected_risk
    ]


# =========================================================
# 1. DASHBOARD OVERVIEW
# =========================================================

if selected_section == "Dashboard Overview":

    st.subheader("Dashboard Overview")

    total_zones = df["zone"].nunique()

    total_records = len(df)

    high_risk_records = (
        df["risk_level"]
        .astype(str)
        .str.strip()
        .str.lower()
        == "high"
    ).sum()

    equipment_faults = (
        df["equipment_status"]
        .astype(str)
        .str.strip()
        .str.lower()
        == "fault"
    ).sum()


    col1, col2, col3, col4 = st.columns(4)


    with col1:

        st.metric(
            "Mine Zones",
            total_zones
        )


    with col2:

        st.metric(
            "Sensor Records",
            total_records
        )


    with col3:

        st.metric(
            "High-Risk Records",
            high_risk_records
        )


    with col4:

        st.metric(
            "Equipment Fault Records",
            equipment_faults
        )


    st.success(
        "MineGuard AI dashboard is monitoring mine safety data."
    )


# =========================================================
# 2. MINE SENSOR DATA
# =========================================================

elif selected_section == "Mine Sensor Data":

    st.subheader("Mine Sensor Data")

    st.write(
        "Complete sensor records collected from the mine."
    )


    st.dataframe(
        df,
        use_container_width=True
    )


# =========================================================
# 3. METHANE MONITORING
# =========================================================

elif selected_section == "Methane Monitoring":

    st.subheader("Methane Monitoring")

    st.write(
        "Methane levels recorded across different mine zones."
    )


    st.bar_chart(
        df,
        x="zone",
        y="methane"
    )


    st.subheader(
        f"Methane Readings - {selected_zone}"
    )


    methane_data = zone_data[
        ["zone", "methane"]
    ]


    st.dataframe(
        methane_data,
        use_container_width=True
    )


    average_methane = zone_data[
        "methane"
    ].mean()


    st.metric(
        "Average Methane Level",
        f"{average_methane:.2f}"
    )


# =========================================================
# 4. MINE ZONE MONITORING
# =========================================================

elif selected_section == "Mine Zone Monitoring":

    st.subheader("Mine Zone Monitoring")


    st.write(
        f"Sensor readings for {selected_zone}"
    )


    st.subheader("Filtered Sensor Records")


    st.dataframe(
        filtered_data,
        use_container_width=True
    )


    st.download_button(
        label="Download Filtered Records",
        data=filtered_data.to_csv(
            index=False
        ),
        file_name="filtered_mine_records.csv",
        mime="text/csv"
    )


    st.subheader(
        f"Sensor Summary - {selected_zone}"
    )


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


# =========================================================
# 5. WORKER MONITORING
# =========================================================

elif selected_section == "Worker Monitoring":

    st.subheader("Worker Monitoring")


    if "worker_count" in zone_data.columns:

        latest_record = zone_data.iloc[-1]

        worker_count = int(
            latest_record["worker_count"]
        )


        st.metric(
            label=f"Workers in {selected_zone}",
            value=worker_count
        )


        st.info(
            f"Latest recorded worker count in "
            f"{selected_zone}: {worker_count}"
        )


        st.subheader("Worker Count Records")


        st.dataframe(
            zone_data[
                [
                    "zone",
                    "worker_count"
                ]
            ],
            use_container_width=True
        )


    else:

        st.warning(
            "Worker count column not found."
        )


# =========================================================
# 6. RISK ANALYSIS
# =========================================================

elif selected_section == "Risk Analysis":

    st.subheader("Mine Zone Risk Analysis")


    risk_counts = (
        zone_data["risk_level"]
        .value_counts()
        .reindex(
            ["Low", "Medium", "High"],
            fill_value=0
        )
        .reset_index()
    )


    risk_counts.columns = [
        "Risk Level",
        "Records"
    ]


    risk_chart = alt.Chart(
        risk_counts
    ).mark_bar().encode(

        x=alt.X(
            "Risk Level:N",
            sort=["Low", "Medium", "High"],
            title="Risk Level"
        ),

        y=alt.Y(
            "Records:Q",
            title="Number of Records"
        ),

        color=alt.Color(
            "Risk Level:N",
            scale=alt.Scale(
                domain=[
                    "High",
                    "Medium",
                    "Low"
                ],
                range=[
                    RISK_COLORS["High"],
                    RISK_COLORS["Medium"],
                    RISK_COLORS["Low"]
                ]
            ),
            legend=alt.Legend(
                title="Risk Level"
            )
        ),

        tooltip=[
            "Risk Level",
            "Records"
        ]
    ).properties(
        height=400
    )


    st.altair_chart(
        risk_chart,
        use_container_width=True
    )


    st.write(
        f"Risk records for {selected_zone}"
    )


    st.dataframe(
        zone_data[
            [
                "zone",
                "methane",
                "co",
                "temperature",
                "risk_level"
            ]
        ],
        use_container_width=True
    )


# =========================================================
# 7. SAFETY ALERTS
# =========================================================

elif selected_section == "Safety Alerts":

    st.subheader("🚨 Mine Safety Alerts")


    zone_alerts = alert_df[
        alert_df["zone"] == selected_zone
    ]


    active_alerts = zone_alerts[
        zone_alerts["alert_status"] == "Alert"
    ]


    if not active_alerts.empty:

        st.error(
            f"⚠️ {len(active_alerts)} active alert record(s) "
            f"found in {selected_zone}"
        )


        alert_messages = (
            active_alerts["alert_message"]
            .dropna()
            .str.split("; ")
            .explode()
            .drop_duplicates()
        )


        recommended_actions = (
            active_alerts["recommended_action"]
            .dropna()
            .str.split("; ")
            .explode()
            .drop_duplicates()
        )


        st.markdown(
            "### 🚨 Alert Types"
        )


        for message in alert_messages:

            st.warning(
                f"🚨 {message}"
            )


        st.markdown(
            "### ✅ Recommended Actions"
        )


        for action in recommended_actions:

            st.info(
                f"✅ {action}"
            )


    else:

        st.success(
            f"✅ No active safety alerts found "
            f"for {selected_zone}."
        )


# =========================================================
# 8. VENTILATION MONITORING
# =========================================================

elif selected_section == "Ventilation Monitoring":

    st.subheader(
        "Ventilation Monitoring"
    )


    ventilation_counts = zone_data[
        "ventilation"
    ].value_counts()


    st.bar_chart(
        ventilation_counts
    )


    st.write(
        "Ventilation Status Summary"
    )


    st.dataframe(
        ventilation_counts.reset_index(),
        use_container_width=True
    )


# =========================================================
# 9. EQUIPMENT MONITORING
# =========================================================

elif selected_section == "Equipment Monitoring":

    st.subheader(
        "Equipment Status Monitoring"
    )


    equipment_counts = zone_data[
        "equipment_status"
    ].value_counts()


    col1, col2 = st.columns(2)


    with col1:

        st.bar_chart(
            equipment_counts
        )


    with col2:

        st.write(
            "Equipment Status Summary"
        )


        st.dataframe(
            equipment_counts,
            use_container_width=True
        )


        fault_count = (
            zone_data["equipment_status"]
            == "Fault"
        ).sum()


        if fault_count > 0:

            st.warning(
                f"{fault_count} equipment fault records "
                f"found in {selected_zone}."
            )

        else:

            st.success(
                "No equipment faults recorded."
            )


# =========================================================
# 10. AI RISK PREDICTION
# =========================================================

elif selected_section == "AI Risk Prediction":

    st.subheader(
        "AI Risk Prediction"
    )


    latest = zone_data.iloc[-1]


    result = predict_risk(
        zone=latest["zone"],
        methane=latest["methane"],
        co=latest["co"],
        temperature=latest["temperature"],
        humidity=latest["humidity"],
        ventilation=latest["ventilation"],
        worker_count=int(
            latest["worker_count"]
        ),
        equipment_status=latest[
            "equipment_status"
        ],
        previous_incidents=int(
            latest["previous_incidents"]
        )
    )


    st.metric(
        "Predicted Risk Level",
        result["risk_level"]
    )


    st.write(
        "Prediction Probabilities"
    )


    probabilities = pd.DataFrame(
        {
            "Risk Level":
                list(
                    result["probabilities"].keys()
                ),

            "Probability (%)":
                [
                    value * 100
                    for value in
                    result["probabilities"].values()
                ]
        }
    )


    # Ensure consistent order
    probabilities["Risk Level"] = pd.Categorical(
        probabilities["Risk Level"],
        categories=[
            "Low",
            "Medium",
            "High"
        ],
        ordered=True
    )


    probabilities = probabilities.sort_values(
        "Risk Level"
    )


    probability_chart = alt.Chart(
        probabilities
    ).mark_bar().encode(

        x=alt.X(
            "Risk Level:N",
            sort=["Low", "Medium", "High"],
            title="Risk Level"
        ),

        y=alt.Y(
            "Probability (%):Q",
            title="Probability (%)"
        ),

        color=alt.Color(
            "Risk Level:N",
            scale=alt.Scale(
                domain=[
                    "High",
                    "Medium",
                    "Low"
                ],
                range=[
                    RISK_COLORS["High"],
                    RISK_COLORS["Medium"],
                    RISK_COLORS["Low"]
                ]
            ),
            legend=alt.Legend(
                title="Risk Level"
            )
        ),

        tooltip=[
            "Risk Level",
            alt.Tooltip(
                "Probability (%):Q",
                format=".2f"
            )
        ]
    ).properties(
        height=400
    )


    st.altair_chart(
        probability_chart,
        use_container_width=True
    )


    # =====================================================
    # AI RISK STATUS
    # =====================================================

    st.subheader(
        "AI Risk Status"
    )


    predicted_risk = (
        str(result["risk_level"])
        .strip()
        .lower()
    )


    if predicted_risk == "high":

        st.error(
            "HIGH RISK - Immediate safety review required."
        )

    elif predicted_risk == "medium":

        st.warning(
            "MEDIUM RISK - Increased monitoring recommended."
        )

    elif predicted_risk == "low":

        st.success(
            "LOW RISK - Continue routine monitoring."
        )

    else:

        st.info(
            f"Predicted risk: "
            f"{result['risk_level']}"
        )


    # =====================================================
    # DATASET RISK VS AI PREDICTION
    # =====================================================

    st.subheader(
        "Dataset Risk vs AI Prediction"
    )


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


    if (
        str(latest["risk_level"]).lower()
        ==
        str(result["risk_level"]).lower()
    ):

        st.success(
            "AI prediction matches the dataset risk label."
        )

    else:

        st.warning(
            "AI prediction differs from the dataset risk label."
        )


# =========================================================
# 11. AI RISK EXPLANATION
# =========================================================

elif selected_section == "AI Risk Explanation":

    st.subheader(
        "AI Risk Explanation"
    )


    latest = zone_data.iloc[-1]


    input_data = {

        "zone": latest["zone"],

        "methane": latest["methane"],

        "co": latest["co"],

        "temperature":
            latest["temperature"],

        "humidity":
            latest["humidity"],

        "ventilation":
            latest["ventilation"],

        "worker_count":
            int(latest["worker_count"]),

        "equipment_status":
            latest["equipment_status"],

        "previous_incidents":
            int(latest["previous_incidents"])
    }


    explanation = explain_risk(
        input_data
    )


    st.metric(
        "Predicted High-Risk Probability",
        f"{explanation['high_risk_probability']:.2f}%"
    )


    impact_df = pd.DataFrame(
        explanation["impacts"]
    )


    st.subheader(
        "Factors Affecting High-Risk Probability"
    )


    st.bar_chart(
        impact_df,
        x="feature",
        y="impact"
    )


    st.dataframe(
        impact_df,
        use_container_width=True
    )


    st.subheader(
        "Why Did the AI Predict This Risk?"
    )


    positive_impacts = impact_df[
        impact_df["impact"] > 0
    ].sort_values(
        "impact",
        ascending=False
    )


    if not positive_impacts.empty:

        st.write(
            "The following factors increased the "
            "model's predicted High-risk probability "
            "compared with the prototype reference values:"
        )


        for _, item in positive_impacts.iterrows():

            st.write(
                f"**{item['feature'].replace('_', ' ').title()}**"
            )


            st.write(
                f"Current value: "
                f"{item['current_value']} | "
                f"Reference value: "
                f"{item['baseline_value']}"
            )


            st.write(
                f"Increase in High-risk probability: "
                f"{item['impact']:.2f} percentage points"
            )


    else:

        st.info(
            "No individual factor increased the "
            "predicted High-risk probability relative "
            "to its prototype reference value."
        )


    st.caption(
        "These are model-based comparisons, not proof "
        "of causation or official mine safety thresholds."
    )


# =========================================================
# 12. PREVIOUS INCIDENTS
# =========================================================

elif selected_section == "Previous Incidents":

    st.subheader(
        "Previous Incident Monitoring"
    )


    latest = zone_data.iloc[-1]


    latest_incidents = int(
        latest["previous_incidents"]
    )


    col1, col2 = st.columns(2)


    with col1:

        st.metric(
            "Previously Recorded Incidents",
            latest_incidents
        )


    with col2:

        if latest_incidents > 0:

            st.warning(
                f"The latest record for "
                f"{selected_zone} reports "
                f"{latest_incidents} previous incidents."
            )

        else:

            st.success(
                "No previous incidents reported "
                "in the latest record."
            )


    st.write(
        "Previous Incident Records"
    )


    incident_counts = (
        zone_data[
            "previous_incidents"
        ]
        .value_counts()
        .sort_index()
    )


    st.bar_chart(
        incident_counts
    )


# =========================================================
# 13. DOWNLOAD REPORT
# =========================================================

elif selected_section == "Download Report":

    st.subheader(
        "Download Mine Zone Report"
    )


    st.write(
        f"Download sensor records and risk "
        f"information for {selected_zone}."
    )


    report_csv = (
        zone_data
        .to_csv(index=False)
        .encode("utf-8")
    )


    st.download_button(
        label="Download Zone Report",
        data=report_csv,
        file_name=(
            f"{selected_zone.replace(' ', '_')}"
            f"_report.csv"
        ),
        mime="text/csv"
    )