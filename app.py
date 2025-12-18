import streamlit as st
import numpy as np
import joblib
from tensorflow.keras.models import load_model
import plotly.graph_objects as go

# ----------------------------
# Load models and scaler
# ----------------------------
nb_model = joblib.load("naive_bayes.pkl")
rf_model = joblib.load("random_forest.pkl")
scaler = joblib.load("x_scaler.pkl")
lstm_model = load_model("lstm_model.h5")

# ----------------------------
# Define ranges (replace with dataset-derived values)
# ----------------------------
#ranges = {
#    "temp": {"safe_max": 85.0, "warning_max": 95.0, "critical_above": 140.0},
#    "pressure": {"safe_max": 12.0, "warning_max": 15.0, "critical_above": 25.0},
#    "vibration": {"safe_max": 4.0, "warning_max": 6.0, "critical_above": 12.0},
#    "humidity": {"safe_max": 70.0, "warning_max": 85.0, "critical_above": 100.0}
#}

TARGETS = {
    "temp": 70.0,
    "pressure": 35.0,
    "vibration": 1.0,
    "humidity": 50.0,
}

TOLERANCES = {
    # acceptable +/- deviation around target before severe penalty
    "temp": 20.0, 
    "pressure": 12.0, 
    "vibration": 1.0,
    "humidity": 15.0,
}
##70.0
## 40.0
## 4.0
## 30.0

# ----------------------------
# Health scoring helpers
# ----------------------------
#def sensor_score(value, safe_max, warning_max):
#    if value <= safe_max:
#        return 25
#    elif value <= warning_max:
#        return 15
#    else:
#        return 5

def sensor_score_continuous(value, target, tolerance, max_points=25, gamma=1.0):
    # 0 penalty at target, linearly increasing penalty with distance,
    # clamped at tolerance; gamma>1 makes penalty steeper near tolerance
    dev = abs(value - target)
    frac = min(dev / tolerance, 1.0)
    penalty = (frac ** gamma) * max_points
    return max_points - penalty

def health_score_continuous(temp, pressure, vibration, humidity):
    s_temp = sensor_score_continuous(temp, TARGETS["temp"], TOLERANCES["temp"])
    s_pressure = sensor_score_continuous(pressure, TARGETS["pressure"], TOLERANCES["pressure"])
    s_vibration = sensor_score_continuous(vibration, TARGETS["vibration"], TOLERANCES["vibration"])
    s_humidity = sensor_score_continuous(humidity, TARGETS["humidity"], TOLERANCES["humidity"])
    return s_temp + s_pressure + s_vibration + s_humidity  # 0–100




#def health_score(temp, pressure, vibration, humidity, ranges):
#    score_temp = sensor_score(temp, ranges["temp"]["safe_max"], ranges["temp"]["warning_max"])
#    score_pressure = sensor_score(pressure, ranges["pressure"]["safe_max"], ranges["pressure"]["warning_max"])
#    score_vibration = sensor_score(vibration, ranges["vibration"]["safe_max"], ranges["vibration"]["warning_max"])
#    score_humidity = sensor_score(humidity, ranges["humidity"]["safe_max"], ranges["humidity"]["warning_max"])
#    return score_temp + score_pressure + score_vibration + score_humidity

def interpret_score(score):
    if score >= 85:
        return "Excellent"
    elif score >= 70:
        return "Good"
    elif score >= 60:
        return "Atten Required"
    elif score >= 50:
        return "Warning"
    else:
        return "Faulty"

def check_critical_params(temp, pressure, vibration, humidity, targets, tolerances):
    alerts = []
    if abs(temp - targets["temp"]) > tolerances["temp"]:
        alerts.append(("Temperature Critical", "red"))
    if abs(pressure - targets["pressure"]) > tolerances["pressure"]:
        alerts.append(("Pressure Critical", "red"))
    if abs(vibration - targets["vibration"]) > tolerances["vibration"]:
        alerts.append(("Vibration Critical", "red"))
    if abs(humidity - targets["humidity"]) > tolerances["humidity"]:
        alerts.append(("Humidity Critical", "red"))
    return alerts if alerts else [("No critical alerts", "green")]

def show_alert(message, color):
    if color == "red":
        st.markdown(
            f"""
            <div style="animation: blinker 1s linear infinite; color:{color}; font-weight:bold;">
                ⚠️ {message}
            </div>
            <style>
            @keyframes blinker {{
              50% {{ opacity: 0; }}
            }}
            </style>
            """,
            unsafe_allow_html=True
        )
    elif color == "orange":
        st.warning(message)
    elif color == "green":
        st.success(message)
    else:
        st.info(message)


# ----------------------------
# Ensemble + Final Health Score helpers
# ----------------------------
def ensemble_fault_prob(nb_prob, rf_prob, lstm_prob, w=(0.3, 0.3, 0.4)):
    """Weighted average of model probabilities"""
    return w[0]*nb_prob + w[1]*rf_prob + w[2]*lstm_prob

#def final_health_score(sensor_score, fault_prob, alpha=0.7):
#    """Blend sensor score with ML ensemble risk"""
#    return alpha * sensor_score + (1 - alpha) * (100 * (1 - fault_prob))


def final_health_score(sensor_score, fault_prob, alpha=0.7):
    # If ML confidence is very high, reduce sensor weight
    if fault_prob > 0.8:
        alpha = 0.3   # ML dominates
    elif fault_prob > 0.7:
        alpha = 0.4   # balanced
    elif fault_prob > 0.6:
        alpha = 0.5   # balanced
    elif fault_prob > 0.5:
        alpha = 0.6   # balanced
    else:
        alpha = alpha
    return alpha * sensor_score + (1 - alpha) * (100 * (1 - fault_prob))



# ----------------------------
# Prediction + Scoring
# ----------------------------
def predict_all_with_score(temp, pressure, vibration, humidity):
    x = np.array([temp, pressure, vibration, humidity]).reshape(1, -1)
    x_scaled = scaler.transform(x)

    nb_prob = nb_model.predict_proba(x_scaled)[0][1]
    nb_label = "Faulty" if nb_prob >= 0.5 else "Healthy"

    rf_prob = rf_model.predict_proba(x_scaled)[0][1]
    rf_label = "Faulty" if rf_prob >= 0.5 else "Healthy"

    x_seq = x_scaled.reshape((1, 1, x_scaled.shape[1]))
    lstm_prob = lstm_model.predict(x_seq, verbose=0)[0][0]
    lstm_label = "Faulty" if lstm_prob >= 0.5 else "Healthy"

# Continuous sensor score
    sensor_score = health_score_continuous(temp, pressure, vibration, humidity)

    # Ensemble + blended score
    fault_prob = ensemble_fault_prob(nb_prob, rf_prob, lstm_prob)
    final_score = final_health_score(sensor_score, fault_prob, alpha=0.7)
    final_status = interpret_score(final_score)

   # critical_alerts = check_critical_params(temp, pressure, vibration, humidity, ranges)
    critical_alerts = check_critical_params(temp, pressure, vibration, humidity, TARGETS, TOLERANCES)

    


    #score = health_score(temp, pressure, vibration, humidity, ranges)
# Continuous sensor score

    #score = health_score_continuous(temp, pressure, vibration, humidity)
    #status = interpret_score(score)
    #critical_alerts = check_critical_params(temp, pressure, vibration, humidity, ranges)

    return {
        "Naive Bayes": {"label": "Faulty" if nb_prob >= 0.5 else "Healthy", "prob": nb_prob},
        "Random Forest": {"label": "Faulty" if rf_prob >= 0.5 else "Healthy", "prob": rf_prob},
        "LSTM": {"label": "Faulty" if lstm_prob >= 0.5 else "Healthy", "prob": lstm_prob},
        "Sensor Score": {"score": sensor_score},
        "Ensemble": {"fault_prob": fault_prob, "final_score": final_score, "final_status": final_status},
        "Critical Alerts": critical_alerts if critical_alerts else ["None"]
    }


# ----------------------------
# Streamlit UI
# ----------------------------
st.title("⚙️ Machine Health Monitoring Dashboard")

st.sidebar.header("Input Sensor Values")


defaults = {"temp": 70.0, "pressure": 35.0, "vibration": 1.0, "humidity": 50.0}
for key, val in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = val



# --- Ideal Values Button ---
if st.sidebar.button("Ideal Values"):
    st.session_state.temp = 70.0
    st.session_state.pressure = 35.0
    st.session_state.vibration = 1.0
    st.session_state.humidity = 50.0

# --- Critical Values Button ---
if st.sidebar.button("Temp Critical"):
    st.session_state.temp = 131.0

# --- Critical Values Button ---
if st.sidebar.button("Press Critical"):
    st.session_state.pressure = 81.0

# --- Critical Values Button ---
if st.sidebar.button("Vib Critical"):
    st.session_state.vibration = 5.0

# --- Critical Values Button ---
if st.sidebar.button("Humid Critical"):
    st.session_state.humidity = 91.0

# --- Sliders (read from session_state) ---
#temp = st.sidebar.slider("Temperature (°C)", 0.0, 200.0, st.session_state.temp, key="temp",step=0.1)

##### temp
col1, col2, col3 = st.sidebar.columns([1,5,2])  # wider right column

with col1:
    if st.button("<", key="temp_minus"):
        st.session_state.temp = round(st.session_state.temp - 0.1, 1)

with col3:
    if st.button(">", key="temp_plus"):
        st.session_state.temp = round(st.session_state.temp + 0.1, 1)

with col2:
    temp = st.slider("Temperature (°C)", 0.0, 200.0, step=0.1, key="temp")

#st.write(f"Temperature: {temp:.1f} °C")

#### pressure
col1, col2, col3 = st.sidebar.columns([1,5,2])  # wider right column

with col1:
    if st.button("<", key="pressure_minus"):
        st.session_state.pressure = round(st.session_state.pressure - 0.1, 1)

with col3:
    if st.button(">", key="pressure_plus"):
        st.session_state.pressure = round(st.session_state.pressure + 0.1, 1)

with col2:
    pressure = st.slider("Pressure (bar)", 0.0, 150.0, step=0.1, key="pressure")

#st.write(f"Pressure: {pressure:.1f} bar")

# --- Vibration ---

col1, col2, col3 = st.sidebar.columns([1,5,2])
with col1:
    if st.button("<", key="vibration_minus"):
        st.session_state.vibration = round(st.session_state.vibration - 0.1, 1)
with col3:
    if st.button(">", key="vibration_plus"):
        st.session_state.vibration = round(st.session_state.vibration + 0.1, 1)
with col2:
    vibration = st.slider("Vibration (g)", 0.0, 20.0, step=0.1, key="vibration")
#st.write(f"Vibration: {vibration:.1f} g")

# --- Humidity ---

col1, col2, col3 = st.sidebar.columns([1,5,2])
with col1:
    if st.button("<", key="humidity_minus"):
        st.session_state.humidity = round(st.session_state.humidity - 0.1, 1)
with col3:
    if st.button(">", key="humidity_plus"):
        st.session_state.humidity = round(st.session_state.humidity + 0.1, 1)
with col2:
    humidity = st.slider("Humidity (%)", 0.0, 100.0, step=0.1, key="humidity")
#st.write(f"Humidity: {humidity:.1f} %")



#pressure = st.sidebar.slider("Pressure (bar)", 0.0, 150.0, st.session_state.pressure, key="pressure",step=0.1)
#vibration = st.sidebar.slider("Vibration (g)", 0.0, 10.0, st.session_state.vibration, key="vibration",step=0.1)
#humidity = st.sidebar.slider("Humidity (%)", 0.0, 150.0, st.session_state.humidity, key="humidity",step=0.1)



result = predict_all_with_score(temp, pressure, vibration, humidity)

# Predictions
st.subheader("Model Predictions")
for model, output in result.items():
    if model in ["Naive Bayes", "Random Forest", "LSTM"]:
        st.write(f"**{model}** → {output['label']} (Prob: {output['prob']:.2f})")

# Health Score Gauge
st.subheader("Health Score")
#score = result["Health Score"]["score"]
#status = result["Health Score"]["status"]

score = result["Ensemble"]["final_score"]
status = result["Ensemble"]["final_status"]


fig = go.Figure(go.Indicator(
    mode="gauge+number",
    value=score,
    title={'text': f"Status: {status}"},
    gauge={
        'axis': {'range': [0, 100]},
        'bar': {'color': "black"},
        'steps': [
            {'range': [0, 50], 'color': 'red'},
            {'range': [50, 70], 'color': 'orange'},
            {'range': [70, 90], 'color': 'blue'},
            {'range': [90, 100], 'color': 'green'}
        ],
        'threshold': {
            'line': {'color': "black", 'width': 4},
            'thickness': 0.75,
            'value': score
        }
    }
))

st.plotly_chart(fig, use_container_width=True)

# Critical Alerts
#st.subheader("Critical Alerts")
#for alert in result["Critical Alerts"]:
#    if alert != "None":
#        st.error(alert)
#    else:
#        st.success("No critical alerts")

st.subheader("Critical Alerts")
for msg, color in result["Critical Alerts"]:
 show_alert(msg, color)