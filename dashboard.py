# PM2.5 dashboard. Run gnn_lstm.ipynb first (it writes dashboard_preds.csv), then: streamlit run dashboard.py
# forecast = hourly PM2.5, risk level = class of 24 h average
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import f1_score

DATA = "processed"
HORIZONS = [24, 48, 72]
MODELS = {"LSTM": "lstm_pm25", "GNN-LSTM": "gnn_pm25"}
LEVELS = {1: "Good", 2: "Sensitive groups", 3: "Unhealthy", 4: "Very unhealthy"}
ADVICE = {
    1: "Normal outdoor activity is fine.",
    2: "Children, older people and people with asthma or heart disease should limit time outdoors.",
    3: "Avoid heavy outdoor exercise. Sensitive groups should stay indoors.",
    4: "Stay indoors, keep windows closed, wear an N95 mask outside.",
}


@st.cache_data
def load():
    preds = pd.read_csv(f"{DATA}/dashboard_preds.csv", parse_dates=["anchor"])
    preds = preds[preds.split == "test"]
    hourly = pd.read_csv(f"{DATA}/raw_hourly_combined.csv", parse_dates=["timestamp"])
    return preds, hourly


def show_alert(level):
    text = f"{LEVELS[level]}: {ADVICE[level]}"
    {1: st.success, 2: st.info, 3: st.warning, 4: st.error}[level](text)


def forecast_page():
    st.subheader(f"{station}, {anchor:%d %b %Y %H:00}")

    worst = int(now.clf_class.max())
    show_alert(worst)

    cols = st.columns(3)
    for c, hz in zip(cols, HORIZONS):
        r = now.loc[hz]
        c.metric(f"+{hz} h PM2.5 (µg/m³)", f"{r[col]:.0f}")
        c.caption(f"Risk level: {LEVELS[int(r.clf_class)]}")

    h = hourly[hourly.station == station].set_index("timestamp")["PM2.5"]
    past = h.loc[anchor - pd.Timedelta(hours=48):anchor].rename("Measured")
    fc = pd.Series([now.loc[hz, col] for hz in HORIZONS],
                   index=[anchor + pd.Timedelta(hours=hz) for hz in HORIZONS], name="Forecast")
    st.line_chart(pd.concat([past, fc], axis=1))
    st.caption("Hourly PM2.5, last 2 days measured and forecast points for the next 3 days.")

    snap = preds[(preds.anchor == anchor) & (preds.horizon == 24)].set_index("station")
    st.write("All stations, +24 h")
    st.dataframe(pd.DataFrame({"PM2.5 (µg/m³)": snap[col].round(0),
                               "Risk level": snap.clf_class.map(LEVELS)}))


def accuracy_page():
    hz = st.radio("Horizon (h)", HORIZONS, horizontal=True)
    t = preds[(preds.station == station) & (preds.horizon == hz)].sort_values("anchor")
    haz = t[t.actual_class == 4]

    c1, c2, c3 = st.columns(3)
    c1.metric("MAE (µg/m³)", f"{(t[col] - t.actual_pm25).abs().mean():.1f}")
    c2.metric("Risk level accuracy", f"{(t.clf_class == t.actual_class).mean():.0%}")
    c3.metric("Very unhealthy recall", f"{(haz.clf_class == 4).mean():.0%}" if len(haz) else "–")

    st.line_chart(t.set_index("anchor")[["actual_pm25", col]].rename(
        columns={"actual_pm25": "Measured", col: "Forecast"}))

    rows = []
    for (s, h_), g in preds.groupby(["station", "horizon"]):
        g4 = g[g.actual_class == 4]
        rows.append({"Station": s, "Horizon (h)": h_,
                     **{f"MAE {k}": (g[v] - g.actual_pm25).abs().mean() for k, v in MODELS.items()},
                     "Macro-F1": f1_score(g.actual_class, g.clf_class, average="macro",
                                          labels=[1, 2, 3, 4], zero_division=0),
                     "Class-4 recall": (g4.clf_class == 4).mean() if len(g4) else np.nan})
    st.write("All stations, test period")
    st.dataframe(pd.DataFrame(rows).round(2), hide_index=True)


st.set_page_config(page_title="Kathmandu PM2.5", layout="wide")
st.title("Kathmandu PM2.5 forecast")
preds, hourly = load()
stations = sorted(preds.station.unique())

with st.sidebar:
    station = st.selectbox("Station", stations)
    model = st.radio("Model", list(MODELS))
    times = preds[(preds.station == station) & (preds.horizon == 24)].anchor.sort_values()
    day = st.date_input("Date", times.iloc[-1].date(), min_value=times.iloc[0].date(),
                        max_value=times.iloc[-1].date())
    hour = st.number_input("Hour (0–23)", 0, 23, 8)

picked = times[times <= pd.Timestamp(day) + pd.Timedelta(hours=hour)]
anchor = picked.iloc[-1] if len(picked) else times.iloc[0]
col = MODELS[model]
now = preds[(preds.station == station) & (preds.anchor == anchor)].set_index("horizon")

tab1, tab2 = st.tabs(["Forecast", "Accuracy"])
with tab1:
    forecast_page()
with tab2:
    accuracy_page()
