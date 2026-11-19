# PM2.5 dashboard. Run gnn_lstm.ipynb first (it writes dashboard_preds.csv), then: streamlit run dashboard.py
# forecast = hourly PM2.5, risk level = class of 24 h average
import pandas as pd
import streamlit as st

DATA = "processed"
HORIZONS = [24, 48, 72]
MODELS = {"LSTM": "lstm_pm25", "GNN-LSTM": "gnn_pm25"}
WHEN = {24: "Tomorrow", 48: "In 2 days", 72: "In 3 days"}
LEVELS = {1: "Good", 2: "Unhealthy for sensitive groups", 3: "Unhealthy", 4: "Very unhealthy"}
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


def sort_table(df, sort_col):
    # plain table, click the sort_col header to sort
    head = "".join(f'<th class="sort" onclick="sortBy({i})">{c}<span class="icon">swap_vert</span></th>' if c == sort_col
                   else f"<th>{c}</th>" for i, c in enumerate(df.columns))
    body = "".join("<tr>" + "".join(f"<td>{v}</td>" for v in row) + "</tr>" for row in df.itertuples(index=False))
    html = f"""
    <link href="https://fonts.googleapis.com/css2?family=Source+Sans+3:wght@400;600&display=swap" rel="stylesheet">
    <link href="https://fonts.googleapis.com/css2?family=Material+Symbols+Rounded:opsz,wght,FILL,GRAD@20,400,0,0" rel="stylesheet">
    <style>
      body {{margin: 0;}}
      table {{border-collapse: collapse; width: 100%; font-family: "Source Sans 3", sans-serif; font-size: 14px; color: #31333f;}}
      th {{text-align: left; font-weight: 400; color: #808495; padding: 8px; border-bottom: 1px solid #e6e9ef;
           background: #f8f9fb; white-space: nowrap;}}
      th.sort {{cursor: pointer;}}
      td {{padding: 8px; border-bottom: 1px solid #f0f2f6;}}
      .icon {{font-family: "Material Symbols Rounded"; font-size: 16px; vertical-align: -3px; margin-left: 4px; color: #b0b3bd;}}
      .icon.on {{color: #31333f;}}
    </style>
    <table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>
    <script>
      let asc = false;
      function sortBy(i) {{
        asc = !asc;
        const tb = document.querySelector("tbody");
        const rows = [...tb.rows].sort((a, b) => {{
          const x = a.cells[i].innerText, y = b.cells[i].innerText;
          const d = isNaN(x) || isNaN(y) ? x.localeCompare(y) : x - y;
          return asc ? d : -d;
        }});
        rows.forEach(r => tb.appendChild(r));
        const icon = document.querySelector("th .icon");
        icon.textContent = asc ? "arrow_upward" : "arrow_downward";
        icon.classList.add("on");
      }}
    </script>"""
    st.iframe(html, height=min(45 + 37 * len(df), 500))


def show_alert(level):
    text = f"Worst air in the next 3 days: **{LEVELS[level]}**. {ADVICE[level]}"
    {1: st.success, 2: st.info, 3: st.warning, 4: st.error}[level](text)


def forecast_page():
    st.subheader(f"{station}, {anchor:%A %d %B %Y, %H:00}")

    show_alert(int(now.clf_class.max()))

    cols = st.columns(3)
    for c, hz in zip(cols, HORIZONS):
        r = now.loc[hz]
        c.metric(f"{WHEN[hz]} ({anchor + pd.Timedelta(hours=hz):%a %d %b})", f"{r[col]:.0f} µg/m³")
        c.write(LEVELS[int(r.clf_class)])

    st.write("Last 2 days and next 3 days")
    h = hourly[hourly.station == station].set_index("timestamp")["PM2.5"]
    past = h.loc[anchor - pd.Timedelta(hours=48):anchor].rename("Measured")
    fc = pd.Series([now.loc[hz, col] for hz in HORIZONS],
                   index=[anchor + pd.Timedelta(hours=hz) for hz in HORIZONS], name="Forecast")
    st.line_chart(pd.concat([past, fc], axis=1), x_label="", y_label="PM2.5 (µg/m³)")

    snap = preds[(preds.anchor == anchor) & (preds.horizon == 24)].set_index("station")
    st.write("Tomorrow at all stations")
    sort_table(pd.DataFrame({"Station": snap.index, "PM2.5 (µg/m³)": snap[col].round().astype(int),
                             "Air quality": snap.clf_class.map(LEVELS)}), sort_col="PM2.5 (µg/m³)")


def accuracy_page():
    st.write("Forecasts compared with measured values on test data (not used in training).")
    hz = st.radio("Forecast for", HORIZONS, format_func=WHEN.get, horizontal=True)
    t = preds[(preds.station == station) & (preds.horizon == hz)].sort_values("anchor")
    haz = t[t.actual_class == 4]

    c1, c2, c3 = st.columns(3)
    c1.metric("Average error", f"{(t[col] - t.actual_pm25).abs().mean():.1f} µg/m³")
    c2.metric("Air quality level correct", f"{(t.clf_class == t.actual_class).mean():.0%}")
    c3.metric("Very unhealthy times caught", f"{(haz.clf_class == 4).mean():.0%}" if len(haz) else "–")

    st.line_chart(t.set_index("anchor")[["actual_pm25", col]].rename(
        columns={"actual_pm25": "Measured", col: "Forecast"}), x_label="", y_label="PM2.5 (µg/m³)")


st.set_page_config(page_title="Kathmandu Air Quality", layout="wide")
st.title("Kathmandu Air Quality Forecast")
st.caption("PM2.5 is fine dust in the air. Higher is worse for health.")
preds, hourly = load()
stations = sorted(preds.station.unique())

with st.sidebar:
    station = st.selectbox("Station", stations)
    times = preds[(preds.station == station) & (preds.horizon == 24)].anchor.sort_values()
    day = st.date_input("Date", times.iloc[-1].date(), min_value=times.iloc[0].date(),
                        max_value=times.iloc[-1].date())
    hour = st.slider("Time of day", 0, 23, 8, format="%d:00")
    model = st.radio("Forecast model", list(MODELS))

picked = times[times <= pd.Timestamp(day) + pd.Timedelta(hours=hour)]
anchor = picked.iloc[-1] if len(picked) else times.iloc[0]
col = MODELS[model]
now = preds[(preds.station == station) & (preds.anchor == anchor)].set_index("horizon")

tab1, tab2 = st.tabs(["Forecast", "Forecast Accuracy"])
with tab1:
    forecast_page()
with tab2:
    accuracy_page()
