import streamlit as st
import pandas as pd
import joblib
from src.utils import load_json, basic_cleaning, add_features
from src.visualize import plot_trajectory, plot_heatmap
from src.predict import predict_xgb, predict_lstm

st.set_page_config(page_title="AIS ETA Demo", layout="wide")
st.title("تجربة توقع ETA من بيانات AIS")

@st.cache_data
def load_data():
    df = load_json("data/ais_data.json")
    df = basic_cleaning(df)
    df = add_features(df)
    return df

df = load_data()
st.sidebar.write("البيانات")
mmsi_list = sorted(df["MMSI"].unique().tolist())
sel_mmsi = st.sidebar.selectbox("اختر MMSI", mmsi_list[:200])

st.header("معلومات عن السفينة المحددة")
ship = df[df["MMSI"]==sel_mmsi].sort_values("TSTAMP")
st.write(ship.tail(5))

st.header("Trajectory")
if st.button("عرض المسار"):
    plot_trajectory(df, sel_mmsi)

st.header("Traffic Heatmap (sample)")
if st.button("عرض الخريطة"):
    plot_heatmap(df)

st.header("Predict ETA (XGBoost)")
if st.button("تنبؤ XGB"):
    # take last row of selected mmsi
    last = ship.sort_values("TSTAMP").iloc[-1]
    pred = predict_xgb(last, model_path="models/xgb_model.joblib", scaler_path="models/xgb_scaler.joblib")
    st.success(f"Predicted time to arrival (hours): {pred:.2f}")

st.header("Predict ETA (LSTM)")
if st.button("تنبؤ LSTM"):
    # get last 10 observations
    seq_len = 10
    if len(ship) >= seq_len:
        seq = ship.sort_values("TSTAMP").iloc[-seq_len:][["LATITUDE","LONGITUDE","speed_knots"]].values
        pred = predict_lstm(seq, model_path="models/lstm_model.h5", scaler_path="models/lstm_scaler.joblib")
        st.success(f"Predicted time to arrival (hours): {pred:.2f}")
    else:
        st.warning("لا توجد سجلات كافية لهذه السفينة للموديل المتسلسل (LSTM)")
