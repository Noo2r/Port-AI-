import argparse
import joblib
import numpy as np
from utils import load_json, basic_cleaning, add_features
from tensorflow.keras.models import load_model

def predict_xgb(sample_row, model_path="models/xgb_model.joblib", scaler_path="models/xgb_scaler.joblib"):
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    # features order same as train
    features = ["LATITUDE","LONGITUDE","speed_knots","cog_sin","cog_cos","hour","dow"]
    X = np.array([sample_row[f] for f in features]).reshape(1,-1)
    Xs = scaler.transform(X)
    pred_hours = model.predict(Xs)[0]
    return pred_hours

def predict_lstm(seq_array, model_path="models/lstm_model.h5", scaler_path="models/lstm_scaler.joblib"):
    scaler = joblib.load(scaler_path)
    model = load_model(model_path)
    # seq_array shape: (seq_len, nfeat)
    seq_len, nfeat = seq_array.shape
    X2d = seq_array.reshape((1*seq_len, nfeat))
    X2d_s = scaler.transform(X2d)
    X_s = X2d_s.reshape((1, seq_len, nfeat))
    pred = model.predict(X_s)[0][0]
    return pred

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["xgb","lstm"], default="xgb")
    parser.add_argument("--model", default="models/xgb_model.joblib")
    parser.add_argument("--scaler", default="models/xgb_scaler.joblib")
    parser.add_argument("--sample_idx", type=int, default=0)
    args = parser.parse_args()

    df = load_json("data/ais_data.json")
    df = basic_cleaning(df)
    df = add_features(df)
    df = df.reset_index(drop=True)

    if args.mode == "xgb":
        row = df.iloc[args.sample_idx]
        pred = predict_xgb(row, model_path=args.model, scaler_path=args.scaler)
        print(f"Predicted time to arrival (hours): {pred:.2f}")
    else:
        # prepare sequence of last 10 observations for a ship
        seq_len = 10
        # find first MMSI with enough records
        grouped = df.groupby("MMSI")
        for mmsi, g in grouped:
            if len(g) >= seq_len:
                sample = g.sort_values("TSTAMP").iloc[-seq_len:][["LATITUDE","LONGITUDE","speed_knots"]].values
                pred = predict_lstm(sample, model_path=args.model, scaler_path=args.scaler)
                print(f"MMSI {mmsi} Predicted time to arrival (hours): {pred:.2f}")
                break
