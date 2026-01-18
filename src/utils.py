import json
import pandas as pd
import numpy as np
from datetime import datetime
import joblib

def load_json(path):
    with open(path, "r") as f:
        data = json.load(f)
    return pd.DataFrame(data)

def basic_cleaning(df):
    df = df.copy()

    # Keep only available useful cols
    needed = ["MMSI","TSTAMP","ETA","DEST","LATITUDE","LONGITUDE","SOG","COG","HEADING","NAME"]
    existing = [c for c in needed if c in df.columns]
    df = df[existing].copy()

    # Convert TSTAMP first
    df["TSTAMP"] = pd.to_datetime(df["TSTAMP"], errors="coerce")

    # --- FIX ETA (MM-DD HH:MM) parsing ---
    def parse_eta(row):
        raw = str(row["ETA"])
        if "-" not in raw:
            return pd.NaT

        try:
            # expected format: MM-DD HH:MM
            month_day, hm = raw.split()
            mm, dd = month_day.split("-")
            hh, mi = hm.split(":")

            # Reject invalid zero or out-of-range values
            mm, dd, hh, mi = int(mm), int(dd), int(hh), int(mi)
            if mm == 0 or dd == 0 or hh > 23 or mi > 59:
                return pd.NaT

            # Infer year from timestamp
            ts = row["TSTAMP"]
            if pd.isna(ts):
                return pd.NaT
            year = ts.year

            eta = datetime(year, mm, dd, hh, mi)

            # If ETA before current timestamp → assume next year
            if eta < ts:
                eta = datetime(year + 1, mm, dd, hh, mi)

            return eta
        except:
            return pd.NaT

    df["ETA"] = df.apply(parse_eta, axis=1)

    # Drop invalid rows
    df = df.dropna(subset=["TSTAMP", "ETA", "LATITUDE", "LONGITUDE", "SOG"])

    # Compute label: time_to_arrival in hours
    df["time_to_arrival"] = (df["ETA"] - df["TSTAMP"]).dt.total_seconds() / 3600.0

    # Keep only positive arrival times
    df = df[df["time_to_arrival"] > 0]

    return df

def add_features(df):
    df = df.copy()

    df["speed_knots"] = df["SOG"]

    # --- FIX: ensure COG is numeric ---
    df["COG"] = pd.to_numeric(df["COG"], errors="coerce")  # force strings → NaN
    df["COG"] = df["COG"].fillna(0)                        # replace invalid COG with 0

    # encode course as sin/cos
    df["cog_sin"] = np.sin(np.deg2rad(df["COG"].astype(float)))
    df["cog_cos"] = np.cos(np.deg2rad(df["COG"].astype(float)))

    # time features
    df["hour"] = df["TSTAMP"].dt.hour
    df["dow"] = df["TSTAMP"].dt.dayofweek

    return df


def save_model(obj, path):
    # joblib for sklearn/xgboost; tensorflow handled separately
    joblib.dump(obj, path)

def load_model(path):
    return joblib.load(path)
