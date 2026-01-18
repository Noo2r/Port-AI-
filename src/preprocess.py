import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

def build_feature_matrix(df, features=None):
    if features is None:
        features = ["LATITUDE","LONGITUDE","speed_knots","cog_sin","cog_cos","hour","dow"]
    X = df[features].values
    y = df["time_to_arrival"].values
    return X, y, features

def train_test_split_and_scale(X, y, test_size=0.2, random_state=42, scaler_path=None):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    if scaler_path:
        joblib.dump(scaler, scaler_path)
    return X_train_s, X_test_s, y_train, y_test, scaler
