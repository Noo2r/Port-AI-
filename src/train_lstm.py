import numpy as np
import joblib
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from utils import load_json, basic_cleaning, add_features

# we will create sequences per MMSI
def create_sequences(df, sequence_length=10):
    X_list, y_list = [], []
    df = df.sort_values(["MMSI","TSTAMP"])
    grouped = df.groupby("MMSI")
    for _, g in grouped:
        g = g.reset_index(drop=True)
        feat = g[["LATITUDE","LONGITUDE","speed_knots"]].values
        labels = g["time_to_arrival"].values
        if len(g) <= sequence_length:
            continue
        for i in range(len(g)-sequence_length):
            X_list.append(feat[i:i+sequence_length])
            y_list.append(labels[i+sequence_length])
    if not X_list:
        return None, None
    X = np.array(X_list)
    y = np.array(y_list)
    return X, y

def main(path_json="data/ais_data.json", model_out="models/lstm_model.h5"):
    df = load_json(path_json)
    df = basic_cleaning(df)
    df = add_features(df)  # speed_knots created here
    X, y = create_sequences(df, sequence_length=10)
    if X is None:
        print("Not enough sequence data.")
        return

    # Scale features per channel (simple scaling)
    # reshape to 2D for scaler then back
    nsamples, seq_len, nfeat = X.shape
    X2d = X.reshape((nsamples*seq_len, nfeat))
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X2d_s = scaler.fit_transform(X2d)
    X_s = X2d_s.reshape((nsamples, seq_len, nfeat))
    joblib.dump(scaler, "models/lstm_scaler.joblib")

    # split
    from sklearn.model_selection import train_test_split
    X_train, X_val, y_train, y_val = train_test_split(X_s, y, test_size=0.2, random_state=42)

    model = Sequential()
    model.add(LSTM(128, input_shape=(seq_len, nfeat)))
    model.add(Dense(64, activation="relu"))
    model.add(Dense(1))

    model.compile(optimizer="adam", loss="mae")
    es = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
    mc = ModelCheckpoint(model_out, monitor="val_loss", save_best_only=True)

    model.fit(X_train, y_train, validation_data=(X_val,y_val), epochs=30, batch_size=64, callbacks=[es, mc])
    print("Saved LSTM model to", model_out)

if __name__ == "__main__":
    main()
