import argparse
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error
import joblib
from utils import load_json, basic_cleaning, add_features
from preprocess import build_feature_matrix, train_test_split_and_scale


def main(
    path_json="./data/ais_data.json",
    model_out="models/xgb_model.joblib",
    scaler_out="models/xgb_scaler.joblib"
):

    print("🔹 Loading JSON...")
    df = load_json(path_json)
    print("➡ After loading:", df.shape)
    print(df["ETA"].head(20))
    # -------------------------
    # STEP 1: Basic Cleaning
    # -------------------------
    df = basic_cleaning(df)
    print("➡ After basic_cleaning:", df.shape)

    # Inspect ETA after cleaning
    if "ETA" in df.columns:
        print("ETA sample after cleaning:")
        print(df["ETA"].head(10))
        print("Missing ETA count:", df["ETA"].isna().sum())
    else:
        print("⚠ WARNING: ETA column missing from dataframe!")

    # -------------------------
    # STEP 2: Add Additional Features
    # -------------------------
    df = add_features(df)
    print("➡ After add_features:", df.shape)

    # -------------------------
    # STEP 3: Build Feature Matrix
    # -------------------------
    print("🔹 Building X, y ...")
    X, y, features = build_feature_matrix(df)
    print("➡ X shape:", X.shape)
    print("➡ y length:", len(y))

    # If dataset is empty → STOP
    if len(X) == 0 or len(y) == 0:
        raise ValueError(
            "\n❌ ERROR: Feature matrix is EMPTY.\n"
            "This means cleaning removed all rows. "
            "Check date parsing, ETA format, or NaN rows.\n"
        )

    # -------------------------
    # STEP 4: Train/Test Split + Scaling
    # -------------------------
    print("🔹 Splitting train/test...")
    X_train, X_test, y_train, y_test, scaler = train_test_split_and_scale(
        X, y, scaler_path=scaler_out
    )

    print("Train size:", X_train.shape, "Test size:", X_test.shape)

    # -------------------------
    # STEP 5: Train XGBoost Model
    # -------------------------
    print("🔹 Training XGBRegressor...")
    model = XGBRegressor(
        n_estimators=400,
        learning_rate=0.05,
        max_depth=8,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1
    )

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_test, y_test)],
        verbose=10
    )

    # -------------------------
    # STEP 6: Evaluate
    # -------------------------
    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    print("\n✅ XGB MAE (hours):", mae)

    # -------------------------
    # STEP 7: Save Model
    # -------------------------
    joblib.dump(model, model_out)
    print("💾 Model saved to:", model_out)


if __name__ == "__main__":
    main()
