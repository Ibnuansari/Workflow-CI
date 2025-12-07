import os
import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

mlflow.set_tracking_uri("file://" + os.path.abspath("mlruns"))
mlflow.sklearn.autolog()  

def load_data(train_path, test_path, target):
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    if target not in train.columns or target not in test.columns:
        raise ValueError(f"Target {target} tidak ada di file.")
    X_train = train.drop(columns=[target])
    y_train = train[target].to_numpy().ravel()
    X_test = test.drop(columns=[target])
    y_test = test[target].to_numpy().ravel()
    return X_train, X_test, y_train, y_test

def train_model(train_path, test_path, target):
    X_train, X_test, y_train, y_test = load_data(train_path, test_path, target)
    with mlflow.start_run() as run:
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        mse = mean_squared_error(y_test, preds)
        rmse = mse ** 0.5
        r2 = r2_score(y_test, preds)

        mlflow.log_metric("mse_manual", float(mse))
        mlflow.log_metric("rmse_manual", float(rmse))
        mlflow.log_metric("r2_manual", float(r2))

        # Simpan model ke artifact 'model'
        mlflow.sklearn.log_model(model, artifact_path="model")

    print(f"Run id: {run.info.run_id} — MSE: {mse:.6f}, RMSE: {rmse:.6f}, R2: {r2:.6f}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True)
    parser.add_argument("--test", required=True)
    parser.add_argument("--target", required=True)
    args = parser.parse_args()
    train_model(args.train, args.test, args.target)
