import os
import yaml
import joblib
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, f1_score, r2_score, mean_squared_error


cfg = yaml.safe_load(open("config/config.yaml", encoding="utf-8"))

df = pd.read_csv(cfg["data_path"])
df = df.drop(columns=[c for c in cfg.get("drop_cols", []) if c in df.columns])
df = df.dropna(subset=[cfg["target_col"]])

X = df.drop(columns=[cfg["target_col"]])
y = df[cfg["target_col"]]

num_cols = X.select_dtypes(include="number").columns
cat_cols = X.select_dtypes(exclude="number").columns

preprocess = make_column_transformer(
    (SimpleImputer(strategy="median"), num_cols),
    (make_pipeline(SimpleImputer(strategy="most_frequent"),
                   OneHotEncoder(handle_unknown="ignore")), cat_cols)
)

task = cfg.get("task_type", "classification")
if task == "regression":
    model = RandomForestRegressor(random_state=cfg.get("random_state", 42))
else:
    model = RandomForestClassifier(random_state=cfg.get("random_state", 42))

pipe = make_pipeline(preprocess, model)

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=cfg.get("test_size", 0.2),
    random_state=cfg.get("random_state", 42)
)

pipe.fit(X_train, y_train)
pred = pipe.predict(X_test)

if task == "regression":
    rmse = np.sqrt(mean_squared_error(y_test, pred))
    r2 = r2_score(y_test, pred)
    metrics = f"rmse={rmse}\nr2={r2}"
else:
    acc = accuracy_score(y_test, pred)
    f1 = f1_score(y_test, pred, average="weighted")
    metrics = f"accuracy={acc}\nf1_score={f1}"

feature_info = {}
for col in X.columns:
    if pd.api.types.is_numeric_dtype(X[col]):
        feature_info[col] = {
            "type": "numeric",
            "default": float(X[col].median()) if X[col].notna().any() else 0.0
        }
    else:
        opts = X[col].dropna().astype(str).unique().tolist()[:30]
        feature_info[col] = {
            "type": "categorical",
            "options": opts,
            "default": opts[0] if opts else ""
        }

bundle = {
    "model": pipe,
    "feature_info": feature_info,
    "target_col": cfg["target_col"],
    "task_type": task
}

os.makedirs("models", exist_ok=True)
joblib.dump(bundle, cfg.get("model_path", "models/model.pkl"))

with open(cfg.get("metrics_path", "metrics.txt"), "w", encoding="utf-8") as f:
    f.write(metrics)
