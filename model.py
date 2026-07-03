import os

import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

import config


def build_model():
    name = config.CLASSIFIER

    if name == "RandomForest":
        return RandomForestClassifier(
            n_estimators=config.RF_N_ESTIMATORS,
            max_depth=config.RF_MAX_DEPTH,
            random_state=config.RF_RANDOM_STATE,
            n_jobs=-1,
        )

    if name == "LogisticRegression":
        return Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        max_iter=config.LR_MAX_ITER,
                        C=config.LR_C,
                    ),
                ),
            ]
        )

    if name == "SVM":
        return Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "clf",
                    SVC(
                        C=config.SVM_C,
                        kernel=config.SVM_KERNEL,
                        gamma=config.SVM_GAMMA,
                        probability=True,  # required so we can output 0..1
                    ),
                ),
            ]
        )

    raise ValueError(f"Unknown CLASSIFIER '{name}'. ")


def _model_path():
    return os.path.join(config.SAVED_MODEL_DIR, config.MODEL_FILENAME)


def save_model(model):
    os.makedirs(config.SAVED_MODEL_DIR, exist_ok=True)
    path = _model_path()
    joblib.dump(model, path)
    return path


def load_model():
    path = _model_path()
    if not os.path.isfile(path):
        raise FileNotFoundError(f"No trained model found at '{path}'")
    return joblib.load(path)
