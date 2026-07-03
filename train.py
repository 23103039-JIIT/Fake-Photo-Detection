import sys

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

import config
import model as model_module
import utils


def _predict_proba_screen(fitted_model, X):
    proba = fitted_model.predict_proba(X)
    classes = list(fitted_model.classes_)
    screen_index = classes.index(1)
    return proba[:, screen_index]


def main():
    print("Loading dataset and extracting features...")
    try:
        X, y = utils.load_dataset()
    except (FileNotFoundError, ValueError) as err:
        print(f"[ERROR] {err}")
        sys.exit(1)

    n_real = int(np.sum(y == 0))
    n_screen = int(np.sum(y == 1))
    print(f"Loaded {len(y)} images  ->  real: {n_real}, screen: {n_screen}")
    print(f"Feature vector length: {X.shape[1]}")

    if n_real == 0 or n_screen == 0:
        print("[ERROR] Need images in BOTH 'real' and 'screen' folders to train.")
        sys.exit(1)

    stratify = y if (n_real >= 2 and n_screen >= 2) else None
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=config.TEST_SPLIT_RATIO,
        random_state=config.RANDOM_SEED,
        stratify=stratify,
    )

    print(f"Training classifier: {config.CLASSIFIER}")
    clf = model_module.build_model()
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    y_scores = _predict_proba_screen(clf, X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])

    if len(np.unique(y_test)) == 2:
        roc_auc = roc_auc_score(y_test, y_scores)
    else:
        roc_auc = float("nan")

    print("\n================ Evaluation ================")
    print(f"Test samples : {len(y_test)}")
    print(f"Accuracy     : {accuracy:.4f}")
    print(f"Precision    : {precision:.4f}")
    print(f"Recall       : {recall:.4f}")
    print(f"F1 Score     : {f1:.4f}")
    print(f"ROC AUC      : {roc_auc:.4f}")
    print("\nConfusion Matrix (rows = actual, cols = predicted):")
    print("               pred_real   pred_screen")
    print(f"actual_real     {cm[0, 0]:>7d}     {cm[0, 1]:>7d}")
    print(f"actual_screen   {cm[1, 0]:>7d}     {cm[1, 1]:>7d}")
    print("==============================================\n")

    saved_path = model_module.save_model(clf)
    print(f"Model saved to: {saved_path}")
    print("Done. You can now run:  python predict.py <image>")


if __name__ == "__main__":
    main()
