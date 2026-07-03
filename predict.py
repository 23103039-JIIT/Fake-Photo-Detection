import sys

import config
import model as model_module
import utils


def predict_probability(image_path):
    clf = model_module.load_model()

    features = utils.extract_features(image_path).reshape(1, -1)

    proba = clf.predict_proba(features)[0]
    classes = list(clf.classes_)
    screen_index = classes.index(1)
    return float(proba[screen_index])


def main():
    if len(sys.argv) != 2:
        print("Usage: python predict.py <image_path>", file=sys.stderr)
        sys.exit(1)

    image_path = sys.argv[1]

    try:
        probability = predict_probability(image_path)
    except FileNotFoundError as err:
        print(f"[ERROR] {err}", file=sys.stderr)
        sys.exit(1)
    except ValueError as err:
        print(f"[ERROR] {err}", file=sys.stderr)
        sys.exit(1)

    print(f"{probability:.2f}")


if __name__ == "__main__":
    main()
