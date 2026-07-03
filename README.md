# Fake Photo Detection (OpenCV + Classical ML)

Given a single image, decide whether it is a **real photograph (0)** or a **photograph of a screen (1)** — a phone, laptop, monitor, tablet, TV, etc.

```bash
python predict.py image.jpg
```

The output is a single probability between 0 and 1, where `0 = definitely real` and `1 = definitely screen`.
We extract handcrafted image-processing features with OpenCV and feed them to a lightweight scikit-learn classifier.

---

## Installation

```bash
pip install -r requirements.txt
```

---

## Training

```bash
python train.py
```

This loads every image, extracts features, trains the classifier, prints an evaluation report, and writes the model to `saved_model/model.joblib`.

---

## Prediction

```bash
python predict.py path/to/image.jpg
```

Prints only the probability.

---

## How the features work

| Features | What it measures | Why it helps |
| --- | --- | --- |
| **Frequency (FFT)** | Image patterns and high-frequency details | Screens create repeating pixel patterns (moire) that are usually not present in real-world photos |
| **Texture (LBP)** | Small texture patterns | Screen pixels form regular, repeating textures, while natural objects have more varied textures |
| **Texture (GLCM)** | Relationship between neighboring pixels | Screen images have a more regular pixel arrangement than natural scenes |
| **Edges** | Number and strength of edges | Pixel grids, screen borders, and scan lines create different edge patterns than real objects |
| **Colour** | Brightness, colors, and saturation | Screens often have more uniform brightness and different color distributions because they emit light |
| **Reflection/Glare** | Bright spots and reflections | Glass screens frequently produce glare and reflections that are uncommon in normal photographs |

All enabled features are concatenated (in a fixed order) into one NumPy vector.

---

## Latency per image

- Feature extraction takes **~25-26 ms per image** on a standard CPU.
- Model prediction takes **less than 3 ms**.
- Total processing time is **~30 ms per image**.

---

## Limitations

- Performance depends on the quality and diversity of the training dataset.
- Very high-quality screen photos may look similar to real images, making them harder to detect.
- Heavy image editing or compression can remove the patterns used for detection.
- Some detection parameters are fixed and may not work equally well on all datasets.

---

## Future work

- Improve accuracy using better hyperparameter tuning and cross-validation.
- Add more image features for stronger detection.
- Balance and augment the training dataset.
- Remove less useful features to make predictions even faster.
- Add a CNN-based model.
