import os

import cv2
import numpy as np
from skimage.feature import local_binary_pattern, graycomatrix, graycoprops

import config


def load_image(image_path):
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not read image (corrupted or unsupported format): {image_path}")
    return image


def preprocess_image(image):
    processed = cv2.resize(image, config.IMAGE_SIZE, interpolation=cv2.INTER_AREA)

    if config.ENABLE_GAUSSIAN_BLUR:
        processed = cv2.GaussianBlur(processed, config.GAUSSIAN_BLUR_KERNEL, 0)

    if config.ENABLE_HIST_EQUALIZATION:
        ycrcb = cv2.cvtColor(processed, cv2.COLOR_BGR2YCrCb)
        ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])
        processed = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)

    if config.ENABLE_GRAYSCALE:
        gray = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)
        processed = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    if config.ENABLE_NORMALIZATION:
        processed = processed.astype(np.float32) / 255.0

    return processed


def _to_gray_uint8(image):
    if image.dtype != np.uint8:
        tmp = np.clip(image * 255.0, 0, 255).astype(np.uint8)
    else:
        tmp = image
    return cv2.cvtColor(tmp, cv2.COLOR_BGR2GRAY)


def extract_fft_features(image):
    gray = _to_gray_uint8(image).astype(np.float32)

    fft = np.fft.fft2(gray)
    fft_shifted = np.fft.fftshift(fft)
    magnitude = np.abs(fft_shifted)

    h, w = magnitude.shape
    cy, cx = h // 2, w // 2

    yy, xx = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    radius = config.FFT_LOW_FREQ_RADIUS_RATIO * min(h, w)
    low_mask = dist_from_center <= radius

    low_energy = float(magnitude[low_mask].sum())
    high_energy = float(magnitude[~low_mask].sum())
    high_low_ratio = high_energy / (low_energy + 1.0)

    flat = magnitude.flatten()
    total = flat.sum()
    if total > 0:
        prob = flat / total
        spectral_entropy = float(-np.sum(prob * np.log2(prob + 1e-12)))
    else:
        spectral_entropy = 0.0

    high_vals = np.sort(magnitude[~low_mask].flatten())[::-1]
    peaks = high_vals[: config.FFT_NUM_PEAKS]
    if len(peaks) < config.FFT_NUM_PEAKS:
        peaks = np.pad(peaks, (0, config.FFT_NUM_PEAKS - len(peaks)))

    features = [low_energy, high_energy, high_low_ratio, spectral_entropy]
    features.extend(float(p) for p in peaks)
    return features


def extract_lbp_features(image):
    gray = _to_gray_uint8(image)

    lbp = local_binary_pattern(
        gray, config.LBP_N_POINTS, config.LBP_RADIUS, method=config.LBP_METHOD
    )

    n_bins = config.LBP_N_POINTS + 2
    hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins))
    hist = hist.astype(np.float32)
    hist /= hist.sum() + 1e-12

    texture_entropy = float(-np.sum(hist * np.log2(hist + 1e-12)))
    texture_variance = float(np.var(lbp))

    features = list(hist)
    features.append(texture_entropy)
    features.append(texture_variance)
    return features


def extract_glcm_features(image):
    gray = _to_gray_uint8(image)

    quantised = (gray.astype(np.float32) / 256.0 * config.GLCM_LEVELS).astype(np.uint8)
    quantised = np.clip(quantised, 0, config.GLCM_LEVELS - 1)

    angles_rad = [np.deg2rad(a) for a in config.GLCM_ANGLES]
    glcm = graycomatrix(
        quantised,
        distances=config.GLCM_DISTANCES,
        angles=angles_rad,
        levels=config.GLCM_LEVELS,
        symmetric=True,
        normed=True,
    )

    features = []
    for prop in ("contrast", "correlation", "energy", "homogeneity"):
        value = float(np.mean(graycoprops(glcm, prop)))
        features.append(value)
    return features


def extract_edge_features(image):
    gray = _to_gray_uint8(image)

    edges = cv2.Canny(gray, config.CANNY_LOW_THRESHOLD, config.CANNY_HIGH_THRESHOLD)
    num_edge_pixels = int(np.count_nonzero(edges))
    edge_density = float(num_edge_pixels) / edges.size

    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    sobel_magnitude = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
    avg_sobel_strength = float(np.mean(sobel_magnitude))

    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    laplacian_variance = float(np.var(laplacian))

    return [edge_density, float(num_edge_pixels), avg_sobel_strength, laplacian_variance]


def extract_color_features(image):
    if image.dtype != np.uint8:
        bgr = np.clip(image * 255.0, 0, 255).astype(np.uint8)
    else:
        bgr = image

    features = []

    for c in range(3):
        channel = bgr[:, :, c].astype(np.float32)
        features.append(float(np.mean(channel)))
        features.append(float(np.std(channel)))

    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    for c in range(3):
        channel = hsv[:, :, c].astype(np.float32)
        features.append(float(np.mean(channel)))
        features.append(float(np.std(channel)))

    brightness = float(np.mean(hsv[:, :, 2]))    
    saturation = float(np.mean(hsv[:, :, 1]))     
    color_variance = float(np.var(bgr.astype(np.float32)))

    features.extend([brightness, saturation, color_variance])
    return features


def extract_reflection_features(image):
    gray = _to_gray_uint8(image)

    bright_mask = gray >= config.REFLECTION_BRIGHTNESS_THRESHOLD
    reflection_percentage = float(np.count_nonzero(bright_mask)) / gray.size

    bright_uint8 = (bright_mask.astype(np.uint8)) * 255
    num_labels, _ = cv2.connectedComponents(bright_uint8)
    num_bright_spots = int(max(num_labels - 1, 0)) 

    max_brightness = float(np.max(gray))
    if np.any(bright_mask):
        mean_bright_region_intensity = float(np.mean(gray[bright_mask]))
    else:
        mean_bright_region_intensity = 0.0

    return [
        reflection_percentage,
        float(num_bright_spots),
        max_brightness,
        mean_bright_region_intensity,
    ]


def extract_features(image_path):
    image = load_image(image_path)
    processed = preprocess_image(image)

    feature_vector = []

    if config.ENABLE_FFT:
        feature_vector.extend(extract_fft_features(processed))
    if config.ENABLE_LBP:
        feature_vector.extend(extract_lbp_features(processed))
    if config.ENABLE_GLCM:
        feature_vector.extend(extract_glcm_features(processed))
    if config.ENABLE_EDGE:
        feature_vector.extend(extract_edge_features(processed))
    if config.ENABLE_COLOR:
        feature_vector.extend(extract_color_features(processed))
    if config.ENABLE_REFLECTION:
        feature_vector.extend(extract_reflection_features(processed))

    vector = np.asarray(feature_vector, dtype=np.float32)
    vector = np.nan_to_num(vector, nan=0.0, posinf=0.0, neginf=0.0)
    return vector


def _list_images(folder):
    if not os.path.isdir(folder):
        return []
    files = []
    for name in sorted(os.listdir(folder)):
        if name.lower().endswith(config.SUPPORTED_EXTENSIONS):
            files.append(os.path.join(folder, name))
    return files


def load_dataset():
    real_dir = os.path.join(config.REAL_DIR)
    screen_dir = os.path.join(config.SCREEN_DIR)

    for d in (real_dir, screen_dir):
        if not os.path.isdir(d):
            raise FileNotFoundError(f"Dataset folder missing: {d}")

    samples = []
    for path in _list_images(real_dir):
        samples.append((path, 0))
    for path in _list_images(screen_dir):
        samples.append((path, 1))

    if not samples:
        raise ValueError(f"No images found. Add .jpg/.jpeg/.png files to '{real_dir}' and '{screen_dir}'.")

    X, y = [], []
    for path, label in samples:
        try:
            X.append(extract_features(path))
            y.append(label)
        except (ValueError, FileNotFoundError) as err:
            print(f"[WARN] Skipping '{path}': {err}")

    if not X:
        raise ValueError("All images failed to load. Check your dataset files.")

    return np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.int64)
