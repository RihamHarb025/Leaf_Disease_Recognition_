import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from skimage.feature import graycomatrix, graycoprops


# ---------------------------
# basic image quality checks
# ---------------------------
def check_brightness(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return np.mean(gray)


def check_contrast(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return np.std(gray)


# ---------------------------
# preprocessing
# ---------------------------
def resize_image(image, size=(512, 512)):
    return cv2.resize(image, size)


def denoise_light(image):
    return cv2.bilateralFilter(image, d=7, sigmaColor=50, sigmaSpace=50)


def enhance_contrast(image):
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_enhanced = clahe.apply(l)

    merged = cv2.merge((l_enhanced, a, b))
    enhanced = cv2.cvtColor(merged, cv2.COLOR_LAB2RGB)
    return enhanced


def preprocess_image(image):
    image = resize_image(image, (512, 512))
    image = denoise_light(image)
    image = enhance_contrast(image)
    return image


# ---------------------------
# morphology / mask utilities
# ---------------------------
def clean_mask(mask, kernel_size=5, open_iter=1, close_iter=1):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))

    cleaned = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=open_iter)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel, iterations=close_iter)

    return cleaned


def keep_largest_component(mask):
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

    if num_labels <= 1:
        return mask

    largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    largest_mask = np.zeros_like(mask)
    largest_mask[labels == largest_label] = 255

    return largest_mask


def remove_small_regions(mask, min_area=80):
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    cleaned = np.zeros_like(mask)

    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_area:
            cleaned[labels == i] = 255

    return cleaned


# ---------------------------
# leaf segmentation
# ---------------------------
def segment_leaf(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

    lower_green = np.array([25, 40, 40])
    upper_green = np.array([95, 255, 255])

    raw_mask = cv2.inRange(hsv, lower_green, upper_green)
    cleaned_mask = clean_mask(raw_mask, kernel_size=5, open_iter=1, close_iter=2)
    largest_mask = keep_largest_component(cleaned_mask)

    contours, _ = cv2.findContours(largest_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled_mask = np.zeros_like(largest_mask)

    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        cv2.drawContours(filled_mask, [largest_contour], -1, 255, thickness=cv2.FILLED)

    return filled_mask


# ---------------------------
# disease segmentation
# ---------------------------
def segment_disease_simple(image, leaf_mask):
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    h, s, v = cv2.split(hsv)

    # likely yellow / brown lesion colors
    disease = (((h >= 8) & (h <= 30)) & (s >= 60) & (v >= 50)).astype(np.uint8) * 255

    # keep disease only inside the leaf
    disease = cv2.bitwise_and(disease, leaf_mask)

    # clean small noise
    disease = clean_mask(disease, kernel_size=3, open_iter=1, close_iter=1)
    disease = remove_small_regions(disease, min_area=80)

    return disease


# ---------------------------
# region extraction
# ---------------------------
def extract_regions(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    num_labels, labels_im = cv2.connectedComponents(mask)

    return {
        "contours": contours,
        "num_labels": num_labels,
        "labels_image": labels_im
    }


# ---------------------------
# severity calculation
# ---------------------------
def calculate_severity(leaf_mask, disease_mask):
    leaf_area = np.count_nonzero(leaf_mask)
    disease_area = np.count_nonzero(disease_mask)

    if leaf_area == 0:
        severity = 0.0
    else:
        severity = (disease_area / leaf_area) * 100

    return leaf_area, disease_area, severity


# ---------------------------
# helper functions for improved features
# ---------------------------
def region_stats(channel, mask):
    values = channel[mask > 0]
    if values.size == 0:
        return 0.0, 0.0
    return float(np.mean(values)), float(np.std(values))


def compute_largest_spot_shape_features(disease_mask):
    contours, _ = cv2.findContours(disease_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return 0.0, 0.0, 0.0

    largest_contour = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest_contour)
    perimeter = cv2.arcLength(largest_contour, True)

    compactness = 0.0
    if perimeter > 0:
        compactness = float((4 * np.pi * area) / (perimeter ** 2))

    x, y, w, h = cv2.boundingRect(largest_contour)
    bbox_area = float(w * h) if w > 0 and h > 0 else 1.0
    extent = float(area / bbox_area)

    aspect_ratio = float(w / h) if h > 0 else 0.0

    return compactness, extent, aspect_ratio


def compute_glcm_features(image_gray, mask):
    # Use only leaf region; non-leaf becomes 0
    region = image_gray.copy()
    region[mask == 0] = 0

    # Reduce gray levels from 0..255 to 0..7 for stable GLCM
    region_quantized = (region / 32).astype(np.uint8)

    glcm = graycomatrix(
        region_quantized,
        distances=[1],
        angles=[0],
        levels=8,
        symmetric=True,
        normed=True
    )

    contrast = float(graycoprops(glcm, 'contrast')[0, 0])
    homogeneity = float(graycoprops(glcm, 'homogeneity')[0, 0])
    energy = float(graycoprops(glcm, 'energy')[0, 0])
    correlation = float(graycoprops(glcm, 'correlation')[0, 0])

    return contrast, homogeneity, energy, correlation


def severity_label(severity):
    if severity < 5:
        return "Minimal"
    elif severity < 20:
        return "Mild"
    elif severity < 40:
        return "Moderate"
    else:
        return "Severe"


# ---------------------------
# improved feature extraction
# ---------------------------
def extract_features(image, leaf_mask, disease_mask):
    leaf_area = np.count_nonzero(leaf_mask)
    disease_area = np.count_nonzero(disease_mask)

    if leaf_area == 0:
        severity = 0.0
    else:
        severity = (disease_area / leaf_area) * 100

    # connected components on disease mask
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(disease_mask, connectivity=8)

    # exclude background
    num_spots = max(0, num_labels - 1)

    if num_spots > 0:
        spot_areas = stats[1:, cv2.CC_STAT_AREA].astype(np.float32)
        largest_spot_area = int(np.max(spot_areas))
        mean_spot_area = float(np.mean(spot_areas))
        std_spot_area = float(np.std(spot_areas))
    else:
        largest_spot_area = 0
        mean_spot_area = 0.0
        std_spot_area = 0.0

    largest_spot_ratio = float(largest_spot_area / leaf_area) if leaf_area > 0 else 0.0
    spot_density = float(num_spots / leaf_area) if leaf_area > 0 else 0.0

    # masks
    disease_pixels = disease_mask > 0
    healthy_mask = np.zeros_like(leaf_mask)
    healthy_mask[(leaf_mask > 0) & (disease_mask == 0)] = 255

    # RGB stats
    r, g, b = cv2.split(image)
    diseased_mean_r, diseased_std_r = region_stats(r, disease_mask)
    diseased_mean_g, diseased_std_g = region_stats(g, disease_mask)
    diseased_mean_b, diseased_std_b = region_stats(b, disease_mask)

    healthy_mean_r, healthy_std_r = region_stats(r, healthy_mask)
    healthy_mean_g, healthy_std_g = region_stats(g, healthy_mask)
    healthy_mean_b, healthy_std_b = region_stats(b, healthy_mask)

    # HSV stats
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    h, s, v = cv2.split(hsv)

    diseased_h_mean, diseased_h_std = region_stats(h, disease_mask)
    diseased_s_mean, diseased_s_std = region_stats(s, disease_mask)
    diseased_v_mean, diseased_v_std = region_stats(v, disease_mask)

    healthy_h_mean, healthy_h_std = region_stats(h, healthy_mask)
    healthy_s_mean, healthy_s_std = region_stats(s, healthy_mask)
    healthy_v_mean, healthy_v_std = region_stats(v, healthy_mask)

    delta_h = diseased_h_mean - healthy_h_mean
    delta_s = diseased_s_mean - healthy_s_mean
    delta_v = diseased_v_mean - healthy_v_mean

    # largest lesion shape
    compactness, extent, aspect_ratio = compute_largest_spot_shape_features(disease_mask)

    # GLCM texture over leaf region
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    glcm_contrast, glcm_homogeneity, glcm_energy, glcm_correlation = compute_glcm_features(gray, leaf_mask)

    features = {
        "leaf_area": int(leaf_area),
        "disease_area": int(disease_area),
        "severity": float(severity),
        "severity_label": severity_label(severity),

        "num_spots": int(num_spots),
        "largest_spot_area": int(largest_spot_area),
        "mean_spot_area": float(mean_spot_area),
        "std_spot_area": float(std_spot_area),
        "largest_spot_ratio": float(largest_spot_ratio),
        "spot_density": float(spot_density),

        "compactness": float(compactness),
        "extent": float(extent),
        "aspect_ratio": float(aspect_ratio),

        "diseased_mean_r": float(diseased_mean_r),
        "diseased_std_r": float(diseased_std_r),
        "diseased_mean_g": float(diseased_mean_g),
        "diseased_std_g": float(diseased_std_g),
        "diseased_mean_b": float(diseased_mean_b),
        "diseased_std_b": float(diseased_std_b),

        "healthy_mean_r": float(healthy_mean_r),
        "healthy_std_r": float(healthy_std_r),
        "healthy_mean_g": float(healthy_mean_g),
        "healthy_std_g": float(healthy_std_g),
        "healthy_mean_b": float(healthy_mean_b),
        "healthy_std_b": float(healthy_std_b),

        "diseased_h_mean": float(diseased_h_mean),
        "diseased_h_std": float(diseased_h_std),
        "diseased_s_mean": float(diseased_s_mean),
        "diseased_s_std": float(diseased_s_std),
        "diseased_v_mean": float(diseased_v_mean),
        "diseased_v_std": float(diseased_v_std),

        "healthy_h_mean": float(healthy_h_mean),
        "healthy_h_std": float(healthy_h_std),
        "healthy_s_mean": float(healthy_s_mean),
        "healthy_s_std": float(healthy_s_std),
        "healthy_v_mean": float(healthy_v_mean),
        "healthy_v_std": float(healthy_v_std),

        "delta_h": float(delta_h),
        "delta_s": float(delta_s),
        "delta_v": float(delta_v),

        "glcm_contrast": float(glcm_contrast),
        "glcm_homogeneity": float(glcm_homogeneity),
        "glcm_energy": float(glcm_energy),
        "glcm_correlation": float(glcm_correlation),
    }

    return features


# ---------------------------
# visualization
# ---------------------------
def highlight_disease(image, disease_mask):
    overlay = image.copy()
    overlay[disease_mask > 0] = [255, 0, 0]
    blended = cv2.addWeighted(image, 0.7, overlay, 0.3, 0)
    return blended


# ---------------------------
# full pipeline
# ---------------------------
def process_leaf_image(image):
    processed = preprocess_image(image)

    leaf_mask = segment_leaf(processed)
    leaf_mask = clean_mask(leaf_mask, kernel_size=5, open_iter=1, close_iter=2)
    leaf_mask = keep_largest_component(leaf_mask)

    disease_mask = segment_disease_simple(processed, leaf_mask)
    disease_mask = clean_mask(disease_mask, kernel_size=3, open_iter=1, close_iter=1)

    leaf_regions = extract_regions(leaf_mask)
    disease_regions = extract_regions(disease_mask)

    leaf_area, disease_area, severity = calculate_severity(leaf_mask, disease_mask)
    features = extract_features(processed, leaf_mask, disease_mask)
    highlighted = highlight_disease(processed, disease_mask)

    return {
        "processed": processed,
        "leaf_mask": leaf_mask,
        "disease_mask": disease_mask,
        "highlighted": highlighted,
        "leaf_regions": leaf_regions,
        "disease_regions": disease_regions,
        "leaf_area": leaf_area,
        "disease_area": disease_area,
        "severity": severity,
        "features": features
    }


# ---------------------------
# testing on one local folder
# ---------------------------
if __name__ == "__main__":
    dataset_path = r"C:\Users\hp\Desktop\Leaf_Disease_Recognition\dataset_imgs"

    if not os.path.exists(dataset_path):
        print("Dataset test folder not found. Change dataset_path in the file if needed.")
    else:
        image_files = [
            f for f in os.listdir(dataset_path)
            if f.lower().endswith((".jpg", ".png", ".jfif", ".jpeg"))
        ]

        if len(image_files) == 0:
            print("No images found in dataset_path.")
        else:
            img_path = os.path.join(dataset_path, image_files[0])
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            result = process_leaf_image(img)

            print("Brightness:", round(check_brightness(img), 2))
            print("Contrast:", round(check_contrast(img), 2))
            print("Leaf area:", result["leaf_area"])
            print("Disease area:", result["disease_area"])
            print("Severity (%):", round(result["severity"], 2))
            print("Severity label:", result["features"]["severity_label"])

            print("\nExtracted Features:")
            for key, value in result["features"].items():
                print(f"{key}: {value}")

            plt.figure(figsize=(18, 10))

            plt.subplot(2, 3, 1)
            plt.imshow(img)
            plt.title("Original")
            plt.axis("off")

            plt.subplot(2, 3, 2)
            plt.imshow(result["processed"])
            plt.title("Preprocessed")
            plt.axis("off")

            plt.subplot(2, 3, 3)
            plt.imshow(result["leaf_mask"], cmap="gray")
            plt.title("Leaf Mask")
            plt.axis("off")

            plt.subplot(2, 3, 4)
            plt.imshow(result["disease_mask"], cmap="gray")
            plt.title("Disease Mask")
            plt.axis("off")

            plt.subplot(2, 3, 5)
            plt.imshow(result["highlighted"])
            plt.title("Highlighted Disease")
            plt.axis("off")

            plt.subplot(2, 3, 6)
            plt.text(0.1, 0.85, f"Severity: {result['severity']:.2f}%", fontsize=16)
            plt.text(0.1, 0.68, f"Label: {result['features']['severity_label']}", fontsize=14)
            plt.text(0.1, 0.50, f"Leaf area: {result['leaf_area']}", fontsize=12)
            plt.text(0.1, 0.35, f"Disease area: {result['disease_area']}", fontsize=12)
            plt.text(0.1, 0.20, f"Spots: {result['features']['num_spots']}", fontsize=12)
            plt.text(0.1, 0.05, f"Largest spot: {result['features']['largest_spot_area']}", fontsize=12)
            plt.axis("off")

            plt.tight_layout()
            plt.show()