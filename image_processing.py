import cv2
import numpy as np
import matplotlib.pyplot as plt
import os


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
# visualization
# ---------------------------
def highlight_disease(image, disease_mask):
    overlay = image.copy()
    overlay[disease_mask > 0] = [255, 0, 0]
    blended = cv2.addWeighted(image, 0.7, overlay, 0.3, 0)
    return blended


# ---------------------------
# full phase 1 pipeline
# ---------------------------
def process_leaf_image_phase1(image):
    processed = preprocess_image(image)

    leaf_mask = segment_leaf(processed)
    leaf_mask = clean_mask(leaf_mask, kernel_size=5, open_iter=1, close_iter=2)
    leaf_mask = keep_largest_component(leaf_mask)

    disease_mask = segment_disease_simple(processed, leaf_mask)
    disease_mask = clean_mask(disease_mask, kernel_size=3, open_iter=1, close_iter=1)

    leaf_regions = extract_regions(leaf_mask)
    disease_regions = extract_regions(disease_mask)

    leaf_area, disease_area, severity = calculate_severity(leaf_mask, disease_mask)
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
        "severity": severity
    }


# ---------------------------
# testing
# ---------------------------
dataset_path = r"C:\Users\hp\Desktop\Leaf_Disease_Recognition\dataset_imgs"
image_files = [
    f for f in os.listdir(dataset_path)
    if f.lower().endswith((".jpg", ".png", ".jfif", ".jpeg"))
]

img_path = os.path.join(dataset_path, image_files[0])
img = cv2.imread(img_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

result = process_leaf_image_phase1(img)

print("Brightness:", round(check_brightness(img), 2))
print("Contrast:", round(check_contrast(img), 2))
print("Leaf area:", result["leaf_area"])
print("Disease area:", result["disease_area"])
print("Severity (%):", round(result["severity"], 2))

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
plt.text(0.1, 0.7, f"Severity: {result['severity']:.2f}%", fontsize=18)
plt.text(0.1, 0.5, f"Leaf area: {result['leaf_area']}", fontsize=14)
plt.text(0.1, 0.3, f"Disease area: {result['disease_area']}", fontsize=14)
plt.axis("off")

plt.tight_layout()
plt.show()