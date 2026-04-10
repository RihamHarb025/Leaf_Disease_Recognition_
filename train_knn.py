import os
import cv2
import numpy as np
import pandas as pd
from collections import Counter

from leaf_pipeline import process_leaf_image


# ---------------------------------------------------
# dataset config
# ---------------------------------------------------
DATASET_ROOT = r"C:\Users\hp\Desktop\Leaf_Disease_Recognition\dataset\plantvillage"

CLASSES = [
    "Tomato_healthy",
    "Tomato_Early_blight",
    "Tomato_Late_blight",
    "Tomato_Septoria_leaf_spot"
]

FEATURE_COLUMNS = [
    "leaf_area",
    "disease_area",
    "severity",
    "num_spots",
    "largest_spot_area",
    "mean_spot_area",
    "std_spot_area",
    "largest_spot_ratio",
    "spot_density",
    "compactness",
    "extent",
    "aspect_ratio",

    "diseased_mean_r",
    "diseased_std_r",
    "diseased_mean_g",
    "diseased_std_g",
    "diseased_mean_b",
    "diseased_std_b",

    "healthy_mean_r",
    "healthy_std_r",
    "healthy_mean_g",
    "healthy_std_g",
    "healthy_mean_b",
    "healthy_std_b",

    "diseased_h_mean",
    "diseased_h_std",
    "diseased_s_mean",
    "diseased_s_std",
    "diseased_v_mean",
    "diseased_v_std",

    "healthy_h_mean",
    "healthy_h_std",
    "healthy_s_mean",
    "healthy_s_std",
    "healthy_v_mean",
    "healthy_v_std",

    "delta_h",
    "delta_s",
    "delta_v",

    "glcm_contrast",
    "glcm_homogeneity",
    "glcm_energy",
    "glcm_correlation"
]


# ---------------------------------------------------
# dataset loading
# ---------------------------------------------------
def load_dataset(dataset_root, class_names):
    rows = []

    for class_name in class_names:
        class_dir = os.path.join(dataset_root, class_name)

        if not os.path.exists(class_dir):
            print(f"Missing folder: {class_dir}")
            continue

        print(f"Reading class: {class_name}")

        for fname in os.listdir(class_dir):
            if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
                continue

            img_path = os.path.join(class_dir, fname)

            try:
                img = cv2.imread(img_path)
                if img is None:
                    continue

                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                result = process_leaf_image(img)
                features = result["features"]

                # skip failed segmentations
                if features["leaf_area"] == 0:
                    continue

                row = {
                    "filename": fname,
                    "label": class_name
                }
                row.update(features)
                rows.append(row)

            except Exception as e:
                print(f"Skipped {img_path}: {e}")

    df = pd.DataFrame(rows)
    return df


# ---------------------------------------------------
# manual balanced split
# ---------------------------------------------------
def split_train_test(df, test_ratio=0.2, random_state=42):
    np.random.seed(random_state)

    train_parts = []
    test_parts = []

    for class_name in df["label"].unique():
        class_df = df[df["label"] == class_name].sample(frac=1, random_state=random_state).reset_index(drop=True)

        test_size = int(len(class_df) * test_ratio)
        test_df = class_df.iloc[:test_size]
        train_df = class_df.iloc[test_size:]

        train_parts.append(train_df)
        test_parts.append(test_df)

    train_df = pd.concat(train_parts).reset_index(drop=True)
    test_df = pd.concat(test_parts).reset_index(drop=True)

    return train_df, test_df


# ---------------------------------------------------
# build matrices
# ---------------------------------------------------
def build_xy(train_df, test_df):
    X_train = train_df[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
    y_train = train_df["label"].to_numpy()

    X_test = test_df[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
    y_test = test_df["label"].to_numpy()

    return X_train, y_train, X_test, y_test


# ---------------------------------------------------
# manual z-score normalization
# ---------------------------------------------------
def zscore_normalize(X_train, X_test):
    train_mean = np.mean(X_train, axis=0)
    train_std = np.std(X_train, axis=0)

    # avoid divide by zero
    train_std[train_std == 0] = 1.0

    X_train_norm = (X_train - train_mean) / train_std
    X_test_norm = (X_test - train_mean) / train_std

    return X_train_norm, X_test_norm, train_mean, train_std


# ---------------------------------------------------
# manual Euclidean distance
# ---------------------------------------------------
def euclidean_distance(train_matrix, test_vector):
    squared_diff = (train_matrix - test_vector) ** 2
    distances = np.sqrt(np.sum(squared_diff, axis=1))
    return distances


# ---------------------------------------------------
# manual k-NN
# ---------------------------------------------------
def knn_predict_one(X_train_norm, y_train, test_vector, k=3):
    distances = euclidean_distance(X_train_norm, test_vector)
    nearest_indices = np.argsort(distances)[:k]
    nearest_labels = y_train[nearest_indices]

    vote_counts = Counter(nearest_labels)
    prediction = vote_counts.most_common(1)[0][0]

    return prediction


def knn_predict_all(X_train_norm, y_train, X_test_norm, k=3):
    predictions = []

    for test_vector in X_test_norm:
        pred = knn_predict_one(X_train_norm, y_train, test_vector, k=k)
        predictions.append(pred)

    return np.array(predictions)


# ---------------------------------------------------
# manual evaluation
# ---------------------------------------------------
def compute_accuracy(y_true, y_pred):
    return float(np.mean(y_true == y_pred))


def compute_confusion_matrix_multiclass(y_true, y_pred, class_names):
    n = len(class_names)
    matrix = np.zeros((n, n), dtype=int)

    class_to_idx = {name: i for i, name in enumerate(class_names)}

    for true_label, pred_label in zip(y_true, y_pred):
        i = class_to_idx[true_label]
        j = class_to_idx[pred_label]
        matrix[i, j] += 1

    return matrix


def compute_metrics_from_confusion(conf_matrix, class_names):
    metrics = {}

    for i, class_name in enumerate(class_names):
        tp = conf_matrix[i, i]
        fp = np.sum(conf_matrix[:, i]) - tp
        fn = np.sum(conf_matrix[i, :]) - tp
        tn = np.sum(conf_matrix) - (tp + fp + fn)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

        metrics[class_name] = {
            "TP": int(tp),
            "FP": int(fp),
            "FN": int(fn),
            "TN": int(tn),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1)
        }

    return metrics


# ---------------------------------------------------
# save reusable model data for app prediction
# ---------------------------------------------------
def save_knn_artifacts(X_train_norm, y_train, train_mean, train_std):
    os.makedirs("models", exist_ok=True)
    np.save("models/X_train_norm.npy", X_train_norm)
    np.save("models/y_train.npy", y_train)
    np.save("models/train_mean.npy", train_mean)
    np.save("models/train_std.npy", train_std)


def predict_single_image(image, k=3):
    result = process_leaf_image(image)
    features = result["features"]

    x = np.array([[features[col] for col in FEATURE_COLUMNS]], dtype=np.float32)

    X_train_norm = np.load("models/X_train_norm.npy", allow_pickle=True)
    y_train = np.load("models/y_train.npy", allow_pickle=True)
    train_mean = np.load("models/train_mean.npy")
    train_std = np.load("models/train_std.npy")

    train_std[train_std == 0] = 1.0
    x_norm = (x - train_mean) / train_std

    pred = knn_predict_one(X_train_norm, y_train, x_norm[0], k=k)

    return pred, result


# ---------------------------------------------------
# main training / evaluation pipeline
# ---------------------------------------------------
def main():
    os.makedirs("outputs", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    print("Loading dataset...")
    df = load_dataset(DATASET_ROOT, CLASSES)

    if df.empty:
        print("No valid samples were loaded. Check dataset path and class folder names.")
        return

    print(f"Total samples loaded: {len(df)}")
    print("\nSamples per class:")
    print(df["label"].value_counts())

    # save all extracted features
    df.to_csv("outputs/all_features.csv", index=False)

    train_df, test_df = split_train_test(df, test_ratio=0.2, random_state=42)

    print(f"\nTrain size: {len(train_df)}")
    print(f"Test size: {len(test_df)}")

    train_df.to_csv("outputs/train_features.csv", index=False)
    test_df.to_csv("outputs/test_features.csv", index=False)

    X_train, y_train, X_test, y_test = build_xy(train_df, test_df)
    X_train_norm, X_test_norm, train_mean, train_std = zscore_normalize(X_train, X_test)

    # try a few k values
    best_k = None
    best_acc = -1
    best_y_pred = None

    for k in [3, 5, 7]:
        if k > len(X_train_norm):
            continue

        y_pred = knn_predict_all(X_train_norm, y_train, X_test_norm, k=k)
        acc = compute_accuracy(y_test, y_pred)

        print(f"k={k} -> Accuracy = {acc * 100:.2f}%")

        if acc > best_acc:
            best_acc = acc
            best_k = k
            best_y_pred = y_pred

    print(f"\nBest k = {best_k}")
    print(f"Best Accuracy = {best_acc * 100:.2f}%")

    conf_matrix = compute_confusion_matrix_multiclass(y_test, best_y_pred, CLASSES)
    metrics = compute_metrics_from_confusion(conf_matrix, CLASSES)

    print("\nConfusion Matrix:")
    print(conf_matrix)

    print("\nPer-class metrics:")
    for cls, vals in metrics.items():
        print(f"\n{cls}")
        for key, value in vals.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.4f}")
            else:
                print(f"  {key}: {value}")

    pd.DataFrame(conf_matrix, index=CLASSES, columns=CLASSES).to_csv("outputs/confusion_matrix.csv")

    with open("outputs/results.txt", "w", encoding="utf-8") as f:
        f.write(f"Best k: {best_k}\n")
        f.write(f"Accuracy: {best_acc * 100:.2f}%\n\n")
        f.write("Confusion Matrix:\n")
        f.write(str(conf_matrix))
        f.write("\n\nPer-class metrics:\n")

        for cls, vals in metrics.items():
            f.write(f"\n{cls}\n")
            for key, value in vals.items():
                if isinstance(value, float):
                    f.write(f"  {key}: {value:.4f}\n")
                else:
                    f.write(f"  {key}: {value}\n")

    # save artifacts using the best training normalization
    save_knn_artifacts(X_train_norm, y_train, train_mean, train_std)

    print("\nSaved:")
    print("- outputs/all_features.csv")
    print("- outputs/train_features.csv")
    print("- outputs/test_features.csv")
    print("- outputs/confusion_matrix.csv")
    print("- outputs/results.txt")
    print("- models/X_train_norm.npy")
    print("- models/y_train.npy")
    print("- models/train_mean.npy")
    print("- models/train_std.npy")


if __name__ == "__main__":
    main()