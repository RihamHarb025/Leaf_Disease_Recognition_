"""
pipeline/__init__.py
Expose the main public API for the LeafGuard pipeline.
"""
from pipeline.preprocessing  import check_quality, preprocess, QualityReport
from pipeline.segmentation   import segment_leaf, segment_stress, detect_stress_edges, highlight_stress
from pipeline.features       import extract_features, feature_vector, FEATURE_NAMES
from pipeline.classifier     import predict, rule_based_classify, train_knn, loo_evaluate, CLASSES
from pipeline.decision_engine import fuse
from pipeline.dataset        import load_dataset, train_from_folder, folder_to_label

__all__ = [
    "check_quality", "preprocess", "QualityReport",
    "segment_leaf", "segment_stress", "detect_stress_edges", "highlight_stress",
    "extract_features", "feature_vector", "FEATURE_NAMES",
    "predict", "rule_based_classify", "train_knn", "loo_evaluate", "CLASSES",
    "fuse",
    "load_dataset", "train_from_folder", "folder_to_label",
]
