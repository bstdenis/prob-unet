import random
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt

from prob_unet.benchmarking.debug_patterns import generate_synthetic_image

from prob_unet.patterns import feature_extraction


def test_feature_stack():
    num_samples = 2000
    num_dimensions = 16

    # Initialize feature stack
    features_stack = feature_extraction.FeatureStack(num_samples=num_samples, num_dimensions=num_dimensions)

    list_of_data = []  # ToDo: in production, do not store all data in memory! retrieve it using the save key later
    # Populate feature stack with synthetic images features
    for i in range(num_samples):
        data, meta = generate_synthetic_image(shape=(128, 128), rng_seed=random.randint(0, 1000000000))
        list_of_data.append(data)  # ToDo: remove this line in production
        features = feature_extraction.custom_features(data, mode=f'{num_dimensions}d')
        features_stack.add(features, key=str(meta))
    # Once we are done populating the stack, normalize and build the hnsw index
    features_stack.normalize_stack()
    features_stack.build_hnsw()

    # Create a random query image and extract its (normalized) features
    data, meta = generate_synthetic_image(shape=(128, 128), rng_seed=random.randint(0, 1000000000))
    features = features_stack.normalize_features(feature_extraction.custom_features(data, mode=f'{num_dimensions}d'))
    # 5 nearest neighbors
    labels, distances = features_stack.hnsw_index.knn_query(features, k=5, filter=None)
    # Test with negative features as well (proxy for farthest away in feature space)
    labels_inv, distances_inv = features_stack.hnsw_index.knn_query(-features, k=5, filter=None)

    # Display
    with tempfile.TemporaryDirectory() as tmp_dir:
        fig = plt.figure(figsize=(10, 2))
        ax = fig.add_subplot(1, 11, 1)
        ax.imshow(data, cmap='gray')
        ax.set_title("Query")
        ax.axis('off')
        for i in range(5):
            ax = fig.add_subplot(1, 11, i+2)
            ax.imshow(list_of_data[int(labels[0, i])], cmap='gray')
            ax.set_title(f"Rank {i+1}")
            ax.axis('off')
        for i in range(5, 10):
            ax = fig.add_subplot(1, 11, i + 2)
            ax.imshow(list_of_data[int(labels_inv[0, i - 5])], cmap='gray')
            ax.set_title(f"Rank -{i + 1}")
            ax.axis('off')
        fig.tight_layout()
        fig.savefig(Path(tmp_dir, "hnsw_query.png"))
        assert Path(tmp_dir, "hnsw_query.png").is_file()
