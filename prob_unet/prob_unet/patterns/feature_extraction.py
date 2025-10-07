import hnswlib
import numpy as np

from prob_unet.patterns import pattern_metrics


def extract_2d_array_features(
    array,
    dct_k=8,
    include_dc=True,
    wavelet_levels=1,
    add_edge_density=True,
    orientation_bins=0
):
    """
    Returns a 1D feature vector for a single 2d array.
    Pipeline:
      - DCT low-frequency zigzag (dct_k dims)
      - Haar wavelet sub-band energies (3 * wavelet_levels dims)
      - Edge density (1 dim if add_edge_density)
      - Optional gradient orientation histogram (orientation_bins dims)
    """
    if (array.min() < 0) or (array.max() > 1):
        raise ValueError("Input array should be normalized to [0, 1]")

    # DCT low-frequency coefficients
    dct_features = pattern_metrics.low_frequency_dct_coefficients(array, k=dct_k, include_dc=include_dc)

    # Wavelet energies across levels
    wavelet_features = pattern_metrics.wavelet_energies(array, levels=wavelet_levels)

    # Edge and orientation stats
    eo_features = []
    if add_edge_density or (orientation_bins and orientation_bins > 0):
        eo_features = pattern_metrics.edge_density_and_orientation_histogram(array, bins=orientation_bins)

    features = dct_features + wavelet_features + eo_features
    return np.asarray(features, dtype=np.float32)


def build_hnsw(features, m=32, ef_construction=200, ef_search=64):
    """
    features: np.ndarray of shape (N, D), already z-scored and L2-normalized.
    """
    n, d = features.shape
    index = hnswlib.Index(space='cosine', dim=d)
    index.init_index(max_elements=n, ef_construction=ef_construction, M=m)
    index.set_ef(ef_search)
    index.add_items(features, np.arange(n))
    return index


def custom_features(array, mode='16d'):
    # array must be 2D, normalized to [0,1]
    if mode == '8d':
        features = extract_2d_array_features(
            array, dct_k=4, include_dc=False, wavelet_levels=1, add_edge_density=True, orientation_bins=0)
    elif mode == '16d':
        features = extract_2d_array_features(
            array, dct_k=6, include_dc=True, wavelet_levels=2, add_edge_density=True, orientation_bins=3)
    elif mode == '24d':
        features = extract_2d_array_features(
            array, dct_k=9, include_dc=True, wavelet_levels=2, add_edge_density=True, orientation_bins=8)
    else:
        raise NotImplementedError(f"Mode {mode} not implemented.")
    return features


class FeatureStack:
    def __init__(self, num_samples=None, num_dimensions=None):
        if (num_samples is None) or (num_dimensions is None):
            self.stack = []
        else:
            self.stack = np.zeros((num_samples, num_dimensions))
        self.stack_length = 0
        self.features_mean = None
        self.features_std = None
        self.hnsw_index = None

    def add(self, features):
        if isinstance(self.stack, list):
            self.stack.append(features)
        else:
            self.stack[self.stack_length, :] = features
        self.stack_length += 1

    def normalize_features(self, features):
        if self.features_mean is None or self.features_std is None:
            raise ValueError("Feature stack not normalized yet.")
        features_norm = (features - self.features_mean) / self.features_std
        features_norm = features_norm / (np.linalg.norm(features_norm) + 1e-12)
        return features_norm

    def normalize_stack(self):
        if isinstance(self.stack, list):
            raise NotImplementedError()
        self.features_mean = self.stack.mean(axis=0, keepdims=True)
        self.features_std = self.stack.std(axis=0, keepdims=True) + 1e-12
        features_stack = (self.stack - self.features_mean) / self.features_std
        features_norm = np.linalg.norm(features_stack, axis=1, keepdims=True) + 1e-12
        self.stack = features_stack / features_norm

    def build_hnsw(self):
        if isinstance(self.stack, list):
            raise NotImplementedError()
        self.hnsw_index = build_hnsw(self.stack)
