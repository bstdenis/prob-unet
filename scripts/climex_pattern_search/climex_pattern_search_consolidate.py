import pickle
from pathlib import Path

import numpy as np
from prob_unet.patterns.feature_extraction import FeatureStack

snakemake_workflow_dir = Path.cwd()  # Normally set by Snakemake


def main():
    list_of_keys = []
    features = []
    for pkl_input in snakemake.input:
        z = pickle.load(open(pkl_input, "rb"))
        list_of_keys.extend(z['keys'])
        features.append(z['features'])
    features = np.concatenate(features, axis=0)
    feature_stack = FeatureStack(num_samples=features.shape[0], num_dimensions=features.shape[1])
    feature_stack.keys = list_of_keys
    feature_stack.stack = features
    feature_stack.stack_length = features.shape[0]
    feature_stack.normalize_stack()
    feature_stack.build_hnsw()
    with open(snakemake.output[0], 'wb') as f:
        pickle.dump(feature_stack, f)


if __name__ == "__main__":
    main()
