import pickle
from pathlib import Path

import numpy as np
from prob_unet.patterns.feature_extraction import FeatureStack

snakemake_workflow_dir = Path.cwd()  # Normally set by Snakemake


def main():
    with open(snakemake.input[0], "rb") as f:
        feature_stack = pickle.load(f)
    list_of_keys = []
    features = []
    for pkl_input in snakemake.input[1:]:
        z = pickle.load(open(pkl_input, "rb"))
        list_of_keys.extend(z['keys'])
        features.append(z['features'])
    features = np.concatenate(features, axis=0)
    neighbors = {}
    for i in range(features.shape[0]):
        neighbors[list_of_keys[i]] = []
        q_feat = feature_stack.normalize_features(features[i, :])
        labels, distances = feature_stack.hnsw_index.knn_query(q_feat, k=6, filter=None)
        # Note that the closest result should naturally be itself!!! But will try to exclude it even if 2nd or later...
        for j in range(6):
            if list_of_keys[i] == list_of_keys[int(labels[0][j])]:
                continue
            neighbors[list_of_keys[i]].append(list_of_keys[int(labels[0][j])])
            if len(neighbors[list_of_keys[i]]) >= 5:
                break
    with open(snakemake.output[0], 'wb') as f:
        pickle.dump(neighbors, f)


if __name__ == "__main__":
    main()
