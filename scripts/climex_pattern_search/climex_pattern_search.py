import json
import pickle
from pathlib import Path

from prob_unet.datasets.climex import pr_multiple_samples_batch

snakemake_workflow_dir = Path.cwd()  # Normally set by Snakemake


def main():
    with open(snakemake.input[0], "r") as f:
        batch = json.load(f)
    list_of_candidates = [tuple(x[1:]) for x in batch]  # removing variable entry
    feature_array = pr_multiple_samples_batch(
        path_climex=snakemake.params.config.path_climex,
        list_of_candidates=list_of_candidates,
        i_min=snakemake.params.config.i_min,
        i_max=snakemake.params.config.i_max,
        j_min=snakemake.params.config.j_min,
        j_max=snakemake.params.config.j_max,
        upscale_factor=snakemake.params.config.upscale_factor,
        feature_mode=snakemake.params.config.feature_mode)
    with open(snakemake.output[0], "wb") as f:
        pickle.dump({'keys': list_of_candidates, 'features': feature_array}, f)


if __name__ == "__main__":
    main()
