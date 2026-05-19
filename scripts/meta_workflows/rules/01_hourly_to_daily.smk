"""Snakemake workflow for climex hourly to daily data processing.

To run this workflow, use the command:
snakemake -s 01_hourly_to_daily.smk -j1 --config config_yaml=config.yaml --directory=/workflow_directory
"""

from pathlib import Path

from resoterre.config_utils import config_from_yaml

from prob_unet.datasets.climex_for_prob_unet import ProbUnetClimexConfig

snakefile_dir = Path(str(workflow.snakefile)).parent
workflow_dir = Path.cwd()
config_obj = config_from_yaml(ProbUnetClimexConfig, config["config_yaml"])


def expected_manifests(wildcards):
    list_of_expected_manifests = []
    for member in config_obj.members:
        for year in range(config_obj.starting_training_year, config_obj.ending_training_year + 1):
            list_of_expected_manifests.append(f"manifests/climex_hourly_to_daily_{member}_{year}.done")
    return list_of_expected_manifests


rule all:
    input:
        expected_manifests


rule hourly_to_daily:
    output:
        touch("manifests/climex_hourly_to_daily_{member}_{year}.done")
    params:
        path_script=Path(snakefile_dir, "01_hourly_to_daily.py"),
        workflow_dir=workflow_dir,
        config_yaml=config["config_yaml"],
    shell:
        """
        python3 {params.path_script} \
            --workflow_dir {params.workflow_dir} \
            --config {params.config_yaml} \
            --member {wildcards.member} \
            --year {wildcards.year}
        """