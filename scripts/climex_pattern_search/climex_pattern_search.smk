# snakemake -s climex_pattern_search.smk --config config_yaml=climex_pattern_search.yaml -j1 --directory=/path/to/climex_pattern_search
from pack_dev.config_utils import config_from_yaml
from prob_unet.datasets.climex_workflow import ClimexPatternSearchConfig, pattern_search_batches_to_json

snakefile_dir = Path(str(workflow.snakefile)).parent  # This is the original directory where the snakefile is located
workflow_dir = Path.cwd()  # This is the current working directory where snakemake is being executed

config_yaml = Path(snakefile_dir, config['config_yaml'])
if not config_yaml.is_file():
    config_yaml = Path(config['config_yaml'])
config_obj = config_from_yaml(ClimexPatternSearchConfig, config_yaml)


def get_batch_file(wildcards):
    _ = checkpoints.pattern_search_batching.get()
    return f"results/climex_batch_{wildcards.j}.json"


def get_feature_files(wildcards):
    _ = checkpoints.pattern_search_batching.get()
    json_files = list(Path(workflow_dir, "results").glob("climex_batch_*.json"))
    pkl_files = [f"results/climex_features_{f.stem.split('_')[-1]}.pkl" for f in json_files]
    return pkl_files


rule all:
    input:
        "results/task.done"


checkpoint pattern_search_batching:
    output:
        touch("results/pattern_search_batching.done")
    run:
        pattern_search_batches_to_json(config_obj)


rule compute_features:
    input:
        get_batch_file
    output:
        "results/climex_features_{j}.pkl"
    params:
        config=config_obj
    script:
        "climex_pattern_search.py"


rule consolidate:
    input:
        get_feature_files
    output:
        touch("results/task.done")
