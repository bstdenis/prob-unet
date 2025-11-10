import json
from dataclasses import dataclass
from pathlib import Path

from prob_unet.datasets import climex


@dataclass(frozen=True, slots=True)
class ClimexPatternSearchConfig:
    path_climex: Path
    variables: list[str]
    i_min: int
    i_max: int
    j_min: int
    j_max: int
    path_workflow: Path | None = None
    simulations: list[str] | None = None
    start_year: int = 1950
    end_year: int = 2099
    months: list[int] | None = None
    upscale_factor: int = 4
    feature_mode: str = "16d"
    batch_size: int = 64


def pattern_search_batches_to_json(config):
    if config.path_workflow is None:
        raise ValueError("path_workflow must be provided in the config.")
    daily_timesteps_climex = climex.list_daily_timesteps(
        path_climex=config.path_climex,
        variables=config.variables,
        simulations=config.simulations,
        years=list(range(config.start_year, config.end_year + 1)),
        months=config.months)
    batches = climex.split_daily_timesteps_in_batch(daily_timesteps_climex, batch_size=config.batch_size)
    for i, batch in enumerate(batches):
        with open(Path(config.path_workflow, "results", f"climex_batch_{i:08d}.json"), "w") as f:
            f.write(json.dumps(batch, indent=2))
