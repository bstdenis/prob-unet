import argparse
import logging
from pathlib import Path

from resoterre.logging_utils import start_root_logger

from prob_unet.datasets.climex_for_prob_unet import climex_upscale_single_year_to_disk

logger = logging.getLogger(__name__)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Climex daily to coarse data processing")
    parser.add_argument("--workflow_dir", type=str, required=True, help="Path to the workflow output directory")
    parser.add_argument("--config", type=str, required=True, help="Path to the configuration file")
    parser.add_argument("--member", type=str, required=True, help="Climex member to process")
    parser.add_argument("--year", type=int, required=True, help="Year to process")
    args = parser.parse_args()

    log_file = start_root_logger(
        basic_config_args={
            "filename": str(Path(args.workflow_dir, "logs", "bucket",
                                 f"daily_to_coarse_{args.member}_{args.year}.log"))
        }
    )

    try:
        climex_upscale_single_year_to_disk(config=args.config, member=args.member, year=args.year)
    except Exception:
        logger.exception("Error calling climex_upscale_single_year_to_disk")
        raise
