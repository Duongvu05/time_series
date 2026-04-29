"""Data pipeline: fetch VN-Index prices + Google Trends, engineer features, export CSV."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from loguru import logger

from time_series.config import DataConfig
from time_series.data import build_split_panels, check_data_quality, engineer_daily_dataset, export_dataset


def main() -> None:
    config = DataConfig()

    Path(config.cache_dir).mkdir(parents=True, exist_ok=True)
    Path(config.data_dir).mkdir(parents=True, exist_ok=True)

    logger.info("Starting data pipeline")
    price_insample, keyword_insample, insample_panel, price_outsample, keyword_outsample, outsample_panel, svi_meta = (
        build_split_panels(config)
    )

    check_data_quality(price_insample, "insample_prices")
    check_data_quality(keyword_insample, "insample_google_trends")
    check_data_quality(insample_panel, "insample_raw_panel")
    check_data_quality(price_outsample, "outsample_prices")
    check_data_quality(keyword_outsample, "outsample_google_trends")
    check_data_quality(outsample_panel, "outsample_raw_panel")

    insample_start = insample_panel.index.min()
    insample_end = insample_panel.index.max()
    outsample_start = outsample_panel.index.min()

    assert insample_end < outsample_start, "Data leakage: in-sample overlaps with out-of-sample"
    logger.info("Split boundary OK: max(insample)={} < min(outsample)={}", insample_end.date(), outsample_start.date())

    insample_df = engineer_daily_dataset(insample_panel)
    outsample_df = engineer_daily_dataset(outsample_panel)

    export_dataset(insample_df, config, config.insample_filename, overwrite=config.update_insample_file)
    export_dataset(outsample_df, config, config.outsample_filename, overwrite=config.update_outsample_file)

    logger.info("In-sample shape:  {}", insample_df.shape)
    logger.info("Out-of-sample shape: {}", outsample_df.shape)
    logger.info("SVI PCA explained variance: {:.4f}", svi_meta["explained_variance_ratio"])
    logger.info("Data pipeline complete.")


if __name__ == "__main__":
    main()
