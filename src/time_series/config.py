from typing import Optional

from pydantic import BaseModel, field_validator


class DataConfig(BaseModel):
    symbol: str = "VNINDEX"
    fallback_symbols: tuple[str, ...] = ("VN-INDEX", "VNI")
    price_source: str = "VCI"

    insample_start_date: str = "2010-01-01"
    insample_end_date: str = "2025-12-31"
    outsample_start_date: str = "2026-01-01"
    outsample_end_date: Optional[str] = "2026-03-31"

    keywords: tuple[str, ...] = ("chung khoan", "VN-Index", "co phieu", "dau tu")
    geo: str = "VN"
    hl: str = "en-US"
    tz: int = 420

    trend_window_days: int = 240
    trend_overlap_days: int = 14
    trend_sleep_seconds: float = 8.0
    trend_max_retries: int = 5
    trend_backoff_base_seconds: float = 5.0
    trend_backoff_max_seconds: float = 10.0

    api_sleep_seconds: float = 1.5
    api_max_retries: int = 4
    api_backoff_base_seconds: float = 2.0
    api_backoff_max_seconds: float = 30.0

    cache_dir: str = "outputs/trends_cache"
    data_dir: str = "data"
    insample_filename: str = "insample_data.csv"
    outsample_filename: str = "outofsample_data.csv"

    update_insample_file: bool = True
    update_outsample_file: bool = True

    @field_validator("trend_overlap_days")
    @classmethod
    def overlap_must_be_smaller_than_window(cls, v: int, info) -> int:
        window = info.data.get("trend_window_days", 240)
        if v >= window:
            raise ValueError("trend_overlap_days must be smaller than trend_window_days")
        return v
