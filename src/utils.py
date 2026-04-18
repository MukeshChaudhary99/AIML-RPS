from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def get_logger(name: str, level: str = "INFO") -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        logger.setLevel(level.upper())
        return logger

    logger.setLevel(level.upper())
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.propagate = False
    return logger


def get_service_period(hour: int) -> str:
    if hour < 8:
        return "pre_service"
    if 8 <= hour <= 11:
        return "breakfast"
    if 12 <= hour <= 15:
        return "lunch"
    if 16 <= hour <= 18:
        return "afternoon"
    if 19 <= hour <= 22:
        return "dinner"
    return "late_night"


def ensure_timestamp(df: pd.DataFrame, column: str = "timestamp") -> pd.DataFrame:
    result = df.copy()
    result[column] = pd.to_datetime(result[column])
    return result


def dataframe_to_records(df: pd.DataFrame) -> list[dict]:
    if df.empty:
        return []

    cleaned_df = df.replace({np.nan: None})
    for col in cleaned_df.columns:
        if pd.api.types.is_datetime64_any_dtype(cleaned_df[col]):
            cleaned_df[col] = cleaned_df[col].astype("string")
    return cleaned_df.to_dict(orient="records")


def format_key_value_block(title: str, items: list[tuple[str, Any]]) -> str:
    if not items:
        return title

    key_width = max(len(str(key)) for key, _ in items)
    lines = [title]
    for key, value in items:
        lines.append(f"  {str(key).ljust(key_width)} : {value}")
    return "\n".join(lines)


def render_dataframe_block(df: pd.DataFrame, max_rows: int = 20) -> str:
    if df is None or df.empty:
        return "No rows"

    preview_df = df.head(max_rows).copy()
    preview_df = preview_df.replace({np.nan: None})

    for col in preview_df.columns:
        if pd.api.types.is_datetime64_any_dtype(preview_df[col]):
            preview_df[col] = preview_df[col].dt.strftime("%Y-%m-%d %H:%M")
        elif pd.api.types.is_float_dtype(preview_df[col]):
            preview_df[col] = preview_df[col].round(4)

    with pd.option_context(
        "display.max_columns",
        None,
        "display.width",
        240,
        "display.max_colwidth",
        60,
        "display.expand_frame_repr",
        False,
    ):
        return preview_df.to_string(index=False)


def format_dataframe_block(title: str, df: pd.DataFrame, max_rows: int = 20) -> str:
    if df is None or df.empty:
        return f"{title}\n  No rows"

    return f"{title}\n{render_dataframe_block(df, max_rows=max_rows)}"
