from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DataConfig:
    data_dir: Path
    feedback_dir: Path


@dataclass(frozen=True)
class ForecastConfig:
    model_version: str
    auto_retrain: bool
    n_jobs: int
    n_estimators: int
    max_depth: int
    learning_rate: float
    subsample: float
    colsample_bytree: float
    min_child_weight: float
    gamma: float
    reg_alpha: float
    reg_lambda: float
    early_stopping_rounds: int
    startup_rows_per_group: int
    startup_max_test_fraction: float
    startup_recent_fraction: float
    startup_random_state: int


@dataclass(frozen=True)
class StaffConfig:
    prep_buffer_multiplier: float
    covers_buffer_multiplier: float


@dataclass(frozen=True)
class IngredientConfig:
    menu_qty_buffer_multiplier: float
    ingredient_qty_buffer_multiplier: float
    reorder_review_days: int


@dataclass(frozen=True)
class FeedbackConfig:
    correction_min_samples: int
    correction_shrinkage: float
    max_correction_pct: float
    retrain_error_threshold: float
    recent_feedback_weight: float
    high_error_weight: float
    manager_feedback_weight: float
    max_sample_weight: float


@dataclass(frozen=True)
class ApiConfig:
    title: str
    version: str
    host: str
    port: int
    log_level: str


@dataclass(frozen=True)
class AppConfig:
    data: DataConfig
    forecast: ForecastConfig
    staff: StaffConfig
    ingredient: IngredientConfig
    feedback: FeedbackConfig
    api: ApiConfig


def _get_bool_env(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def load_app_config() -> AppConfig:
    data_dir = Path(os.getenv("RPS_DATA_DIR", "V2Data"))
    feedback_dir = Path(os.getenv("RPS_FEEDBACK_DIR", "feedback_store"))

    return AppConfig(
        data=DataConfig(
            data_dir=data_dir,
            feedback_dir=feedback_dir,
        ),
        forecast=ForecastConfig(
            model_version=os.getenv("RPS_MODEL_VERSION", "xgb_feedback_v1"),
            auto_retrain=_get_bool_env("RPS_AUTO_RETRAIN", True),
            n_jobs=int(os.getenv("RPS_XGB_N_JOBS", "1")),
            n_estimators=int(os.getenv("RPS_XGB_N_ESTIMATORS", "800")),
            max_depth=int(os.getenv("RPS_XGB_MAX_DEPTH", "4")),
            learning_rate=float(os.getenv("RPS_XGB_LEARNING_RATE", "0.025")),
            subsample=float(os.getenv("RPS_XGB_SUBSAMPLE", "0.90")),
            colsample_bytree=float(os.getenv("RPS_XGB_COLSAMPLE_BYTREE", "0.90")),
            min_child_weight=float(os.getenv("RPS_XGB_MIN_CHILD_WEIGHT", "2.0")),
            gamma=float(os.getenv("RPS_XGB_GAMMA", "0.05")),
            reg_alpha=float(os.getenv("RPS_XGB_REG_ALPHA", "0.25")),
            reg_lambda=float(os.getenv("RPS_XGB_REG_LAMBDA", "5.0")),
            early_stopping_rounds=int(
                os.getenv("RPS_XGB_EARLY_STOPPING_ROUNDS", "50")
            ),
            startup_rows_per_group=int(
                os.getenv("RPS_STARTUP_ROWS_PER_GROUP", "6")
            ),
            startup_max_test_fraction=float(
                os.getenv("RPS_STARTUP_MAX_TEST_FRACTION", "0.18")
            ),
            startup_recent_fraction=float(
                os.getenv("RPS_STARTUP_RECENT_FRACTION", "0.40")
            ),
            startup_random_state=int(os.getenv("RPS_STARTUP_RANDOM_STATE", "42")),
        ),
        staff=StaffConfig(
            prep_buffer_multiplier=float(
                os.getenv("RPS_PREP_BUFFER_MULTIPLIER", "1.10")
            ),
            covers_buffer_multiplier=float(
                os.getenv("RPS_COVERS_BUFFER_MULTIPLIER", "1.05")
            ),
        ),
        ingredient=IngredientConfig(
            menu_qty_buffer_multiplier=float(
                os.getenv("RPS_MENU_QTY_BUFFER_MULTIPLIER", "1.05")
            ),
            ingredient_qty_buffer_multiplier=float(
                os.getenv("RPS_INGREDIENT_QTY_BUFFER_MULTIPLIER", "1.03")
            ),
            reorder_review_days=int(os.getenv("RPS_REORDER_REVIEW_DAYS", "1")),
        ),
        feedback=FeedbackConfig(
            correction_min_samples=int(os.getenv("RPS_CORRECTION_MIN_SAMPLES", "8")),
            correction_shrinkage=float(os.getenv("RPS_CORRECTION_SHRINKAGE", "0.60")),
            max_correction_pct=float(os.getenv("RPS_MAX_CORRECTION_PCT", "0.25")),
            retrain_error_threshold=float(
                os.getenv("RPS_RETRAIN_ERROR_THRESHOLD", "0.18")
            ),
            recent_feedback_weight=float(
                os.getenv("RPS_RECENT_FEEDBACK_WEIGHT", "1.75")
            ),
            high_error_weight=float(os.getenv("RPS_HIGH_ERROR_WEIGHT", "1.50")),
            manager_feedback_weight=float(
                os.getenv("RPS_MANAGER_FEEDBACK_WEIGHT", "2.25")
            ),
            max_sample_weight=float(os.getenv("RPS_MAX_SAMPLE_WEIGHT", "6.0")),
        ),
        api=ApiConfig(
            title=os.getenv("RPS_API_TITLE", "Restaurant Resource Planning System"),
            version=os.getenv("RPS_API_VERSION", "0.1.0"),
            host=os.getenv("RPS_API_HOST", "0.0.0.0"),
            port=int(os.getenv("RPS_API_PORT", "8000")),
            log_level=os.getenv("RPS_LOG_LEVEL", "INFO"),
        ),
    )
