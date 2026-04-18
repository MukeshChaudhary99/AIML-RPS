from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src.forecaster import RestaurantForecaster


@dataclass
class FeedbackLoopConfig:
    data_dir: Path = Path("V2Data")
    feedback_dir: Path = Path("feedback_store")
    correction_min_samples: int = 8
    correction_shrinkage: float = 0.60
    max_correction_pct: float = 0.25
    retrain_error_threshold: float = 0.18
    recent_feedback_weight: float = 1.75
    high_error_weight: float = 1.50
    manager_feedback_weight: float = 2.25
    max_sample_weight: float = 6.0

    # for startup testing loading of model
    startup_rows_per_group: int = 6
    startup_max_test_fraction: float = 0.18
    startup_recent_fraction: float = 0.40
    startup_random_state: int = 42


class ForecastFeedbackLoop:
    """
    Closed-loop forecasting system around the XGBoost covers forecaster.

    The loop supports:
    - prediction logging
    - actual outcome logging
    - manager note ingestion
    - scenario bias detection
    - correction factor generation
    - feedback-aware model retraining
    """

    PREDICTION_LOG_COLUMNS = [
        "timestamp",
        "model_version",
        "raw_predicted_covers",
        "predicted_covers",
        "correction_multiplier",
        "applied_scenarios",
        "promotion_flag",
        "holiday_flag",
        "event_flag",
        "rain_mm",
        "temp_c",
        "is_weekend",
        "season",
        "created_at",
    ]
    ACTUALS_LOG_COLUMNS = [
        "timestamp",
        "actual_covers",
        "actual_reservations",
        "actual_walk_ins",
        "ingested_at",
    ]
    MANAGER_FEEDBACK_COLUMNS = [
        "timestamp",
        "manager_note",
        "corrected_covers",
        "manager_reason",
        "manager_note_tags",
        "ingested_at",
    ]

    def __init__(
        self,
        config: Optional[FeedbackLoopConfig] = None,
        base_covers_df: Optional[pd.DataFrame] = None,
    ):
        self.config = config or FeedbackLoopConfig()
        self.data_dir = Path(self.config.data_dir)
        self.feedback_dir = Path(self.config.feedback_dir)
        self.feedback_dir.mkdir(parents=True, exist_ok=True)

        self.base_covers_df = (
            self._load_base_covers_df()
            if base_covers_df is None
            else self._prepare_base_covers_df(base_covers_df)
        )
        self.forecaster = RestaurantForecaster()
        self.feature_cols: list[str] = []
        self.training_history_df: Optional[pd.DataFrame] = None
        self.correction_table_df: Optional[pd.DataFrame] = None

        self.prediction_log_path = self.feedback_dir / "prediction_log.csv"
        self.actuals_log_path = self.feedback_dir / "actuals_log.csv"
        self.manager_feedback_path = self.feedback_dir / "manager_feedback.csv"
        self.scenario_corrections_path = self.feedback_dir / "scenario_corrections.csv"
        self.retraining_history_path = self.feedback_dir / "retraining_history.csv"

    def _load_base_covers_df(self) -> pd.DataFrame:
        historical_sales_df = pd.read_csv(
            self.data_dir / "historical_sales.csv", parse_dates=["timestamp"]
        )
        external_features_df = pd.read_csv(
            self.data_dir / "external_features.csv", parse_dates=["timestamp"]
        )
        return self._prepare_base_covers_df(
            historical_sales_df.merge(external_features_df, on="timestamp", how="inner")
        )

    def _prepare_base_covers_df(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        return df.sort_values("timestamp").drop_duplicates("timestamp", keep="last").reset_index(
            drop=True
        )

    def _read_log(self, path: Path, columns: list[str]) -> pd.DataFrame:
        if not path.exists():
            return pd.DataFrame(columns=columns)

        df = pd.read_csv(path)
        for col in ["timestamp", "created_at", "ingested_at", "trained_at"]:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col])
        return df

    def _write_upsert_log(
        self,
        path: Path,
        df: pd.DataFrame,
        columns: list[str],
        sort_column: str,
    ) -> pd.DataFrame:
        incoming_df = df.copy()
        incoming_df["timestamp"] = pd.to_datetime(incoming_df["timestamp"])

        existing_df = self._read_log(path, columns)
        combined_df = pd.concat([existing_df, incoming_df], ignore_index=True)

        for col in columns:
            if col not in combined_df.columns:
                combined_df[col] = np.nan

        combined_df = combined_df[columns]
        combined_df = combined_df.sort_values(sort_column).drop_duplicates(
            subset=["timestamp"], keep="last"
        )
        combined_df.to_csv(path, index=False)
        return combined_df

    def _tag_manager_note(self, note: str) -> str:
        if note is None or pd.isna(note):
            return ""

        note_text = str(note).lower()
        tags = []
        keyword_map = {
            "rain": ["rain", "storm", "monsoon", "wet"],
            "event": ["event", "match", "crowd", "concert", "party", "wedding"],
            "promotion": ["promotion", "promo", "offer", "discount", "campaign"],
            "holiday": ["holiday", "festival", "new year", "public holiday"],
            "heat": ["heat", "hot", "temperature"],
            "cold": ["cold", "winter", "chill"],
            "staff": ["staff", "crew", "shortage", "shift"],
            "stockout": ["stock", "stockout", "unavailable", "out of stock"],
            "system": ["system", "pos", "app", "power", "maintenance"],
        }

        for tag, keywords in keyword_map.items():
            if any(keyword in note_text for keyword in keywords):
                tags.append(tag)

        return "|".join(sorted(set(tags)))

    def load_prediction_log(self) -> pd.DataFrame:
        return self._read_log(self.prediction_log_path, self.PREDICTION_LOG_COLUMNS)

    def load_actuals_log(self) -> pd.DataFrame:
        return self._read_log(self.actuals_log_path, self.ACTUALS_LOG_COLUMNS)

    def load_manager_feedback_log(self) -> pd.DataFrame:
        return self._read_log(self.manager_feedback_path, self.MANAGER_FEEDBACK_COLUMNS)

    def log_predictions(self, predictions_df: pd.DataFrame, model_version: str) -> pd.DataFrame:
        log_df = predictions_df.copy()
        log_df["timestamp"] = pd.to_datetime(log_df["timestamp"])

        if "raw_predicted_covers" not in log_df.columns:
            log_df["raw_predicted_covers"] = log_df["predicted_covers"]
        if "correction_multiplier" not in log_df.columns:
            log_df["correction_multiplier"] = 1.0
        if "applied_scenarios" not in log_df.columns:
            log_df["applied_scenarios"] = ""

        for col in [
            "promotion_flag",
            "holiday_flag",
            "event_flag",
            "rain_mm",
            "temp_c",
            "is_weekend",
            "season",
        ]:
            if col not in log_df.columns:
                log_df[col] = np.nan

        log_df["model_version"] = model_version
        log_df["created_at"] = pd.Timestamp.utcnow().tz_localize(None)

        return self._write_upsert_log(
            self.prediction_log_path,
            log_df[self.PREDICTION_LOG_COLUMNS],
            self.PREDICTION_LOG_COLUMNS,
            "created_at",
        )

    def log_actuals(self, actuals_df: pd.DataFrame) -> pd.DataFrame:
        log_df = actuals_df.copy()
        log_df["timestamp"] = pd.to_datetime(log_df["timestamp"])

        if "actual_covers" not in log_df.columns and "covers" in log_df.columns:
            log_df = log_df.rename(columns={"covers": "actual_covers"})

        for col in ["actual_reservations", "actual_walk_ins"]:
            if col not in log_df.columns:
                log_df[col] = np.nan

        log_df["ingested_at"] = pd.Timestamp.utcnow().tz_localize(None)
        return self._write_upsert_log(
            self.actuals_log_path,
            log_df[self.ACTUALS_LOG_COLUMNS],
            self.ACTUALS_LOG_COLUMNS,
            "ingested_at",
        )

    def log_manager_feedback(self, manager_feedback_df: pd.DataFrame) -> pd.DataFrame:
        log_df = manager_feedback_df.copy()
        log_df["timestamp"] = pd.to_datetime(log_df["timestamp"])

        if "manager_note" not in log_df.columns:
            log_df["manager_note"] = ""
        if "corrected_covers" not in log_df.columns:
            log_df["corrected_covers"] = np.nan
        if "manager_reason" not in log_df.columns:
            log_df["manager_reason"] = np.nan

        log_df["manager_note_tags"] = log_df["manager_note"].apply(self._tag_manager_note)
        log_df["ingested_at"] = pd.Timestamp.utcnow().tz_localize(None)

        return self._write_upsert_log(
            self.manager_feedback_path,
            log_df[self.MANAGER_FEEDBACK_COLUMNS],
            self.MANAGER_FEEDBACK_COLUMNS,
            "ingested_at",
        )

    def build_feedback_frame(self) -> pd.DataFrame:
        prediction_log_df = self.load_prediction_log()
        actuals_log_df = self.load_actuals_log()
        manager_feedback_df = self.load_manager_feedback_log()

        if prediction_log_df.empty or actuals_log_df.empty:
            return pd.DataFrame()

        feedback_df = prediction_log_df.merge(
            actuals_log_df[["timestamp", "actual_covers"]],
            on="timestamp",
            how="inner",
        )
        feedback_df = feedback_df.merge(
            manager_feedback_df[
                ["timestamp", "manager_note", "corrected_covers", "manager_reason", "manager_note_tags"]
            ],
            on="timestamp",
            how="left",
        )

        context_cols = [
            "timestamp",
            "promotion_flag",
            "holiday_flag",
            "event_flag",
            "rain_mm",
            "temp_c",
            "is_weekend",
            "season",
        ]
        base_context_df = self.base_covers_df[context_cols].copy()
        feedback_df = feedback_df.merge(base_context_df, on="timestamp", how="left", suffixes=("", "_base"))

        for col in ["promotion_flag", "holiday_flag", "event_flag", "rain_mm", "temp_c", "is_weekend", "season"]:
            base_col = f"{col}_base"
            if base_col in feedback_df.columns:
                feedback_df[col] = feedback_df[col].fillna(feedback_df[base_col])
                feedback_df = feedback_df.drop(columns=[base_col])

        feedback_df["effective_actual_covers"] = feedback_df["corrected_covers"].fillna(
            feedback_df["actual_covers"]
        )
        feedback_df["error"] = (
            feedback_df["predicted_covers"] - feedback_df["effective_actual_covers"]
        )
        feedback_df["abs_error"] = feedback_df["error"].abs()
        feedback_df["error_pct"] = np.where(
            feedback_df["effective_actual_covers"] > 0,
            feedback_df["error"] / feedback_df["effective_actual_covers"],
            0.0,
        )
        feedback_df["predicted_bias_pct"] = np.where(
            feedback_df["predicted_covers"] > 0,
            feedback_df["error"] / feedback_df["predicted_covers"],
            0.0,
        )
        feedback_df["actual_to_pred_ratio"] = np.where(
            feedback_df["predicted_covers"] > 0,
            feedback_df["effective_actual_covers"] / feedback_df["predicted_covers"],
            1.0,
        )

        feedback_df["timestamp"] = pd.to_datetime(feedback_df["timestamp"])
        feedback_df["hour"] = feedback_df["timestamp"].dt.hour
        feedback_df["day_of_week"] = feedback_df["timestamp"].dt.dayofweek
        feedback_df["service_period"] = np.select(
            [
                feedback_df["hour"].between(8, 11),
                feedback_df["hour"].between(12, 15),
                feedback_df["hour"].between(16, 18),
            ],
            ["breakfast", "lunch", "afternoon"],
            default="dinner",
        )
        feedback_df["heavy_rain_flag"] = (feedback_df["rain_mm"].fillna(0) >= 15).astype(int)
        feedback_df["hot_weather_flag"] = (feedback_df["temp_c"].fillna(0) >= 35).astype(int)
        feedback_df["manager_note_tags"] = feedback_df["manager_note_tags"].fillna("")

        return feedback_df.sort_values("timestamp").reset_index(drop=True)

    def _scenario_rules(self, df: pd.DataFrame) -> dict[str, pd.Series]:
        note_tags = df.get("manager_note_tags", pd.Series("", index=df.index)).fillna("")
        note_has_rain = note_tags.str.contains("rain")
        note_has_event = note_tags.str.contains("event")
        note_has_promotion = note_tags.str.contains("promotion")
        note_has_holiday = note_tags.str.contains("holiday")
        note_has_heat = note_tags.str.contains("heat")

        return {
            "weekend": df["is_weekend"] == 1,
            "holiday": (df["holiday_flag"] == 1) | note_has_holiday,
            "event": (df["event_flag"] == 1) | note_has_event,
            "promotion": (df["promotion_flag"] == 1) | note_has_promotion,
            "heavy_rain": (df["heavy_rain_flag"] == 1) | note_has_rain,
            "hot_weather": (df["hot_weather_flag"] == 1) | note_has_heat,
            "breakfast": df["service_period"] == "breakfast",
            "lunch": df["service_period"] == "lunch",
            "afternoon": df["service_period"] == "afternoon",
            "dinner": df["service_period"] == "dinner",
        }

    def compute_scenario_corrections(
        self, feedback_df: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        if feedback_df is None:
            feedback_df = self.build_feedback_frame()

        if feedback_df.empty:
            empty_df = pd.DataFrame(
                columns=[
                    "scenario",
                    "observations",
                    "support_weight",
                    "mean_error",
                    "mean_error_pct",
                    "median_actual_to_pred_ratio",
                    "correction_factor",
                    "recommended_adjustment_pct",
                    "is_actionable",
                ]
            )
            empty_df.to_csv(self.scenario_corrections_path, index=False)
            self.correction_table_df = empty_df
            return empty_df

        rows = []
        for scenario, mask in self._scenario_rules(feedback_df).items():
            subset_df = feedback_df[mask].copy()
            if subset_df.empty:
                continue

            support_weight = len(subset_df) / (
                len(subset_df) + self.config.correction_min_samples
            )
            median_ratio = float(subset_df["actual_to_pred_ratio"].median())
            shrunk_factor = 1 + (
                (median_ratio - 1) * support_weight * self.config.correction_shrinkage
            )
            correction_factor = float(
                np.clip(
                    shrunk_factor,
                    1 - self.config.max_correction_pct,
                    1 + self.config.max_correction_pct,
                )
            )

            rows.append(
                {
                    "scenario": scenario,
                    "observations": int(len(subset_df)),
                    "support_weight": round(float(support_weight), 4),
                    "mean_error": round(float(subset_df["error"].mean()), 4),
                    "mean_error_pct": round(
                        float(subset_df["predicted_bias_pct"].mean() * 100), 4
                    ),
                    "median_actual_to_pred_ratio": round(median_ratio, 4),
                    "correction_factor": round(correction_factor, 4),
                    "recommended_adjustment_pct": round(
                        (correction_factor - 1) * 100, 4
                    ),
                    "is_actionable": int(len(subset_df) >= self.config.correction_min_samples),
                }
            )

        correction_table_df = pd.DataFrame(rows).sort_values(
            ["is_actionable", "observations", "scenario"],
            ascending=[False, False, True],
        )
        correction_table_df.to_csv(self.scenario_corrections_path, index=False)
        self.correction_table_df = correction_table_df
        return correction_table_df

    def apply_scenario_corrections(
        self,
        predictions_df: pd.DataFrame,
        correction_table_df: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        corrected_df = predictions_df.copy()
        corrected_df["timestamp"] = pd.to_datetime(corrected_df["timestamp"])

        if "raw_predicted_covers" not in corrected_df.columns:
            corrected_df["raw_predicted_covers"] = corrected_df["predicted_covers"]
        for col in ["promotion_flag", "holiday_flag", "event_flag", "rain_mm", "temp_c", "season"]:
            if col not in corrected_df.columns:
                corrected_df[col] = np.nan

        corrected_df["hour"] = corrected_df["timestamp"].dt.hour
        if "is_weekend" not in corrected_df.columns:
            corrected_df["is_weekend"] = (
                corrected_df["timestamp"].dt.dayofweek >= 5
            ).astype(int)
        if "heavy_rain_flag" not in corrected_df.columns:
            corrected_df["heavy_rain_flag"] = (
                corrected_df["rain_mm"].fillna(0) >= 15
            ).astype(int)
        if "hot_weather_flag" not in corrected_df.columns:
            corrected_df["hot_weather_flag"] = (
                corrected_df["temp_c"].fillna(0) >= 35
            ).astype(int)
        if "service_period" not in corrected_df.columns:
            corrected_df["service_period"] = np.select(
                [
                    corrected_df["hour"].between(8, 11),
                    corrected_df["hour"].between(12, 15),
                    corrected_df["hour"].between(16, 18),
                ],
                ["breakfast", "lunch", "afternoon"],
                default="dinner",
            )

        if correction_table_df is None:
            correction_table_df = self.correction_table_df
        if correction_table_df is None:
            correction_table_df = self.compute_scenario_corrections()
        if correction_table_df.empty:
            corrected_df["correction_multiplier"] = 1.0
            corrected_df["applied_scenarios"] = ""
            corrected_df["predicted_covers"] = (
                corrected_df["raw_predicted_covers"].clip(lower=0).round().astype(int)
            )
            return corrected_df

        active_table_df = correction_table_df[correction_table_df["is_actionable"] == 1].copy()
        factor_map = (
            active_table_df.set_index("scenario")[["correction_factor", "support_weight"]]
            .to_dict("index")
        )

        multipliers = []
        scenario_strings = []

        for _, row in corrected_df.iterrows():
            active_factors = []
            active_weights = []
            active_names = []

            for scenario, mask in self._scenario_rules(pd.DataFrame([row])).items():
                if bool(mask.iloc[0]) and scenario in factor_map:
                    active_factors.append(factor_map[scenario]["correction_factor"])
                    active_weights.append(factor_map[scenario]["support_weight"])
                    active_names.append(scenario)

            if active_factors:
                weighted_adjustment = np.average(
                    np.array(active_factors) - 1,
                    weights=np.array(active_weights),
                )
                multiplier = 1 + weighted_adjustment
            else:
                multiplier = 1.0

            multiplier = float(
                np.clip(
                    multiplier,
                    1 - self.config.max_correction_pct,
                    1 + self.config.max_correction_pct,
                )
            )
            multipliers.append(multiplier)
            scenario_strings.append("|".join(active_names))

        corrected_df["correction_multiplier"] = multipliers
        corrected_df["applied_scenarios"] = scenario_strings
        corrected_df["predicted_covers"] = (
            corrected_df["raw_predicted_covers"] * corrected_df["correction_multiplier"]
        ).clip(lower=0).round().astype(int)
        corrected_df["raw_predicted_covers"] = (
            corrected_df["raw_predicted_covers"].clip(lower=0).round().astype(int)
        )

        return corrected_df

    def _build_training_history(self) -> pd.DataFrame:
        training_history_df = self.base_covers_df.copy()
        feedback_df = self.build_feedback_frame()
        if feedback_df.empty:
            return training_history_df

        feedback_history_df = feedback_df.copy()
        feedback_history_df["covers"] = feedback_history_df["effective_actual_covers"]

        expected_columns = list(training_history_df.columns)
        for col in expected_columns:
            if col not in feedback_history_df.columns:
                feedback_history_df[col] = np.nan

        feedback_history_df = feedback_history_df[expected_columns]
        training_history_df = pd.concat(
            [training_history_df, feedback_history_df], ignore_index=True
        )
        training_history_df = self._prepare_base_covers_df(training_history_df)
        return training_history_df

    def _build_sample_weights(
        self,
        model_frame_df: pd.DataFrame,
        feedback_df: pd.DataFrame,
        correction_table_df: pd.DataFrame,
    ) -> pd.Series:
        weights = pd.Series(1.0, index=model_frame_df.index)
        if model_frame_df.empty:
            return weights

        if not feedback_df.empty:
            feedback_features_df = feedback_df[
                ["timestamp", "abs_error", "manager_note", "manager_note_tags"]
            ].copy()
            model_frame_df = model_frame_df.merge(
                feedback_features_df,
                on="timestamp",
                how="left",
            )

            weights.loc[model_frame_df["abs_error"].notna()] *= self.config.recent_feedback_weight

            if model_frame_df["abs_error"].notna().any():
                error_threshold = float(
                    np.nanpercentile(model_frame_df["abs_error"].dropna(), 75)
                )
                weights.loc[model_frame_df["abs_error"] >= error_threshold] *= (
                    self.config.high_error_weight
                )

            weights.loc[
                model_frame_df["manager_note"].fillna("").str.len() > 0
            ] *= self.config.manager_feedback_weight

        if not correction_table_df.empty:
            actionable_df = correction_table_df[
                (correction_table_df["is_actionable"] == 1)
                & (correction_table_df["recommended_adjustment_pct"].abs() >= 2.0)
            ]
            scenario_masks = self._scenario_rules(model_frame_df.assign(
                service_period=np.select(
                    [
                        model_frame_df.get("period_breakfast", 0) == 1,
                        model_frame_df.get("period_lunch", 0) == 1,
                        model_frame_df.get("period_afternoon", 0) == 1,
                    ],
                    ["breakfast", "lunch", "afternoon"],
                    default="dinner",
                ),
                heavy_rain_flag=model_frame_df.get("heavy_rain_flag", 0),
                hot_weather_flag=model_frame_df.get("temp_hot", 0),
            ))

            for _, row in actionable_df.iterrows():
                scenario = row["scenario"]
                if scenario not in scenario_masks:
                    continue
                adjustment_strength = min(
                    0.75, abs(float(row["correction_factor"]) - 1.0) * 3.0
                )
                weights.loc[scenario_masks[scenario]] *= 1 + adjustment_strength

        return weights.clip(lower=1.0, upper=self.config.max_sample_weight)

    def _build_startup_validation_split(
        self,
        model_frame_df: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
        total_rows = len(model_frame_df)
        if total_rows < 120:
            holdout_rows = max(12, int(total_rows * 0.10))
            holdout_rows = min(holdout_rows, max(total_rows - 24, 0))
            test_df = model_frame_df.tail(holdout_rows).copy()
            train_df = model_frame_df.iloc[:-holdout_rows].copy()
            return train_df, test_df, {
                "split_strategy": "recent_tail_fallback",
                "candidate_rows": holdout_rows,
                "selected_rows": len(test_df),
                "group_count": 1,
                "scenario_counts": {},
                "group_counts": {},
            }

        candidate_rows = max(
            int(total_rows * self.config.startup_recent_fraction),
            self.config.startup_rows_per_group * 12,
        )
        candidate_rows = min(candidate_rows, total_rows)
        candidate_df = model_frame_df.tail(candidate_rows).copy()
        candidate_df = candidate_df.reset_index().rename(columns={"index": "original_index"})

        def season_label(row: pd.Series) -> str:
            for season_name in ["winter", "summer", "monsoon", "festive"]:
                if row.get(f"season_{season_name}", 0) == 1:
                    return season_name
            return "unknown"

        def service_period_label(row: pd.Series) -> str:
            if row.get("period_breakfast", 0) == 1:
                return "breakfast"
            if row.get("period_lunch", 0) == 1:
                return "lunch"
            if row.get("period_afternoon", 0) == 1:
                return "afternoon"
            if row.get("period_late_night", 0) == 1:
                return "late_night"
            return "dinner"

        def scenario_label(row: pd.Series) -> str:
            labels = []
            if row.get("holiday_flag", 0) == 1:
                labels.append("holiday")
            if row.get("event_flag", 0) == 1:
                labels.append("event")
            if row.get("promotion_flag", 0) == 1:
                labels.append("promotion")
            if row.get("heavy_rain_flag", 0) == 1:
                labels.append("heavy_rain")
            elif row.get("rain_flag", 0) == 1:
                labels.append("rain")
            if row.get("temp_hot", 0) == 1 or row.get("hot_weather_flag", 0) == 1:
                labels.append("hot_weather")
            if row.get("is_weekend", 0) == 1:
                labels.append("weekend")
            if not labels:
                labels.append("weekday_normal")
            return "+".join(labels)

        candidate_df["eval_season"] = candidate_df.apply(season_label, axis=1)
        candidate_df["eval_period"] = candidate_df.apply(service_period_label, axis=1)
        candidate_df["eval_scenario"] = candidate_df.apply(scenario_label, axis=1)
        candidate_df["eval_group"] = (
            candidate_df["eval_season"]
            + "|"
            + candidate_df["eval_period"]
            + "|"
            + candidate_df["eval_scenario"]
        )

        max_test_rows = max(
            self.config.startup_rows_per_group * 8,
            int(total_rows * self.config.startup_max_test_fraction),
        )
        max_test_rows = min(max_test_rows, max(total_rows - 60, 0))
        if max_test_rows <= 0:
            return model_frame_df.copy(), pd.DataFrame(), {
                "split_strategy": "startup_diverse",
                "candidate_rows": candidate_rows,
                "selected_rows": 0,
                "group_count": 0,
                "scenario_counts": {},
                "group_counts": {},
            }

        rng = np.random.default_rng(self.config.startup_random_state)
        selected_indices: set[int] = set()
        group_counts: dict[str, int] = {}
        scenario_counts: dict[str, int] = {}

        grouped = candidate_df.groupby("eval_group", sort=True)
        for group_name, group_df in grouped:
            if len(selected_indices) >= max_test_rows:
                break

            sample_n = min(
                self.config.startup_rows_per_group,
                len(group_df),
                max_test_rows - len(selected_indices),
            )
            if sample_n <= 0:
                continue

            chosen = rng.choice(group_df["original_index"].to_numpy(), size=sample_n, replace=False)
            selected_indices.update(int(idx) for idx in chosen)
            group_counts[group_name] = int(sample_n)
            scenario_name = str(group_df["eval_scenario"].iloc[0])
            scenario_counts[scenario_name] = scenario_counts.get(scenario_name, 0) + int(sample_n)

        if len(selected_indices) < max_test_rows:
            remaining_df = candidate_df[
                ~candidate_df["original_index"].isin(selected_indices)
            ].copy()
            remaining_needed = min(max_test_rows - len(selected_indices), len(remaining_df))
            if remaining_needed > 0:
                chosen = rng.choice(
                    remaining_df["original_index"].to_numpy(),
                    size=remaining_needed,
                    replace=False,
                )
                selected_indices.update(int(idx) for idx in chosen)

        test_index_list = sorted(selected_indices)
        test_df = model_frame_df.loc[test_index_list].copy()
        train_df = model_frame_df.drop(index=test_index_list).copy()

        return train_df, test_df, {
            "split_strategy": "startup_diverse",
            "candidate_rows": int(candidate_rows),
            "selected_rows": int(len(test_df)),
            "group_count": int(len(group_counts)),
            "scenario_counts": dict(sorted(scenario_counts.items())),
            "group_counts": dict(sorted(group_counts.items())),
        }

    def _build_prediction_summary(
        self,
        test_df: pd.DataFrame,
        predictions: np.ndarray,
    ) -> pd.DataFrame:
        if test_df.empty or len(predictions) == 0:
            return pd.DataFrame()

        summary_df = pd.DataFrame(
            {
                "timestamp": pd.to_datetime(test_df["timestamp"]),
                "actual_covers": test_df["covers"].astype(float).round(2),
                "predicted_covers": np.round(predictions, 2),
            }
        )
        summary_df["error"] = (
            summary_df["predicted_covers"] - summary_df["actual_covers"]
        ).round(2)
        summary_df["abs_error"] = summary_df["error"].abs().round(2)
        summary_df["error_pct"] = np.where(
            summary_df["actual_covers"] > 0,
            (summary_df["error"] / summary_df["actual_covers"]) * 100,
            0.0,
        ).round(2)

        feature_candidates = [
            "hour",
            "day_of_week",
            "is_weekend",
            "holiday_flag",
            "event_flag",
            "promotion_flag",
            "rain_mm",
            "temp_c",
            "heavy_rain_flag",
            "temp_hot",
        ]
        for column in feature_candidates:
            if column in test_df.columns:
                summary_df[column] = test_df[column].values

        summary_df["season"] = "unknown"
        for season_name in ["winter", "summer", "monsoon", "festive"]:
            season_col = f"season_{season_name}"
            if season_col in test_df.columns:
                summary_df.loc[test_df[season_col] == 1, "season"] = season_name

        summary_df["service_period"] = "dinner"
        service_map = {
            "period_pre_service": "pre_service",
            "period_breakfast": "breakfast",
            "period_lunch": "lunch",
            "period_afternoon": "afternoon",
            "period_dinner": "dinner",
            "period_late_night": "late_night",
        }
        for column, label in service_map.items():
            if column in test_df.columns:
                summary_df.loc[test_df[column] == 1, "service_period"] = label

        scenario_parts = []
        scenario_parts.append(
            np.where(summary_df.get("holiday_flag", 0) == 1, "holiday", "")
        )
        scenario_parts.append(
            np.where(summary_df.get("event_flag", 0) == 1, "event", "")
        )
        scenario_parts.append(
            np.where(summary_df.get("promotion_flag", 0) == 1, "promotion", "")
        )
        scenario_parts.append(
            np.where(summary_df.get("heavy_rain_flag", 0) == 1, "heavy_rain", "")
        )
        if "rain_mm" in summary_df.columns:
            scenario_parts.append(
                np.where(
                    (summary_df["rain_mm"].fillna(0) > 0)
                    & (summary_df.get("heavy_rain_flag", 0) == 0),
                    "rain",
                    "",
                )
            )
        scenario_parts.append(
            np.where(summary_df.get("temp_hot", 0) == 1, "hot_weather", "")
        )
        scenario_parts.append(
            np.where(summary_df.get("is_weekend", 0) == 1, "weekend", "")
        )

        eval_scenarios = []
        for idx in range(len(summary_df)):
            labels = [part[idx] for part in scenario_parts if part[idx]]
            eval_scenarios.append("+".join(labels) if labels else "weekday_normal")
        summary_df["eval_scenario"] = eval_scenarios

        ordered_cols = [
            "timestamp",
            "actual_covers",
            "predicted_covers",
            "error",
            "abs_error",
            "error_pct",
            "hour",
            "service_period",
            "season",
            "eval_scenario",
            "holiday_flag",
            "event_flag",
            "promotion_flag",
            "is_weekend",
            "rain_mm",
            "temp_c",
        ]
        available_cols = [col for col in ordered_cols if col in summary_df.columns]
        summary_df = summary_df[available_cols]

        return summary_df.sort_values("timestamp").reset_index(drop=True)

    def _build_scenario_error_summary(
        self,
        prediction_summary_df: pd.DataFrame,
    ) -> pd.DataFrame:
        if prediction_summary_df.empty:
            return pd.DataFrame()

        grouped_df = (
            prediction_summary_df.groupby(
                ["eval_scenario", "service_period", "season"], as_index=False
            )
            .agg(
                rows=("timestamp", "count"),
                actual_covers_mean=("actual_covers", "mean"),
                predicted_covers_mean=("predicted_covers", "mean"),
                mae=("abs_error", "mean"),
                rmse=("error", lambda s: float(np.sqrt(np.mean(np.square(s))))),
                bias=("error", "mean"),
                wape_num=("abs_error", "sum"),
                wape_den=("actual_covers", "sum"),
            )
        )
        grouped_df["wape"] = np.where(
            grouped_df["wape_den"] > 0,
            grouped_df["wape_num"] / grouped_df["wape_den"],
            0.0,
        )
        grouped_df = grouped_df.drop(columns=["wape_num", "wape_den"])

        rounded_cols = [
            "actual_covers_mean",
            "predicted_covers_mean",
            "mae",
            "rmse",
            "bias",
            "wape",
        ]
        grouped_df[rounded_cols] = grouped_df[rounded_cols].round(4)
        return grouped_df.sort_values(["rows", "mae"], ascending=[False, False]).reset_index(
            drop=True
        )

    def train_feedback_aware_model(
        self,
        holdout_hours: int = 0,
        split_strategy: str = "tail",
    ) -> dict:
        feedback_df = self.build_feedback_frame()
        correction_table_df = self.compute_scenario_corrections(feedback_df)
        training_history_df = self._build_training_history()

        model_frame_df, feature_cols = self.forecaster.prepare_model_frame(training_history_df)
        self.feature_cols = feature_cols
        self.forecaster.feature_cols = feature_cols

        split_summary = {
            "split_strategy": split_strategy,
            "candidate_rows": 0,
            "selected_rows": 0,
            "group_count": 0,
            "scenario_counts": {},
            "group_counts": {},
        }
        if split_strategy == "startup_diverse":
            train_df, test_df, split_summary = self._build_startup_validation_split(
                model_frame_df
            )
        elif holdout_hours > 0 and holdout_hours < len(model_frame_df):
            train_df = model_frame_df.iloc[:-holdout_hours].copy()
            test_df = model_frame_df.iloc[-holdout_hours:].copy()
            split_summary = {
                "split_strategy": "tail",
                "candidate_rows": int(len(model_frame_df)),
                "selected_rows": int(len(test_df)),
                "group_count": 1,
                "scenario_counts": {},
                "group_counts": {"tail_holdout": int(len(test_df))},
            }
        else:
            train_df = model_frame_df.copy()
            test_df = pd.DataFrame()
            split_summary = {
                "split_strategy": "full_train",
                "candidate_rows": int(len(model_frame_df)),
                "selected_rows": 0,
                "group_count": 0,
                "scenario_counts": {},
                "group_counts": {},
            }

        sample_weights = self._build_sample_weights(train_df.copy(), feedback_df, correction_table_df)

        X_train = train_df[feature_cols]
        y_train = train_df["covers"]
        X_test = pd.DataFrame()
        y_test = pd.Series(dtype=float)
        eval_sample_weights = None
        if not test_df.empty:
            X_test = test_df[feature_cols]
            y_test = test_df["covers"]
            eval_sample_weights = np.ones(len(test_df))

        self.forecaster.train(
            X_train,
            y_train,
            sample_weight=sample_weights.values,
            X_eval=X_test if not test_df.empty else None,
            y_eval=y_test if not test_df.empty else None,
            eval_sample_weight=eval_sample_weights,
        )

        metrics = {}
        test_predictions = np.array([])
        prediction_summary_df = pd.DataFrame()
        scenario_error_summary_df = pd.DataFrame()
        if not test_df.empty:
            test_predictions = self.forecaster.predict(X_test)
            metrics = self.forecaster.evaluate(X_test, y_test)
            prediction_summary_df = self._build_prediction_summary(test_df, test_predictions)
            scenario_error_summary_df = self._build_scenario_error_summary(
                prediction_summary_df
            )

        self.training_history_df = training_history_df
        self.correction_table_df = correction_table_df

        retraining_row = pd.DataFrame(
            [
                {
                    "trained_at": pd.Timestamp.utcnow().tz_localize(None),
                    "training_rows": len(train_df),
                    "feedback_rows": len(feedback_df),
                    "holdout_rows": len(test_df),
                    "split_strategy": split_summary["split_strategy"],
                    "mae": metrics.get("mae"),
                    "rmse": metrics.get("rmse"),
                    "wape": metrics.get("wape"),
                }
            ]
        )
        existing_history_df = (
            pd.read_csv(self.retraining_history_path)
            if self.retraining_history_path.exists()
            else pd.DataFrame(columns=retraining_row.columns)
        )
        pd.concat([existing_history_df, retraining_row], ignore_index=True).to_csv(
            self.retraining_history_path,
            index=False,
        )

        return {
            "train_df": train_df,
            "test_df": test_df,
            "feature_cols": feature_cols,
            "metrics": metrics,
            "test_predictions": test_predictions,
            "prediction_summary_df": prediction_summary_df,
            "scenario_error_summary_df": scenario_error_summary_df,
            "feedback_df": feedback_df,
            "correction_table_df": correction_table_df,
            "split_summary": split_summary,
        }

    def should_retrain(self) -> bool:
        feedback_df = self.build_feedback_frame()
        if feedback_df.empty:
            return False

        recent_feedback_df = feedback_df.sort_values("timestamp").tail(
            max(20, self.config.correction_min_samples)
        )
        recent_wape = np.sum(recent_feedback_df["abs_error"]) / np.sum(
            recent_feedback_df["effective_actual_covers"].clip(lower=1)
        )
        return recent_wape >= self.config.retrain_error_threshold

    def forecast_with_feedback(
        self,
        future_context_df: pd.DataFrame,
        model_version: str = "xgb_feedback_v1",
        auto_retrain: bool = True,
    ) -> pd.DataFrame:
        if self.training_history_df is None or self.forecaster.model is None:
            self.train_feedback_aware_model(holdout_hours=0)
        elif auto_retrain and self.should_retrain():
            self.train_feedback_aware_model(holdout_hours=0)

        future_context_df = future_context_df.copy()
        future_context_df["timestamp"] = pd.to_datetime(future_context_df["timestamp"])
        future_feature_df = self.forecaster.prepare_future_frame(
            future_context_df,
            self.training_history_df,
        )

        raw_pred = self.forecaster.predict(future_feature_df[self.feature_cols])

        predictions_df = future_context_df.copy()
        predictions_df["raw_predicted_covers"] = raw_pred
        predictions_df["predicted_covers"] = raw_pred

        predictions_df = self.apply_scenario_corrections(predictions_df, self.correction_table_df)
        self.log_predictions(predictions_df, model_version=model_version)
        return predictions_df.sort_values("timestamp").reset_index(drop=True)

    def update_with_feedback(
        self,
        actuals_df: pd.DataFrame,
        manager_feedback_df: Optional[pd.DataFrame] = None,
        retrain: bool = True,
    ) -> dict:
        self.log_actuals(actuals_df)
        if manager_feedback_df is not None and not manager_feedback_df.empty:
            self.log_manager_feedback(manager_feedback_df)

        feedback_df = self.build_feedback_frame()
        correction_table_df = self.compute_scenario_corrections(feedback_df)

        retrained = False
        retrain_result = {}
        if retrain and self.should_retrain():
            retrain_result = self.train_feedback_aware_model(holdout_hours=0)
            retrained = True

        summary = {
            "feedback_rows": int(len(feedback_df)),
            "mean_abs_error": float(feedback_df["abs_error"].mean()) if not feedback_df.empty else 0.0,
            "mean_error_pct": float(feedback_df["predicted_bias_pct"].mean()) if not feedback_df.empty else 0.0,
            "actionable_scenarios": int(
                (correction_table_df["is_actionable"] == 1).sum()
            ) if not correction_table_df.empty else 0,
            "retrained": retrained,
        }
        summary.update(retrain_result.get("metrics", {}))
        return summary
