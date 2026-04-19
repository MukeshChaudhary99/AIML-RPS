from __future__ import annotations

from html import escape
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src.config import AppConfig, load_app_config
from src.feedback_loop import FeedbackLoopConfig, ForecastFeedbackLoop
from src.ingredient_planner import IngredientPlanner, IngredientPlannerConfig
from src.staff_planner import PlannerConfig, StaffPlanner
from src.utils import format_dataframe_block, format_key_value_block, get_logger


class RestaurantPlanningService:
    def __init__(self, app_config: Optional[AppConfig] = None):
        self.config = app_config or load_app_config()
        self.logger = get_logger(__name__, self.config.api.log_level)

        self.feedback_loop = ForecastFeedbackLoop(
            config=FeedbackLoopConfig(
                data_dir=self.config.data.data_dir,
                feedback_dir=self.config.data.feedback_dir,
                correction_min_samples=self.config.feedback.correction_min_samples,
                correction_shrinkage=self.config.feedback.correction_shrinkage,
                max_correction_pct=self.config.feedback.max_correction_pct,
                retrain_error_threshold=self.config.feedback.retrain_error_threshold,
                recent_feedback_weight=self.config.feedback.recent_feedback_weight,
                high_error_weight=self.config.feedback.high_error_weight,
                manager_feedback_weight=self.config.feedback.manager_feedback_weight,
                max_sample_weight=self.config.feedback.max_sample_weight,
                startup_rows_per_group=self.config.forecast.startup_rows_per_group,
                startup_max_test_fraction=self.config.forecast.startup_max_test_fraction,
                startup_recent_fraction=self.config.forecast.startup_recent_fraction,
                startup_random_state=self.config.forecast.startup_random_state,
                xgb_params={
                    "n_jobs": self.config.forecast.n_jobs,
                    "n_estimators": self.config.forecast.n_estimators,
                    "max_depth": self.config.forecast.max_depth,
                    "learning_rate": self.config.forecast.learning_rate,
                    "subsample": self.config.forecast.subsample,
                    "colsample_bytree": self.config.forecast.colsample_bytree,
                    "min_child_weight": self.config.forecast.min_child_weight,
                    "gamma": self.config.forecast.gamma,
                    "reg_alpha": self.config.forecast.reg_alpha,
                    "reg_lambda": self.config.forecast.reg_lambda,
                },
                early_stopping_rounds=self.config.forecast.early_stopping_rounds,
            )
        )
        self.staff_planner = StaffPlanner(
            config=PlannerConfig(
                data_dir=self.config.data.data_dir,
                prep_buffer_multiplier=self.config.staff.prep_buffer_multiplier,
                covers_buffer_multiplier=self.config.staff.covers_buffer_multiplier,
            )
        )
        self.ingredient_planner = IngredientPlanner(
            config=IngredientPlannerConfig(
                data_dir=self.config.data.data_dir,
                menu_qty_buffer_multiplier=self.config.ingredient.menu_qty_buffer_multiplier,
                ingredient_qty_buffer_multiplier=self.config.ingredient.ingredient_qty_buffer_multiplier,
                reorder_review_days=self.config.ingredient.reorder_review_days,
            )
        )

    def _serialize_scalar(self, value):
        if pd.isna(value):
            return None
        if isinstance(value, pd.Timestamp):
            return value.isoformat()
        if isinstance(value, np.integer):
            return int(value)
        if isinstance(value, np.floating):
            return float(round(value, 3))
        return value

    def build_forecast_response(self, forecast_df: pd.DataFrame) -> list[dict]:
        if forecast_df.empty:
            return []

        ordered_df = forecast_df.copy()
        ordered_df["timestamp"] = pd.to_datetime(ordered_df["timestamp"])
        ordered_df = ordered_df.sort_values("timestamp")
        response_rows = []
        for _, row in ordered_df.iterrows():
            response_rows.append(
                {
                    "timestamp": row["timestamp"].isoformat(),
                    "predicted_covers": int(row["predicted_covers"]),
                }
            )
        return response_rows

    def build_staff_response(
        self,
        forecast_df: pd.DataFrame,
        hourly_staff_plan_df: pd.DataFrame,
    ) -> list[dict]:
        if forecast_df.empty:
            return []

        forecast_df = forecast_df.copy()
        forecast_df["timestamp"] = pd.to_datetime(forecast_df["timestamp"])
        hourly_staff_plan_df = hourly_staff_plan_df.copy()
        hourly_staff_plan_df["timestamp"] = pd.to_datetime(hourly_staff_plan_df["timestamp"])

        staff_by_timestamp = {}
        if not hourly_staff_plan_df.empty:
            ordered_staff_df = hourly_staff_plan_df.sort_values(
                ["timestamp", "station", "role"]
            )
            for timestamp, group_df in ordered_staff_df.groupby("timestamp"):
                staff_by_timestamp[timestamp] = [
                    {
                        "role": row["role"],
                        "station": row["station"],
                        "required_staff": int(row["required_staff"]),
                    }
                    for _, row in group_df.iterrows()
                ]

        hourly_rows = []
        for _, row in forecast_df.sort_values("timestamp").iterrows():
            timestamp = row["timestamp"]
            hourly_rows.append(
                {
                    "timestamp": timestamp.isoformat(),
                    "predicted_covers": int(row["predicted_covers"]),
                    "staff": staff_by_timestamp.get(timestamp, []),
                }
            )
        return hourly_rows

    def build_ingredient_hourly_response(
        self,
        forecast_df: pd.DataFrame,
        ingredient_hourly_df: pd.DataFrame,
    ) -> list[dict]:
        if forecast_df.empty:
            return []

        forecast_df = forecast_df.copy()
        forecast_df["timestamp"] = pd.to_datetime(forecast_df["timestamp"])
        ingredient_hourly_df = ingredient_hourly_df.copy()
        ingredient_hourly_df["timestamp"] = pd.to_datetime(ingredient_hourly_df["timestamp"])

        ingredients_by_timestamp = {}
        if not ingredient_hourly_df.empty:
            ordered_ingredient_df = ingredient_hourly_df.sort_values(
                ["timestamp", "ingredient_name"]
            )
            for timestamp, group_df in ordered_ingredient_df.groupby("timestamp"):
                ingredients_by_timestamp[timestamp] = [
                    {
                        "ingredient_id": row["ingredient_id"],
                        "ingredient_name": row["ingredient_name"],
                        "required_qty": float(round(row["required_qty"], 3)),
                        "uom": row["uom"],
                    }
                    for _, row in group_df.iterrows()
                ]

        hourly_rows = []
        for _, row in forecast_df.sort_values("timestamp").iterrows():
            timestamp = row["timestamp"]
            hourly_rows.append(
                {
                    "timestamp": timestamp.isoformat(),
                    "predicted_covers": int(row["predicted_covers"]),
                    "ingredients": ingredients_by_timestamp.get(timestamp, []),
                }
            )
        return hourly_rows

    def build_purchase_response(self, purchase_recommendation_df: pd.DataFrame) -> list[dict]:
        if purchase_recommendation_df.empty:
            return []

        ordered_df = purchase_recommendation_df.sort_values(
            ["recommended_order_qty", "ingredient_name"], ascending=[False, True]
        )
        rows = []
        for _, row in ordered_df.iterrows():
            rows.append(
                {
                    "ingredient_id": row["ingredient_id"],
                    "ingredient_name": row["ingredient_name"],
                    "recommended_order_qty": float(round(row["recommended_order_qty"], 3)),
                    "uom": row["uom"],
                    "lead_time_days": int(row["lead_time_days"]),
                    "shelf_life_days": int(row["shelf_life_days"]),
                }
            )
        return rows

    def build_full_day_response(
        self,
        forecast_df: pd.DataFrame,
        hourly_staff_plan_df: pd.DataFrame,
        ingredient_hourly_df: pd.DataFrame,
        purchase_recommendation_df: pd.DataFrame,
    ) -> dict:
        forecast_rows = self.build_forecast_response(forecast_df)
        staff_rows = self.build_staff_response(forecast_df, hourly_staff_plan_df)
        ingredient_rows = self.build_ingredient_hourly_response(
            forecast_df, ingredient_hourly_df
        )

        staff_map = {row["timestamp"]: row["staff"] for row in staff_rows}
        ingredient_map = {
            row["timestamp"]: row["ingredients"] for row in ingredient_rows
        }

        hourly_plan = []
        for forecast_row in forecast_rows:
            timestamp = forecast_row["timestamp"]
            hourly_plan.append(
                {
                    "timestamp": timestamp,
                    "predicted_covers": forecast_row["predicted_covers"],
                    "staff": staff_map.get(timestamp, []),
                    "ingredients": ingredient_map.get(timestamp, []),
                }
            )

        service_date = None
        if hourly_plan:
            service_date = hourly_plan[0]["timestamp"][:10]

        return {
            "date": service_date,
            "hours": hourly_plan,
            "purchase_recommendation": self.build_purchase_response(
                purchase_recommendation_df
            ),
        }

    def load_model_on_startup(self) -> dict:
        self.logger.info("Loading forecasting model during server startup")
        result = self.feedback_loop.train_feedback_aware_model(
            holdout_hours=0,
            split_strategy="startup_diverse",
        )
        metrics = result.get("metrics", {})
        split_summary = result.get("split_summary", {})
        train_rows = len(result.get("train_df", pd.DataFrame()))
        test_rows = len(result.get("test_df", pd.DataFrame()))
        feedback_rows = len(result.get("feedback_df", pd.DataFrame()))
        feature_count = len(result.get("feature_cols", []))
        prediction_summary_df = result.get("prediction_summary_df", pd.DataFrame())
        scenario_error_summary_df = result.get("scenario_error_summary_df", pd.DataFrame())

        overview_block = format_key_value_block(
            "Startup Model Summary",
            [
                ("loaded", self.feedback_loop.forecaster.model is not None),
                ("train_rows", train_rows),
                ("test_rows", test_rows),
                ("feedback_rows", feedback_rows),
                ("feature_count", feature_count),
                ("mae", round(metrics.get("mae", 0.0), 4) if metrics else None),
                ("rmse", round(metrics.get("rmse", 0.0), 4) if metrics else None),
                ("wape", round(metrics.get("wape", 0.0), 4) if metrics else None),
                ("mape", round(metrics.get("mape", 0.0), 4) if metrics else None),
                ("bias", round(metrics.get("bias", 0.0), 4) if metrics else None),
                ("split_strategy", split_summary.get("split_strategy")),
                ("candidate_rows", split_summary.get("candidate_rows")),
                ("selected_rows", split_summary.get("selected_rows")),
                ("group_count", split_summary.get("group_count")),
            ],
        )
        scenario_block = format_key_value_block(
            "Startup Scenario Coverage",
            list(split_summary.get("scenario_counts", {}).items()),
        )
        prediction_block = format_dataframe_block(
            "Startup Prediction Preview",
            prediction_summary_df,
            max_rows=20,
        )
        scenario_error_block = format_dataframe_block(
            "Startup Scenario Error Summary",
            scenario_error_summary_df,
            max_rows=20,
        )

        self.logger.info(
            "Startup model ready: loaded=%s train_rows=%s test_rows=%s mae=%s rmse=%s wape=%s",
            self.feedback_loop.forecaster.model is not None,
            train_rows,
            test_rows,
            metrics.get("mae"),
            metrics.get("rmse"),
            metrics.get("wape"),
        )
        print(
            f"\n{overview_block}\n\n{scenario_block}\n\n{prediction_block}\n\n{scenario_error_block}\n"
        )

        return {
            "loaded": self.feedback_loop.forecaster.model is not None,
            "training_rows": train_rows,
            "test_rows": test_rows,
            "feedback_rows": feedback_rows,
            "feature_count": feature_count,
            "metrics": metrics,
            "split_summary": split_summary,
            "prediction_summary_preview": (
                prediction_summary_df.head(25).replace({pd.NA: None}).to_dict("records")
                if prediction_summary_df is not None and not prediction_summary_df.empty
                else []
            ),
            "scenario_error_summary_preview": (
                scenario_error_summary_df.head(25).replace({pd.NA: None}).to_dict("records")
                if scenario_error_summary_df is not None and not scenario_error_summary_df.empty
                else []
            ),
        }

    def _dict_to_html_table(self, data: dict, title: str) -> str:
        if not data:
            return f"<section><h2>{escape(title)}</h2><p>No data</p></section>"

        rows = "".join(
            f"<tr><th>{escape(str(key))}</th><td>{escape(str(value))}</td></tr>"
            for key, value in data.items()
        )
        return (
            f"<section><h2>{escape(title)}</h2>"
            f"<table class='kv-table'><tbody>{rows}</tbody></table></section>"
        )

    def _records_to_html_table(
        self,
        records: list[dict],
        title: str,
        max_rows: int = 25,
    ) -> str:
        if not records:
            return f"<section><h2>{escape(title)}</h2><p>No rows</p></section>"

        df = pd.DataFrame(records).head(max_rows).copy()
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                df[col] = df[col].round(4)

        headers = "".join(f"<th>{escape(str(col))}</th>" for col in df.columns)
        body_rows = []
        for _, row in df.iterrows():
            cells = "".join(f"<td>{escape(str(value))}</td>" for value in row.tolist())
            body_rows.append(f"<tr>{cells}</tr>")

        return (
            f"<section><h2>{escape(title)}</h2>"
            f"<div class='table-wrap'><table><thead><tr>{headers}</tr></thead>"
            f"<tbody>{''.join(body_rows)}</tbody></table></div></section>"
        )

    def render_dashboard_html(self, startup_status: Optional[dict] = None) -> str:
        startup_status = startup_status or {}
        model_status = self.get_model_status()
        summary_dict = {
            "api_title": self.config.api.title,
            "api_version": self.config.api.version,
            "model_version": self.config.forecast.model_version,
            "data_dir": self.config.data.data_dir,
            "feedback_dir": self.config.data.feedback_dir,
            "loaded": startup_status.get("loaded"),
            "training_rows": startup_status.get("training_rows"),
            "test_rows": startup_status.get("test_rows"),
            "feature_count": startup_status.get("feature_count"),
            "mae": startup_status.get("metrics", {}).get("mae"),
            "rmse": startup_status.get("metrics", {}).get("rmse"),
            "wape": startup_status.get("metrics", {}).get("wape"),
            "mape": startup_status.get("metrics", {}).get("mape"),
            "bias": startup_status.get("metrics", {}).get("bias"),
        }
        scenario_rows = [
            {"scenario": key, "rows": value}
            for key, value in startup_status.get("split_summary", {}).get(
                "scenario_counts", {}
            ).items()
        ]

        return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{escape(self.config.api.title)}</title>
  <style>
    :root {{
      --bg: #f6f3ee;
      --panel: #fffdf8;
      --line: #d9cfbf;
      --text: #1f1d1a;
      --muted: #6c6256;
      --accent: #996c2b;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      padding: 32px;
      background: linear-gradient(180deg, #f5efe5 0%, var(--bg) 100%);
      color: var(--text);
      font: 14px/1.45 Georgia, "Times New Roman", serif;
    }}
    .container {{ max-width: 1400px; margin: 0 auto; }}
    h1 {{ margin: 0 0 8px; font-size: 30px; }}
    .subtitle {{ color: var(--muted); margin-bottom: 24px; }}
    section {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 14px;
      padding: 18px 20px;
      margin-bottom: 18px;
      box-shadow: 0 10px 30px rgba(74, 55, 24, 0.05);
    }}
    h2 {{
      margin: 0 0 14px;
      font-size: 18px;
      color: var(--accent);
    }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
      gap: 18px;
      align-items: start;
    }}
    .kv-table, table {{
      width: 100%;
      border-collapse: collapse;
    }}
    th, td {{
      border-bottom: 1px solid var(--line);
      padding: 8px 10px;
      text-align: left;
      vertical-align: top;
      white-space: nowrap;
    }}
    .kv-table th {{
      width: 180px;
      color: var(--muted);
      font-weight: 600;
    }}
    thead th {{
      position: sticky;
      top: 0;
      background: #f2ebde;
      z-index: 1;
    }}
    .table-wrap {{
      overflow: auto;
      max-height: 520px;
      border: 1px solid var(--line);
      border-radius: 10px;
    }}
    .note {{
      color: var(--muted);
      font-size: 13px;
      margin-top: 8px;
    }}
    code {{
      background: #efe6d6;
      padding: 2px 6px;
      border-radius: 6px;
      font-size: 12px;
    }}
  </style>
</head>
<body>
  <div class="container">
    <h1>{escape(self.config.api.title)}</h1>
    <div class="subtitle">Startup validation dashboard for the feedback-aware covers forecaster.</div>

    <div class="grid">
      {self._dict_to_html_table(summary_dict, "Startup Summary")}
      {self._dict_to_html_table(model_status, "Current Model Status")}
    </div>

    {self._records_to_html_table(startup_status.get("prediction_summary_preview", []), "Startup Prediction Preview", max_rows=25)}
    {self._records_to_html_table(startup_status.get("scenario_error_summary_preview", []), "Startup Scenario Error Summary", max_rows=25)}
    {self._records_to_html_table(scenario_rows, "Startup Scenario Coverage", max_rows=50)}
  </div>
</body>
</html>"""

    def get_model_status(self) -> dict:
        model_loaded = self.feedback_loop.forecaster.model is not None
        training_history_df = self.feedback_loop.training_history_df
        correction_table_df = self.feedback_loop.correction_table_df

        status = {
            "model_loaded": model_loaded,
            "feature_count": len(self.feedback_loop.feature_cols),
            "training_rows": int(len(training_history_df)) if training_history_df is not None else 0,
            "feedback_rows": int(len(self.feedback_loop.build_feedback_frame())),
            "actionable_scenarios": int(
                (correction_table_df["is_actionable"] == 1).sum()
            ) if correction_table_df is not None and not correction_table_df.empty else 0,
            "history_start": None,
            "history_end": None,
        }

        if training_history_df is not None and not training_history_df.empty:
            status["history_start"] = str(training_history_df["timestamp"].min())
            status["history_end"] = str(training_history_df["timestamp"].max())

        if self.feedback_loop.retraining_history_path.exists():
            retraining_history_df = pd.read_csv(
                self.feedback_loop.retraining_history_path,
                parse_dates=["trained_at"],
            )
            if not retraining_history_df.empty:
                latest_row = retraining_history_df.sort_values("trained_at").iloc[-1]
                status["last_trained_at"] = str(latest_row["trained_at"])
                status["last_train_rows"] = int(latest_row.get("training_rows", 0))
                status["last_feedback_rows"] = int(latest_row.get("feedback_rows", 0))
                status["last_holdout_rows"] = int(latest_row.get("holdout_rows", 0))
                status["last_split_strategy"] = latest_row.get("split_strategy")
                for metric_name in ["mae", "rmse", "wape"]:
                    metric_value = latest_row.get(metric_name)
                    status[f"last_{metric_name}"] = (
                        float(metric_value) if pd.notna(metric_value) else None
                    )
            else:
                status["last_trained_at"] = None
        else:
            status["last_trained_at"] = None

        return status

    def _normalize_context_df(self, context_df: pd.DataFrame) -> pd.DataFrame:
        if context_df.empty:
            raise ValueError("At least one hourly record is required.")
        if "timestamp" not in context_df.columns:
            raise ValueError("Each input row must include a timestamp.")

        normalized_df = context_df.copy()
        normalized_df["timestamp"] = pd.to_datetime(normalized_df["timestamp"])
        normalized_df = normalized_df.sort_values("timestamp").reset_index(drop=True)
        return normalized_df

    def forecast_covers(
        self,
        context_df: pd.DataFrame,
        auto_retrain: Optional[bool] = None,
        model_version: Optional[str] = None,
    ) -> pd.DataFrame:
        normalized_df = self._normalize_context_df(context_df)
        auto_retrain = (
            self.config.forecast.auto_retrain if auto_retrain is None else auto_retrain
        )
        model_version = model_version or self.config.forecast.model_version

        self.logger.info(
            "forecast_covers request received for %s hours, auto_retrain=%s",
            len(normalized_df),
            auto_retrain,
        )
        predictions_df = self.feedback_loop.forecast_with_feedback(
            normalized_df,
            model_version=model_version,
            auto_retrain=auto_retrain,
        )
        self.logger.info(
            "forecast_covers completed for %s hours using model_version=%s",
            len(predictions_df),
            model_version,
        )
        return predictions_df

    def plan_staff(
        self,
        context_df: pd.DataFrame,
        auto_retrain: Optional[bool] = None,
    ) -> dict[str, pd.DataFrame]:
        normalized_df = self._normalize_context_df(context_df)

        forecast_df = self.forecast_covers(
            normalized_df,
            auto_retrain=auto_retrain,
        )
        planner_input_df = forecast_df[["timestamp", "predicted_covers"]].copy()
        station_workload_df = self.staff_planner.estimate_station_workload(
            planner_input_df,
            external_features_df=normalized_df,
        )
        hourly_staff_plan_df = self.staff_planner.plan_hourly_staff(
            planner_input_df,
            external_features_df=normalized_df,
        )
        shift_schedule_df = self.staff_planner.build_shift_schedule(hourly_staff_plan_df)

        self.logger.info(
            "plan_staff completed for %s hours",
            len(normalized_df),
        )
        return {
            "covers_forecast": forecast_df,
            "station_workload": station_workload_df,
            "hourly_staff_plan": hourly_staff_plan_df,
            "shift_schedule": shift_schedule_df,
        }

    def plan_ingredients(
        self,
        context_df: pd.DataFrame,
        auto_retrain: Optional[bool] = None,
    ) -> dict[str, pd.DataFrame]:
        normalized_df = self._normalize_context_df(context_df)

        forecast_df = self.forecast_covers(
            normalized_df,
            auto_retrain=auto_retrain,
        )
        planner_input_df = forecast_df[["timestamp", "predicted_covers"]].copy()
        menu_demand_df, ingredient_hourly_df = (
            self.ingredient_planner.estimate_ingredient_demand(
                planner_input_df,
                external_features_df=normalized_df,
            )
        )
        _, daily_ingredient_df, purchase_recommendation_df = (
            self.ingredient_planner.build_purchase_recommendation(
                planner_input_df,
                external_features_df=normalized_df,
            )
        )

        self.logger.info(
            "plan_ingredients completed for %s hours",
            len(normalized_df),
        )
        return {
            "covers_forecast": forecast_df,
            "predicted_menu_demand": menu_demand_df,
            "ingredient_hourly_demand": ingredient_hourly_df,
            "daily_ingredient_demand": daily_ingredient_df,
            "purchase_recommendation": purchase_recommendation_df,
        }

    def plan_full(
        self,
        context_df: pd.DataFrame,
        auto_retrain: Optional[bool] = None,
    ) -> dict[str, pd.DataFrame]:
        normalized_df = self._normalize_context_df(context_df)

        forecast_df = self.forecast_covers(
            normalized_df,
            auto_retrain=auto_retrain,
        )
        staff_input_df = forecast_df[["timestamp", "predicted_covers"]].copy()
        ingredient_input_df = forecast_df[["timestamp", "predicted_covers"]].copy()

        station_workload_df = self.staff_planner.estimate_station_workload(
            staff_input_df,
            external_features_df=normalized_df,
        )
        hourly_staff_plan_df = self.staff_planner.plan_hourly_staff(
            staff_input_df,
            external_features_df=normalized_df,
        )
        shift_schedule_df = self.staff_planner.build_shift_schedule(hourly_staff_plan_df)

        menu_demand_df, ingredient_hourly_df = (
            self.ingredient_planner.estimate_ingredient_demand(
                ingredient_input_df,
                external_features_df=normalized_df,
            )
        )
        _, daily_ingredient_df, purchase_recommendation_df = (
            self.ingredient_planner.build_purchase_recommendation(
                ingredient_input_df,
                external_features_df=normalized_df,
            )
        )

        self.logger.info(
            "plan_full completed for %s hours",
            len(normalized_df),
        )
        return {
            "covers_forecast": forecast_df,
            "station_workload": station_workload_df,
            "hourly_staff_plan": hourly_staff_plan_df,
            "shift_schedule": shift_schedule_df,
            "predicted_menu_demand": menu_demand_df,
            "ingredient_hourly_demand": ingredient_hourly_df,
            "daily_ingredient_demand": daily_ingredient_df,
            "purchase_recommendation": purchase_recommendation_df,
        }

    def submit_feedback(
        self,
        feedback_entries_df: pd.DataFrame,
        retrain: bool = True,
    ) -> dict:
        normalized_df = self._normalize_context_df(feedback_entries_df)

        actual_columns = ["timestamp", "actual_covers", "actual_reservations", "actual_walk_ins"]
        actuals_df = normalized_df.copy()
        if "actual_covers" not in actuals_df.columns and "covers" in actuals_df.columns:
            actuals_df = actuals_df.rename(columns={"covers": "actual_covers"})
        for col in actual_columns:
            if col not in actuals_df.columns:
                actuals_df[col] = None
        actuals_df = actuals_df[actual_columns]

        manager_feedback_columns = [
            "timestamp",
            "predicted_covers",
            "holiday_flag",
            "event_flag",
            "promotion_flag",
            "rain_mm",
            "temp_c",
            "is_weekend",
            "season",
            "manager_note",
        ]
        manager_feedback_df = normalized_df.copy()
        for col in manager_feedback_columns:
            if col not in manager_feedback_df.columns:
                manager_feedback_df[col] = None
        manager_feedback_df = manager_feedback_df[manager_feedback_columns]

        if (
            manager_feedback_df["manager_note"].isna().all()
            and manager_feedback_df["predicted_covers"].isna().all()
            and manager_feedback_df["holiday_flag"].isna().all()
            and manager_feedback_df["event_flag"].isna().all()
            and manager_feedback_df["promotion_flag"].isna().all()
            and manager_feedback_df["rain_mm"].isna().all()
            and manager_feedback_df["temp_c"].isna().all()
            and manager_feedback_df["is_weekend"].isna().all()
            and manager_feedback_df["season"].isna().all()
        ):
            manager_feedback_df = None

        summary = self.feedback_loop.update_with_feedback(
            actuals_df=actuals_df,
            manager_feedback_df=manager_feedback_df,
            retrain=retrain,
        )
        scenario_corrections_df = self.feedback_loop.compute_scenario_corrections()
        feedback_frame_df = self.feedback_loop.build_feedback_frame()

        self.logger.info(
            "submit_feedback processed %s entries, retrain=%s, retrained=%s",
            len(normalized_df),
            retrain,
            summary.get("retrained"),
        )
        return {
            "summary": summary,
            "scenario_corrections": scenario_corrections_df,
            "feedback_frame": feedback_frame_df,
        }

    def get_scenario_corrections(self) -> pd.DataFrame:
        return self.feedback_loop.compute_scenario_corrections()

    def get_feedback_summary(self) -> dict:
        feedback_df = self.feedback_loop.build_feedback_frame()
        if feedback_df.empty:
            return {
                "feedback_rows": 0,
                "mean_abs_error": 0.0,
                "wape": 0.0,
            }

        wape = float(
            feedback_df["abs_error"].sum()
            / feedback_df["actual_covers"].clip(lower=1).sum()
        )
        return {
            "feedback_rows": int(len(feedback_df)),
            "mean_abs_error": float(feedback_df["abs_error"].mean()),
            "wape": wape,
        }
