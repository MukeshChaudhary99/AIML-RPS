from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src.utils import get_service_period


@dataclass
class PlannerConfig:
    data_dir: Path = Path("V2Data")
    prep_buffer_multiplier: float = 1.10
    covers_buffer_multiplier: float = 1.05


class StaffPlanner:
    """
    Convert forecasted covers into staffing requirements by role and station.

    The planner uses:
    - cover-based staffing rules for dining/floor roles such as server and manager
    - station workload derived from historical menu sales for production roles
    - a simple greedy shift builder to convert hourly demand into shift blocks
    """

    def __init__(self, config: Optional[PlannerConfig] = None):
        self.config = config or PlannerConfig()
        self.data_dir = Path(self.config.data_dir)

        self.historical_sales_df = pd.read_csv(
            self.data_dir / "historical_sales.csv", parse_dates=["timestamp"]
        )
        self.external_features_df = pd.read_csv(
            self.data_dir / "external_features.csv", parse_dates=["timestamp"]
        )
        self.historical_menu_sales_df = pd.read_csv(
            self.data_dir / "historical_menu_sales.csv", parse_dates=["timestamp"]
        )
        self.menu_item_master_df = pd.read_csv(
            self.data_dir / "menu_items_master.csv"
        )
        self.staff_roles_df = pd.read_csv(self.data_dir / "staff_roles.csv")

        self._validate_core_schema()
        self.station_ratio_exact_df: Optional[pd.DataFrame] = None
        self.station_ratio_period_weekend_df: Optional[pd.DataFrame] = None
        self.station_ratio_period_df: Optional[pd.DataFrame] = None
        self.station_ratio_global_df: Optional[pd.DataFrame] = None

    def _validate_core_schema(self) -> None:
        missing_role_cols = {
            "role",
            "station",
            "capacity_unit",
            "capacity_per_hour",
            "min_staff_per_shift",
            "max_staff_per_shift",
            "shift_length_hours",
            "hourly_cost",
        } - set(self.staff_roles_df.columns)
        if missing_role_cols:
            raise ValueError(f"staff_roles.csv is missing columns: {sorted(missing_role_cols)}")

        if "covers" not in self.historical_sales_df.columns:
            raise ValueError("historical_sales.csv must contain a covers column")

        if "menu_item_id" not in self.historical_menu_sales_df.columns:
            raise ValueError("historical_menu_sales.csv must contain menu_item_id")

    def _prepare_context(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df["hour"] = df["timestamp"].dt.hour
        df["day_of_week"] = df["timestamp"].dt.dayofweek
        df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
        df["service_period"] = df["hour"].apply(get_service_period)

        if "rain_mm" in df.columns:
            df["heavy_rain_flag"] = (df["rain_mm"] >= 10).astype(int)
        else:
            df["rain_mm"] = 0.0
            df["heavy_rain_flag"] = 0

        for col in ["holiday_flag", "event_flag"]:
            if col not in df.columns:
                df[col] = 0

        return df

    def fit_station_workload_model(self) -> pd.DataFrame:

        menu_sales_df = self.historical_menu_sales_df.copy()
        menu_sales_df["timestamp"] = pd.to_datetime(menu_sales_df["timestamp"])

        active_menu_df = self.menu_item_master_df.copy()
        if "is_active" in active_menu_df.columns:
            active_menu_df = active_menu_df[active_menu_df["is_active"] == 1].copy()

        menu_station_df = menu_sales_df.merge(
            active_menu_df[["menu_item_id", "station", "prep_minutes"]],
            on="menu_item_id",
            how="left",
        )
        menu_station_df["prep_minutes"] = menu_station_df["prep_minutes"].fillna(0)
        menu_station_df["station_workload_minutes"] = (
            menu_station_df["qty_sold"] * menu_station_df["prep_minutes"]
        )

        station_workload_df = (
            menu_station_df.groupby(["timestamp", "station"], as_index=False)[
                "station_workload_minutes"
            ]
            .sum()
            .sort_values(["timestamp", "station"])
        )

        context_df = self.historical_sales_df[["timestamp", "covers"]].merge(
            self.external_features_df, on="timestamp", how="left"
        )
        context_df = self._prepare_context(context_df)

        workload_history_df = station_workload_df.merge(
            context_df[
                [
                    "timestamp",
                    "covers",
                    "service_period",
                    "is_weekend",
                    "holiday_flag",
                    "event_flag",
                    "heavy_rain_flag",
                ]
            ],
            on="timestamp",
            how="left",
        )

        workload_history_df = workload_history_df[workload_history_df["covers"] > 0].copy()
        workload_history_df["minutes_per_cover"] = (
            workload_history_df["station_workload_minutes"] / workload_history_df["covers"]
        )

        self.station_ratio_exact_df = (
            workload_history_df.groupby(
                [
                    "station",
                    "service_period",
                    "is_weekend",
                    "holiday_flag",
                    "event_flag",
                    "heavy_rain_flag",
                ],
                as_index=False,
            )["minutes_per_cover"]
            .median()
            .rename(columns={"minutes_per_cover": "minutes_per_cover_exact"})
        )

        self.station_ratio_period_weekend_df = (
            workload_history_df.groupby(
                ["station", "service_period", "is_weekend"], as_index=False
            )["minutes_per_cover"]
            .median()
            .rename(columns={"minutes_per_cover": "minutes_per_cover_period_weekend"})
        )

        self.station_ratio_period_df = (
            workload_history_df.groupby(["station", "service_period"], as_index=False)[
                "minutes_per_cover"
            ]
            .median()
            .rename(columns={"minutes_per_cover": "minutes_per_cover_period"})
        )

        self.station_ratio_global_df = (
            workload_history_df.groupby("station", as_index=False)["minutes_per_cover"]
            .median()
            .rename(columns={"minutes_per_cover": "minutes_per_cover_global"})
        )

        return workload_history_df

    def estimate_station_workload(
        self,
        covers_forecast_df: pd.DataFrame,
        external_features_df: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        
        if self.station_ratio_exact_df is None:
            self.fit_station_workload_model()

        forecast_df = covers_forecast_df.copy()
        forecast_df["timestamp"] = pd.to_datetime(forecast_df["timestamp"])

        if "predicted_covers" not in forecast_df.columns:
            raise ValueError("covers_forecast_df must contain a predicted_covers column")

        if external_features_df is None:
            external_features_df = self.external_features_df.copy()
        else:
            external_features_df = external_features_df.copy()

        external_features_df["timestamp"] = pd.to_datetime(external_features_df["timestamp"])

        forecast_df = forecast_df.merge(external_features_df, on="timestamp", how="left")
        forecast_df = self._prepare_context(forecast_df)

        stations_df = self.station_ratio_global_df[["station"]].copy()
        station_forecast_df = forecast_df.merge(stations_df, how="cross")

        station_forecast_df = station_forecast_df.merge(
            self.station_ratio_exact_df,
            on=[
                "station",
                "service_period",
                "is_weekend",
                "holiday_flag",
                "event_flag",
                "heavy_rain_flag",
            ],
            how="left",
        )
        station_forecast_df = station_forecast_df.merge(
            self.station_ratio_period_weekend_df,
            on=["station", "service_period", "is_weekend"],
            how="left",
        )
        station_forecast_df = station_forecast_df.merge(
            self.station_ratio_period_df,
            on=["station", "service_period"],
            how="left",
        )
        station_forecast_df = station_forecast_df.merge(
            self.station_ratio_global_df,
            on="station",
            how="left",
        )

        station_forecast_df["minutes_per_cover"] = (
            station_forecast_df["minutes_per_cover_exact"]
            .fillna(station_forecast_df["minutes_per_cover_period_weekend"])
            .fillna(station_forecast_df["minutes_per_cover_period"])
            .fillna(station_forecast_df["minutes_per_cover_global"])
            .fillna(0.0)
        )

        station_forecast_df["predicted_station_workload_minutes"] = (
            station_forecast_df["predicted_covers"]
            * station_forecast_df["minutes_per_cover"]
            * self.config.prep_buffer_multiplier
        )

        return station_forecast_df[
            [
                "timestamp",
                "station",
                "predicted_covers",
                "service_period",
                "is_weekend",
                "holiday_flag",
                "event_flag",
                "heavy_rain_flag",
                "minutes_per_cover",
                "predicted_station_workload_minutes",
            ]
        ].sort_values(["timestamp", "station"])

    def plan_hourly_staff(
        self,
        covers_forecast_df: pd.DataFrame,
        external_features_df: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        
        station_workload_df = self.estimate_station_workload(
            covers_forecast_df=covers_forecast_df,
            external_features_df=external_features_df,
        )

        cover_roles_df = self.staff_roles_df[
            self.staff_roles_df["capacity_unit"] == "covers"
        ].copy()
        prep_roles_df = self.staff_roles_df[
            self.staff_roles_df["capacity_unit"] == "prep_minutes"
        ].copy()

        cover_plan_frames = []
        if not cover_roles_df.empty:
            base_forecast_df = covers_forecast_df.copy()
            base_forecast_df["timestamp"] = pd.to_datetime(base_forecast_df["timestamp"])
            for _, role_row in cover_roles_df.iterrows():
                role_df = base_forecast_df.copy()
                role_df["role"] = role_row["role"]
                role_df["station"] = role_row["station"]
                role_df["capacity_unit"] = role_row["capacity_unit"]
                role_df["capacity_per_hour"] = role_row["capacity_per_hour"]
                role_df["min_staff_per_shift"] = role_row["min_staff_per_shift"]
                role_df["max_staff_per_shift"] = role_row["max_staff_per_shift"]
                role_df["shift_length_hours"] = role_row["shift_length_hours"]
                role_df["hourly_cost"] = role_row["hourly_cost"]
                role_df["required_load"] = (
                    role_df["predicted_covers"] * self.config.covers_buffer_multiplier
                )
                cover_plan_frames.append(role_df)

        cover_plan_df = (
            pd.concat(cover_plan_frames, ignore_index=True)
            if cover_plan_frames
            else pd.DataFrame()
        )

        prep_plan_df = station_workload_df.merge(
            prep_roles_df,
            on="station",
            how="inner",
        )
        if not prep_plan_df.empty:
            prep_plan_df["required_load"] = prep_plan_df["predicted_station_workload_minutes"]

        hourly_plan_df = pd.concat(
            [df for df in [cover_plan_df, prep_plan_df] if not df.empty],
            ignore_index=True,
        )

        hourly_plan_df["required_staff"] = np.ceil(
            hourly_plan_df["required_load"] / hourly_plan_df["capacity_per_hour"]
        ).astype(int)

        hourly_plan_df["required_staff"] = hourly_plan_df["required_staff"].clip(
            lower=hourly_plan_df["min_staff_per_shift"],
            upper=hourly_plan_df["max_staff_per_shift"],
        )
        
        hourly_plan_df["estimated_hourly_cost"] = (
            hourly_plan_df["required_staff"] * hourly_plan_df["hourly_cost"]
        ).round(2)

        output_cols = [
            "timestamp",
            "role",
            "station",
            "capacity_unit",
            "capacity_per_hour",
            "required_load",
            "required_staff",
            "shift_length_hours",
            "hourly_cost",
            "estimated_hourly_cost",
        ]
        return hourly_plan_df[output_cols].sort_values(["timestamp", "role", "station"])

    def build_shift_schedule(self, hourly_plan_df: pd.DataFrame) -> pd.DataFrame:
        if hourly_plan_df.empty:
            return pd.DataFrame(
                columns=[
                    "date",
                    "role",
                    "station",
                    "shift_start",
                    "shift_end",
                    "staff_count",
                    "shift_length_hours",
                    "hourly_cost",
                    "estimated_shift_cost",
                ]
            )

        hourly_plan_df = hourly_plan_df.copy()
        hourly_plan_df["timestamp"] = pd.to_datetime(hourly_plan_df["timestamp"])
        hourly_plan_df["date"] = hourly_plan_df["timestamp"].dt.date

        shift_rows = []

        for (date_value, role, station), group_df in hourly_plan_df.groupby(
            ["date", "role", "station"], sort=True
        ):
            group_df = group_df.sort_values("timestamp").reset_index(drop=True)
            shift_length_hours = int(group_df["shift_length_hours"].iloc[0])
            hourly_cost = float(group_df["hourly_cost"].iloc[0])
            last_timestamp = group_df["timestamp"].max()
            active_shift_end_times: list[pd.Timestamp] = []

            for _, row in group_df.iterrows():
                current_ts = row["timestamp"]
                active_shift_end_times = [
                    end_ts for end_ts in active_shift_end_times if end_ts > current_ts
                ]
                active_count = len(active_shift_end_times)
                required_count = int(row["required_staff"])

                if active_count >= required_count:
                    continue

                new_shifts_needed = required_count - active_count
                for _ in range(new_shifts_needed):
                    natural_end = current_ts + pd.Timedelta(hours=shift_length_hours)
                    clipped_end = min(natural_end, last_timestamp + pd.Timedelta(hours=1))
                    active_shift_end_times.append(clipped_end)
                    shift_rows.append(
                        {
                            "date": date_value,
                            "role": role,
                            "station": station,
                            "shift_start": current_ts,
                            "shift_end": clipped_end,
                            "staff_count": 1,
                            "shift_length_hours": (clipped_end - current_ts).total_seconds()
                            / 3600,
                            "hourly_cost": hourly_cost,
                            "estimated_shift_cost": round(
                                ((clipped_end - current_ts).total_seconds() / 3600)
                                * hourly_cost,
                                2,
                            ),
                        }
                    )

        shift_schedule_df = pd.DataFrame(shift_rows)
        if shift_schedule_df.empty:
            return shift_schedule_df

        return (
            shift_schedule_df.groupby(
                [
                    "date",
                    "role",
                    "station",
                    "shift_start",
                    "shift_end",
                    "shift_length_hours",
                    "hourly_cost",
                ],
                as_index=False,
            )["staff_count"]
            .sum()
            .assign(
                estimated_shift_cost=lambda df: (
                    df["staff_count"] * df["shift_length_hours"] * df["hourly_cost"]
                ).round(2)
            )
            .sort_values(["date", "shift_start", "role", "station"])
        )


def build_staffing_plan_from_forecast(
    covers_forecast_df: pd.DataFrame,
    data_dir: str | Path = "V2Data",
    external_features_df: Optional[pd.DataFrame] = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    
    planner = StaffPlanner(PlannerConfig(data_dir=Path(data_dir)))
    
    hourly_plan_df = planner.plan_hourly_staff(
        covers_forecast_df=covers_forecast_df,
        external_features_df=external_features_df,
    )
    
    shift_schedule_df = planner.build_shift_schedule(hourly_plan_df)
    
    return hourly_plan_df, shift_schedule_df
