from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src.utils import get_service_period


@dataclass
class IngredientPlannerConfig:
    data_dir: Path = Path("V2Data")
    menu_qty_buffer_multiplier: float = 1.05
    ingredient_qty_buffer_multiplier: float = 1.03
    reorder_review_days: int = 1


class IngredientPlanner:
    """
    Convert forecasted covers into menu demand and then into ingredient ordering.

    The planner uses:
    - historical item sales to learn contextual menu mix
    - recipe mappings to convert menu demand into ingredient demand
    - inventory and supplier constraints to recommend order quantities
    """

    def __init__(self, config: Optional[IngredientPlannerConfig] = None):
        self.config = config or IngredientPlannerConfig()
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
        self.menu_ingredients_df = pd.read_csv(self.data_dir / "menu_ingredients.csv")
        self.ingredient_master_df = pd.read_csv(self.data_dir / "ingredient_master.csv")

        self._validate_core_schema()
        self.item_mix_exact_df: Optional[pd.DataFrame] = None
        self.item_mix_period_weekend_df: Optional[pd.DataFrame] = None
        self.item_mix_period_df: Optional[pd.DataFrame] = None
        self.item_mix_global_df: Optional[pd.DataFrame] = None

    def _validate_core_schema(self) -> None:
        missing_recipe_cols = {
            "menu_item_id",
            "ingredient_id",
            "qty_per_dish",
            "yield_loss_pct",
        } - set(self.menu_ingredients_df.columns)
        if missing_recipe_cols:
            raise ValueError(
                f"menu_ingredients.csv is missing columns: {sorted(missing_recipe_cols)}"
            )

        missing_inventory_cols = {
            "ingredient_id",
            "shelf_life_days",
            "lead_time_days",
            "min_order_qty",
            "current_stock",
            "safety_stock",
        } - set(self.ingredient_master_df.columns)
        if missing_inventory_cols:
            raise ValueError(
                f"ingredient_master.csv is missing columns: {sorted(missing_inventory_cols)}"
            )

    def _prepare_context(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df["hour"] = df["timestamp"].dt.hour
        df["day_of_week"] = df["timestamp"].dt.dayofweek
        df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
        df["service_period"] = df["hour"].apply(get_service_period)

        for col in ["holiday_flag", "event_flag"]:
            if col not in df.columns:
                df[col] = 0

        if "rain_mm" in df.columns:
            df["heavy_rain_flag"] = (df["rain_mm"] >= 10).astype(int)
        else:
            df["rain_mm"] = 0.0
            df["heavy_rain_flag"] = 0

        return df

    def _is_item_allowed(self, item_service_period: str, forecast_service_period: str) -> bool:
        if item_service_period == "all_day":
            return True
        if item_service_period == "breakfast":
            return forecast_service_period == "breakfast"
        if item_service_period == "evening":
            return forecast_service_period == "afternoon"
        if item_service_period == "lunch_dinner":
            return forecast_service_period in {"lunch", "dinner"}
        return True

    def fit_menu_mix_model(self) -> pd.DataFrame:
        hourly_context_df = self.historical_sales_df[["timestamp", "covers"]].merge(
            self.external_features_df, on="timestamp", how="left"
        )
        hourly_context_df = self._prepare_context(hourly_context_df)
        hourly_context_df = hourly_context_df[hourly_context_df["covers"] > 0].copy()

        menu_sales_context_df = self.historical_menu_sales_df.merge(
            hourly_context_df[
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
        menu_sales_context_df = menu_sales_context_df.merge(
            self.menu_item_master_df[
                [
                    "menu_item_id",
                    "menu_item_name",
                    "category",
                    "sub_category",
                    "service_period",
                    "is_active",
                ]
            ].rename(columns={"service_period": "item_service_period"}),
            on="menu_item_id",
            how="left",
        )
        if "is_active" in menu_sales_context_df.columns:
            menu_sales_context_df = menu_sales_context_df[
                menu_sales_context_df["is_active"] == 1
            ].copy()

        menu_sales_context_df["qty_per_cover"] = (
            menu_sales_context_df["qty_sold"] / menu_sales_context_df["covers"]
        )

        self.item_mix_exact_df = (
            menu_sales_context_df.groupby(
                [
                    "menu_item_id",
                    "service_period",
                    "is_weekend",
                    "holiday_flag",
                    "event_flag",
                    "heavy_rain_flag",
                ],
                as_index=False,
            )["qty_per_cover"]
            .median()
            .rename(columns={"qty_per_cover": "qty_per_cover_exact"})
        )

        self.item_mix_period_weekend_df = (
            menu_sales_context_df.groupby(
                ["menu_item_id", "service_period", "is_weekend"], as_index=False
            )["qty_per_cover"]
            .median()
            .rename(columns={"qty_per_cover": "qty_per_cover_period_weekend"})
        )

        self.item_mix_period_df = (
            menu_sales_context_df.groupby(
                ["menu_item_id", "service_period"], as_index=False
            )["qty_per_cover"]
            .median()
            .rename(columns={"qty_per_cover": "qty_per_cover_period"})
        )

        self.item_mix_global_df = (
            menu_sales_context_df.groupby("menu_item_id", as_index=False)["qty_per_cover"]
            .median()
            .rename(columns={"qty_per_cover": "qty_per_cover_global"})
        )

        return menu_sales_context_df

    def estimate_menu_item_demand(
        self,
        covers_forecast_df: pd.DataFrame,
        external_features_df: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        if self.item_mix_exact_df is None:
            self.fit_menu_mix_model()

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

        active_menu_df = self.menu_item_master_df.copy()
        if "is_active" in active_menu_df.columns:
            active_menu_df = active_menu_df[active_menu_df["is_active"] == 1].copy()

        item_forecast_df = forecast_df.merge(
            active_menu_df[
                [
                    "menu_item_id",
                    "menu_item_name",
                    "category",
                    "sub_category",
                    "service_period",
                    "station",
                ]
            ].rename(columns={"service_period": "item_service_period"}),
            how="cross",
        )
        item_forecast_df = item_forecast_df[
            item_forecast_df.apply(
                lambda row: self._is_item_allowed(
                    row["item_service_period"], row["service_period"]
                ),
                axis=1,
            )
        ].copy()

        item_forecast_df = item_forecast_df.merge(
            self.item_mix_exact_df,
            on=[
                "menu_item_id",
                "service_period",
                "is_weekend",
                "holiday_flag",
                "event_flag",
                "heavy_rain_flag",
            ],
            how="left",
        )
        item_forecast_df = item_forecast_df.merge(
            self.item_mix_period_weekend_df,
            on=["menu_item_id", "service_period", "is_weekend"],
            how="left",
        )
        item_forecast_df = item_forecast_df.merge(
            self.item_mix_period_df,
            on=["menu_item_id", "service_period"],
            how="left",
        )
        item_forecast_df = item_forecast_df.merge(
            self.item_mix_global_df,
            on="menu_item_id",
            how="left",
        )

        item_forecast_df["qty_per_cover"] = (
            item_forecast_df["qty_per_cover_exact"]
            .fillna(item_forecast_df["qty_per_cover_period_weekend"])
            .fillna(item_forecast_df["qty_per_cover_period"])
            .fillna(item_forecast_df["qty_per_cover_global"])
            .fillna(0.0)
        )

        item_forecast_df["predicted_menu_qty"] = (
            item_forecast_df["predicted_covers"]
            * item_forecast_df["qty_per_cover"]
            * self.config.menu_qty_buffer_multiplier
        )
        item_forecast_df["predicted_menu_qty"] = item_forecast_df["predicted_menu_qty"].clip(
            lower=0
        )

        return item_forecast_df[
            [
                "timestamp",
                "menu_item_id",
                "menu_item_name",
                "category",
                "sub_category",
                "station",
                "predicted_covers",
                "service_period",
                "is_weekend",
                "holiday_flag",
                "event_flag",
                "heavy_rain_flag",
                "qty_per_cover",
                "predicted_menu_qty",
            ]
        ].sort_values(["timestamp", "menu_item_id"])

    def estimate_ingredient_demand(
        self,
        covers_forecast_df: pd.DataFrame,
        external_features_df: Optional[pd.DataFrame] = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        menu_demand_df = self.estimate_menu_item_demand(
            covers_forecast_df=covers_forecast_df,
            external_features_df=external_features_df,
        )

        ingredient_demand_df = menu_demand_df.merge(
            self.menu_ingredients_df,
            on="menu_item_id",
            how="left",
        )
        ingredient_demand_df["gross_required_qty"] = (
            ingredient_demand_df["predicted_menu_qty"]
            * ingredient_demand_df["qty_per_dish"]
            * (1 + ingredient_demand_df["yield_loss_pct"].fillna(0))
            * self.config.ingredient_qty_buffer_multiplier
        )

        ingredient_hourly_df = (
            ingredient_demand_df.groupby(
                ["timestamp", "ingredient_id", "ingredient_name", "uom"], as_index=False
            )["gross_required_qty"]
            .sum()
            .rename(columns={"gross_required_qty": "required_qty"})
            .sort_values(["timestamp", "ingredient_id"])
        )

        return menu_demand_df, ingredient_hourly_df

    def build_purchase_recommendation(
        self,
        covers_forecast_df: pd.DataFrame,
        external_features_df: Optional[pd.DataFrame] = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        menu_demand_df, ingredient_hourly_df = self.estimate_ingredient_demand(
            covers_forecast_df=covers_forecast_df,
            external_features_df=external_features_df,
        )

        forecast_start = pd.to_datetime(covers_forecast_df["timestamp"]).min().normalize()
        ingredient_hourly_df["forecast_date"] = ingredient_hourly_df["timestamp"].dt.normalize()

        daily_ingredient_df = (
            ingredient_hourly_df.groupby(
                ["forecast_date", "ingredient_id", "ingredient_name", "uom"], as_index=False
            )["required_qty"]
            .sum()
            .sort_values(["ingredient_id", "forecast_date"])
        )

        recommendation_rows = []
        ingredient_master_df = self.ingredient_master_df.copy()

        for _, ingredient_row in ingredient_master_df.iterrows():
            ingredient_id = ingredient_row["ingredient_id"]
            ingredient_daily_df = daily_ingredient_df[
                daily_ingredient_df["ingredient_id"] == ingredient_id
            ].copy()

            lead_time_days = int(ingredient_row["lead_time_days"])
            shelf_life_days = int(ingredient_row["shelf_life_days"])
            safety_stock = float(ingredient_row["safety_stock"])
            current_stock = float(ingredient_row["current_stock"])
            min_order_qty = float(ingredient_row["min_order_qty"])

            lead_time_cutoff = forecast_start + pd.Timedelta(
                days=max(lead_time_days, 0) + self.config.reorder_review_days
            )
            shelf_life_cutoff = forecast_start + pd.Timedelta(days=max(shelf_life_days, 0))

            demand_during_lead_time = float(
                ingredient_daily_df.loc[
                    ingredient_daily_df["forecast_date"] < lead_time_cutoff, "required_qty"
                ].sum()
            )
            demand_before_expiry = float(
                ingredient_daily_df.loc[
                    ingredient_daily_df["forecast_date"] < shelf_life_cutoff, "required_qty"
                ].sum()
            )

            target_stock = min(
                demand_before_expiry + safety_stock,
                max(demand_during_lead_time + safety_stock, safety_stock),
            )
            raw_order_qty = max(target_stock - current_stock, 0.0)

            if min_order_qty > 0 and raw_order_qty > 0:
                recommended_order_qty = (
                    np.ceil(raw_order_qty / min_order_qty) * min_order_qty
                )
            else:
                recommended_order_qty = raw_order_qty

            projected_stock_after_lead_time = (
                current_stock - demand_during_lead_time + recommended_order_qty
            )
            projected_stock_before_expiry = (
                current_stock - demand_before_expiry + recommended_order_qty
            )

            recommendation_rows.append(
                {
                    "ingredient_id": ingredient_id,
                    "ingredient_name": ingredient_row["ingredient_name"],
                    "uom": ingredient_row["uom"],
                    "category": ingredient_row["category"],
                    "lead_time_days": lead_time_days,
                    "shelf_life_days": shelf_life_days,
                    "min_order_qty": min_order_qty,
                    "current_stock": current_stock,
                    "safety_stock": safety_stock,
                    "demand_during_lead_time": round(demand_during_lead_time, 3),
                    "demand_before_expiry": round(demand_before_expiry, 3),
                    "raw_order_qty": round(raw_order_qty, 3),
                    "recommended_order_qty": round(float(recommended_order_qty), 3),
                    "projected_stock_after_lead_time": round(
                        float(projected_stock_after_lead_time), 3
                    ),
                    "projected_stock_before_expiry": round(
                        float(projected_stock_before_expiry), 3
                    ),
                    "stockout_risk_before_replenishment": int(
                        current_stock < demand_during_lead_time
                    ),
                    "expiry_risk_after_order": int(
                        recommended_order_qty > 0
                        and (current_stock + recommended_order_qty)
                        > (demand_before_expiry + safety_stock)
                    ),
                }
            )

        purchase_recommendation_df = pd.DataFrame(recommendation_rows).sort_values(
            ["recommended_order_qty", "ingredient_id"], ascending=[False, True]
        )

        return menu_demand_df, daily_ingredient_df, purchase_recommendation_df


def build_ingredient_plan_from_forecast(
    covers_forecast_df: pd.DataFrame,
    data_dir: str | Path = "V2Data",
    external_features_df: Optional[pd.DataFrame] = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    
    planner = IngredientPlanner(IngredientPlannerConfig(data_dir=Path(data_dir)))
    
    return planner.build_purchase_recommendation(
        covers_forecast_df=covers_forecast_df,
        external_features_df=external_features_df,
    )
