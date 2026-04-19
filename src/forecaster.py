import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

from src.utils import clip_round_positive, get_logger, get_service_period


class RestaurantForecaster:
    """
    Single-model restaurant forecaster using XGBoost.
    Includes:
    - feature engineering
    - point forecast
    - scenario testing support
    """

    def __init__(
        self,
        feature_cols=None,
        model_params=None,
        early_stopping_rounds=60,
        logger=None,
    ):
        self.feature_cols = feature_cols or []
        self.model = None
        self.model_params = model_params or {}
        self.early_stopping_rounds = early_stopping_rounds
        self.logger = logger or get_logger(__name__)

    def build_enhanced_features(self, df, covers_df=None, train_end_ts=None):
        df = df.copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values("timestamp").reset_index(drop=True)
        self.logger.info(
            "Building enhanced features for %s row(s); future_mode=%s.",
            len(df),
            covers_df is not None and train_end_ts is not None,
        )

        # ----------------------------
        # TIME FEATURES
        # ----------------------------
        df["hour"] = df["timestamp"].dt.hour
        df["day_of_week"] = df["timestamp"].dt.dayofweek
        df["day_of_month"] = df["timestamp"].dt.day
        df["month"] = df["timestamp"].dt.month
        df["week_of_year"] = df["timestamp"].dt.isocalendar().week.astype(int)

        if "is_weekend" not in df.columns:
            df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

        df["is_friday"] = (df["day_of_week"] == 4).astype(int)
        df["is_month_start"] = df["timestamp"].dt.is_month_start.astype(int)
        df["is_month_end"] = df["timestamp"].dt.is_month_end.astype(int)

        # Peaks
        df["is_breakfast_peak"] = df["hour"].between(9, 11).astype(int)
        df["is_lunch_peak"] = df["hour"].between(12, 14).astype(int)
        df["is_dinner_peak"] = df["hour"].between(19, 21).astype(int)
        df["is_breakfast_period"] = df["hour"].between(8, 11).astype(int)
        df["is_afternoon_period"] = df["hour"].between(16, 18).astype(int)
        df["is_any_peak"] = (
            df["is_breakfast_peak"] | df["is_lunch_peak"] | df["is_dinner_peak"]
        ).astype(int)

        df["hours_to_lunch"] = (13 - df["hour"]).clip(-12, 12)
        df["hours_to_dinner"] = (20 - df["hour"]).clip(-12, 12)

        # Cyclic
        df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
        df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
        df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
        df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)
        df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
        df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

        # ----------------------------
        # WEATHER FEATURES
        # ----------------------------
        if "rain_mm" in df.columns:
            df["rain_flag"] = (df["rain_mm"] > 0).astype(int)
            df["moderate_rain_flag"] = df["rain_mm"].between(5, 15).astype(int)
            df["heavy_rain_flag"] = (df["rain_mm"] >= 15).astype(int)
            df["rain_category"] = pd.cut(
                df["rain_mm"],
                bins=[-np.inf, 0, 5, 15, np.inf],
                labels=["none", "light", "moderate", "heavy"],
            )

            df["is_weekend_dinner"] = df["is_weekend"] * df["is_dinner_peak"]

            df["rain_evening_effect"] = df["heavy_rain_flag"] * df["hour"].between(
                16, 19
            ).astype(int)

            df["rain_dinner_penalty"] = df["heavy_rain_flag"] * df["hour"].between(
                20, 22
            ).astype(int)

        if "temp_c" in df.columns:
            df["temp_comfortable"] = df["temp_c"].between(22, 30).astype(int)
            df["temp_hot"] = (df["temp_c"] > 35).astype(int)
            df["temp_very_hot"] = (df["temp_c"] > 38).astype(int)
            df["temp_cold"] = (df["temp_c"] < 18).astype(int)

        if "season" in df.columns:
            df["is_festive_season"] = (df["season"] == "festive").astype(int)
            df["is_monsoon_season"] = (df["season"] == "monsoon").astype(int)

        # ----------------------------
        # SERVICE PERIOD
        # ----------------------------
        df["service_period"] = df["hour"].apply(get_service_period)

        # ----------------------------
        # INTERACTIONS
        # ----------------------------
        if "holiday_flag" in df.columns:
            df["holiday_x_hour"] = df["holiday_flag"] * df["hour"]
            df["holiday_x_lunch"] = df["holiday_flag"] * df["is_lunch_peak"]
            df["holiday_x_dinner"] = df["holiday_flag"] * df["is_dinner_peak"]

        if "event_flag" in df.columns:
            df["event_x_hour"] = df["event_flag"] * df["hour"]
            df["event_x_dinner"] = df["event_flag"] * df["is_dinner_peak"]

        if "promotion_flag" in df.columns:
            df["promo_x_hour"] = df["promotion_flag"] * df["hour"]
            df["promo_x_lunch"] = df["promotion_flag"] * df["is_lunch_peak"]
            df["promo_x_breakfast"] = (
                df["promotion_flag"] * df["is_breakfast_period"]
            )
            df["promo_x_festive_breakfast"] = (
                df["promotion_flag"]
                * df["is_breakfast_period"]
                * df.get("is_festive_season", 0)
            )

        if "rain_mm" in df.columns:
            df["rain_x_hour"] = df["rain_mm"] * df["hour"]
            df["rain_x_lunch"] = df["rain_mm"] * df["is_lunch_peak"]
            df["rain_x_dinner"] = df["rain_mm"] * df["is_dinner_peak"]
            df["rain_x_breakfast"] = df["rain_mm"] * df["is_breakfast_period"]
            df["rain_x_afternoon"] = df["rain_mm"] * df["is_afternoon_period"]
            df["heavy_rain_x_breakfast"] = (
                df["heavy_rain_flag"] * df["is_breakfast_period"]
            )
            df["heavy_rain_x_afternoon"] = (
                df["heavy_rain_flag"] * df["is_afternoon_period"]
            )
            df["rain_x_weekend_breakfast"] = (
                df["rain_mm"] * df["is_weekend"] * df["is_breakfast_period"]
            )
            df["rain_x_festive_afternoon"] = (
                df["rain_mm"] * df.get("is_festive_season", 0) * df["is_afternoon_period"]
            )

        if "temp_c" in df.columns:
            df["temp_x_hour"] = df["temp_c"] * df["hour"]
            df["hot_x_lunch"] = df["temp_hot"] * df["is_lunch_peak"]
            df["hot_x_dinner"] = df["temp_hot"] * df["is_dinner_peak"]
            df["hot_x_breakfast"] = df["temp_hot"] * df["is_breakfast_period"]
            df["cold_x_breakfast"] = df["temp_cold"] * df["is_breakfast_period"]

        df["weekend_x_breakfast"] = df["is_weekend"] * df["is_breakfast_period"]
        df["weekend_x_afternoon"] = df["is_weekend"] * df["is_afternoon_period"]

        if "season" in df.columns:
            df["festive_x_breakfast"] = (
                df["is_festive_season"] * df["is_breakfast_period"]
            )
            df["festive_x_afternoon"] = (
                df["is_festive_season"] * df["is_afternoon_period"]
            )
            df["festive_x_weekend"] = df["is_festive_season"] * df["is_weekend"]
            df["monsoon_x_breakfast"] = (
                df["is_monsoon_season"] * df["is_breakfast_period"]
            )
            df["monsoon_x_afternoon"] = (
                df["is_monsoon_season"] * df["is_afternoon_period"]
            )
            df["winter_x_breakfast"] = (
                (df["season"] == "winter").astype(int) * df["is_breakfast_period"]
            )

        if "reservations" in df.columns:
            df["reservations"] = df["reservations"].fillna(0)
            df["reservations_x_breakfast"] = (
                df["reservations"] * df["is_breakfast_period"]
            )
            df["reservations_x_lunch"] = df["reservations"] * df["is_lunch_peak"]
            df["reservations_x_dinner"] = df["reservations"] * df["is_dinner_peak"]
            df["reservations_x_weekend"] = df["reservations"] * df["is_weekend"]
            df["reservations_x_festive_breakfast"] = (
                df["reservations"]
                * df["is_breakfast_period"]
                * df.get("is_festive_season", 0)
            )
            df["reservations_x_winter_breakfast"] = (
                df["reservations"]
                * df["is_breakfast_period"]
                * ((df["season"] == "winter").astype(int) if "season" in df.columns else 0)
            )

        # ----------------------------
        # LAGS / ROLLING
        # ----------------------------
        if covers_df is not None and train_end_ts is not None:
            self.logger.info(
                "Preparing future lag features using history through %s.",
                pd.Timestamp(train_end_ts),
            )
            lag_features = self._get_scenario_aware_lags(df, covers_df, train_end_ts)
            for col, val in lag_features.items():
                df[col] = val
        else:
            if "covers" in df.columns:
                df["covers_lag_1h"] = df["covers"].shift(1)
                df["covers_lag_1d"] = df["covers"].shift(15)
                df["covers_lag_7d"] = df["covers"].shift(7 * 15)
                df["covers_lag_1w_same_hour"] = df.groupby("hour")["covers"].shift(
                    7 * 15
                )
                df["covers_lag_1d_same_period"] = df.groupby("service_period")[
                    "covers"
                ].shift(15)

                df["covers_roll_mean_3h"] = df["covers"].shift(1).rolling(3).mean()
                df["covers_roll_mean_6h"] = df["covers"].shift(1).rolling(6).mean()
                df["covers_roll_mean_1d"] = df["covers"].shift(1).rolling(15).mean()
                df["covers_roll_mean_3d"] = df["covers"].shift(1).rolling(45).mean()
                df["covers_roll_mean_7d"] = df["covers"].shift(1).rolling(105).mean()

                df["covers_roll_std_1d"] = df["covers"].shift(1).rolling(15).std()
                df["covers_roll_std_3d"] = df["covers"].shift(1).rolling(45).std()
                df["covers_roll_max_1d"] = df["covers"].shift(1).rolling(15).max()
                df["covers_roll_min_1d"] = df["covers"].shift(1).rolling(15).min()
                df["covers_lag_prev_same_period_weektype"] = df.groupby(
                    ["service_period", "is_weekend"]
                )["covers"].shift(1)
                if "season" in df.columns:
                    df["covers_lag_prev_same_period_season"] = df.groupby(
                        ["service_period", "season"]
                    )["covers"].shift(1)

                df["covers_volatility_3d"] = df["covers_roll_std_3d"] / (
                    df["covers_roll_mean_3d"] + 1
                )
                if "reservations" in df.columns:
                    df["reservation_pressure_1d"] = df["reservations"] / (
                        df["covers_roll_mean_1d"] + 1
                    )
                    df["reservation_pressure_3d"] = df["reservations"] / (
                        df["covers_roll_mean_3d"] + 1
                    )

        # ----------------------------
        # CATEGORICALS
        # ----------------------------
        if "season" in df.columns:
            df["season"] = pd.Categorical(
                df["season"], categories=["winter", "summer", "monsoon", "festive"]
            )
            df = pd.get_dummies(df, columns=["season"], prefix="season", dtype=int)

        if "service_period" in df.columns:
            df["service_period"] = pd.Categorical(
                df["service_period"],
                categories=[
                    "pre_service",
                    "breakfast",
                    "lunch",
                    "afternoon",
                    "dinner",
                    "late_night",
                ],
            )
            df = pd.get_dummies(
                df, columns=["service_period"], prefix="period", dtype=int
            )

        if "rain_category" in df.columns:
            df["rain_category"] = pd.Categorical(
                df["rain_category"], categories=["none", "light", "moderate", "heavy"]
            )
            df = pd.get_dummies(
                df, columns=["rain_category"], prefix="rain_cat", dtype=int
            )

        return df

    def prepare_model_frame(self, covers_df):
        full_enhanced = self.build_enhanced_features(covers_df.copy())

        exclude = {
            "timestamp",
            "covers",
            "walk_ins",
            "num_orders",
            "gross_sales",
            "avg_ticket_size",
            "holiday_name",
            "event_name",
        }
        pruned_feature_columns = {
            "day_of_month",
            "month",
            "week_of_year",
            "is_friday",
            "is_month_start",
            "is_month_end",
            "holiday_x_hour",
            "event_x_hour",
            "promo_x_hour",
            "rain_x_hour",
            "temp_x_hour",
            "period_pre_service",
            "period_breakfast",
            "period_lunch",
            "period_afternoon",
            "period_dinner",
            "period_late_night",
            "covers_lag_prev_same_period_weektype",
            "covers_lag_prev_same_period_season",
        }

        base_feature_cols = [c for c in full_enhanced.columns if c not in exclude]
        full_enhanced = full_enhanced.dropna(subset=base_feature_cols).reset_index(drop=True)

        feature_cols = [
            c
            for c in base_feature_cols
            if c not in pruned_feature_columns
        ]
        self.logger.info(
            "Prepared model frame with %s usable row(s) and %s feature(s).",
            len(full_enhanced),
            len(feature_cols),
        )
        return full_enhanced, feature_cols

    def prepare_future_frame(self, future_df, covers_history_df):
        covers_history_df = covers_history_df.copy()
        covers_history_df["timestamp"] = pd.to_datetime(covers_history_df["timestamp"])
        train_end_ts = covers_history_df["timestamp"].max()
        self.logger.info(
            "Preparing future frame for %s row(s) against %s historical row(s).",
            len(future_df),
            len(covers_history_df),
        )

        future_features = self.build_enhanced_features(
            future_df.copy(),
            covers_df=covers_history_df,
            train_end_ts=train_end_ts,
        )

        for col in self.feature_cols:
            if col not in future_features.columns:
                future_features[col] = 0

        extra_cols = [c for c in future_features.columns if c not in self.feature_cols]
        ordered_cols = (
            ["timestamp"]
            + self.feature_cols
            + [c for c in extra_cols if c != "timestamp"]
        )
        future_features = future_features[ordered_cols]
        future_features = future_features.fillna(0)
        self.logger.info(
            "Future frame ready with %s feature column(s).",
            len(self.feature_cols),
        )
        return future_features

    def _get_scenario_aware_lags(self, df, covers_df, train_end_ts):
        history_df = covers_df[covers_df["timestamp"] <= train_end_ts].copy()
        history_df["timestamp"] = pd.to_datetime(history_df["timestamp"])
        history_df = history_df.sort_values("timestamp").reset_index(drop=True)
        self.logger.info(
            "Building scenario-aware lag features for %s forecast row(s) from %s history row(s).",
            len(df),
            len(history_df),
        )

        if history_df.empty:
            zeros = np.zeros(len(df), dtype=float)
            return {
                "covers_lag_1h": zeros.copy(),
                "covers_lag_1d": zeros.copy(),
                "covers_lag_7d": zeros.copy(),
                "covers_lag_1w_same_hour": zeros.copy(),
                "covers_lag_1d_same_period": zeros.copy(),
                "covers_roll_mean_3h": zeros.copy(),
                "covers_roll_mean_6h": zeros.copy(),
                "covers_roll_mean_1d": zeros.copy(),
                "covers_roll_mean_3d": zeros.copy(),
                "covers_roll_mean_7d": zeros.copy(),
                "covers_roll_std_1d": zeros.copy(),
                "covers_roll_std_3d": zeros.copy(),
                "covers_roll_max_1d": zeros.copy(),
                "covers_roll_min_1d": zeros.copy(),
                "covers_lag_prev_same_period_weektype": zeros.copy(),
                "covers_lag_prev_same_period_season": zeros.copy(),
                "covers_volatility_3d": zeros.copy(),
            }

        history_df["service_period"] = history_df["timestamp"].dt.hour.apply(get_service_period)
        history_df["hour"] = history_df["timestamp"].dt.hour
        history_df["is_weekend"] = history_df.get("is_weekend", 0).fillna(0).astype(int)

        def _scenario_subset(row: pd.Series, prior_history_df: pd.DataFrame) -> pd.DataFrame:
            if prior_history_df.empty:
                return history_df

            mask = pd.Series(True, index=prior_history_df.index)

            if "holiday_flag" in row and row["holiday_flag"]:
                mask &= prior_history_df.get("holiday_flag", 0).fillna(0) == 1
            if "event_flag" in row and row["event_flag"]:
                mask &= prior_history_df.get("event_flag", 0).fillna(0) == 1
            if "promotion_flag" in row and row["promotion_flag"]:
                mask &= prior_history_df.get("promotion_flag", 0).fillna(0) == 1
            if "rain_mm" in row and row["rain_mm"] >= 15:
                mask &= prior_history_df.get("rain_mm", 0).fillna(0) >= 15
            elif "rain_mm" in row and row["rain_mm"] > 0:
                mask &= prior_history_df.get("rain_mm", 0).fillna(0) > 0
            if "temp_c" in row and row["temp_c"] >= 35:
                mask &= prior_history_df.get("temp_c", 0).fillna(0) >= 35
            if "is_weekend" in row:
                mask &= prior_history_df["is_weekend"] == int(row["is_weekend"])

            subset = prior_history_df[mask].copy()
            if len(subset) < 24:
                subset = prior_history_df.copy()

            return subset.sort_values("timestamp").reset_index(drop=True)

        def _latest_cover(frame: pd.DataFrame) -> float | None:
            if frame.empty:
                return None
            return float(frame.iloc[-1]["covers"])

        def _cover_at_timestamp(frame: pd.DataFrame, timestamp: pd.Timestamp) -> float | None:
            match_df = frame[frame["timestamp"] == timestamp]
            return _latest_cover(match_df)

        lag_feature_map = {
            "covers_lag_1h": [],
            "covers_lag_1d": [],
            "covers_lag_7d": [],
            "covers_lag_1w_same_hour": [],
            "covers_lag_1d_same_period": [],
            "covers_roll_mean_3h": [],
            "covers_roll_mean_6h": [],
            "covers_roll_mean_1d": [],
            "covers_roll_mean_3d": [],
            "covers_roll_mean_7d": [],
            "covers_roll_std_1d": [],
            "covers_roll_std_3d": [],
            "covers_roll_max_1d": [],
            "covers_roll_min_1d": [],
            "covers_lag_prev_same_period_weektype": [],
            "covers_lag_prev_same_period_season": [],
            "covers_volatility_3d": [],
        }

        for _, row in df.iterrows():
            row_ts = pd.Timestamp(row["timestamp"])
            prior_history_df = history_df[history_df["timestamp"] < row_ts].copy()
            if prior_history_df.empty:
                prior_history_df = history_df.copy()

            scenario_df = _scenario_subset(row, prior_history_df)
            scenario_median = float(scenario_df["covers"].median())
            scenario_std = float(scenario_df["covers"].std()) if len(scenario_df) > 1 else 0.0

            same_hour_df = prior_history_df[prior_history_df["hour"] == row["hour"]]
            same_period_df = prior_history_df[
                prior_history_df["service_period"] == row["service_period"]
            ]
            same_period_weektype_df = prior_history_df[
                (prior_history_df["service_period"] == row["service_period"])
                & (prior_history_df["is_weekend"] == int(row.get("is_weekend", 0)))
            ]
            same_period_season_df = prior_history_df[
                prior_history_df["service_period"] == row["service_period"]
            ]
            if "season" in row and "season" in prior_history_df.columns:
                same_period_season_df = same_period_season_df[
                    same_period_season_df["season"] == row["season"]
                ]

            lag_1h = _latest_cover(prior_history_df)
            lag_1d = _cover_at_timestamp(prior_history_df, row_ts - pd.Timedelta(days=1))
            lag_7d = _cover_at_timestamp(prior_history_df, row_ts - pd.Timedelta(days=7))
            lag_1w_same_hour = lag_7d if lag_7d is not None else _latest_cover(same_hour_df)

            if lag_1d is None:
                lag_1d = _latest_cover(same_hour_df)
            if lag_7d is None:
                lag_7d = _latest_cover(same_hour_df)

            roll_3h = prior_history_df["covers"].tail(3)
            roll_6h = prior_history_df["covers"].tail(6)
            roll_1d = prior_history_df["covers"].tail(15)
            roll_3d = prior_history_df["covers"].tail(45)
            roll_7d = prior_history_df["covers"].tail(105)

            roll_mean_3d = float(roll_3d.mean()) if len(roll_3d) > 0 else scenario_median
            roll_std_3d = float(roll_3d.std()) if len(roll_3d) > 1 else scenario_std

            lag_feature_map["covers_lag_1h"].append(
                lag_1h if lag_1h is not None else scenario_median
            )
            lag_feature_map["covers_lag_1d"].append(
                lag_1d if lag_1d is not None else scenario_median
            )
            lag_feature_map["covers_lag_7d"].append(
                lag_7d if lag_7d is not None else scenario_median
            )
            lag_feature_map["covers_lag_1w_same_hour"].append(
                lag_1w_same_hour if lag_1w_same_hour is not None else scenario_median
            )
            lag_feature_map["covers_lag_1d_same_period"].append(
                _latest_cover(same_period_df) if not same_period_df.empty else scenario_median
            )
            lag_feature_map["covers_roll_mean_3h"].append(
                float(roll_3h.mean()) if len(roll_3h) > 0 else scenario_median
            )
            lag_feature_map["covers_roll_mean_6h"].append(
                float(roll_6h.mean()) if len(roll_6h) > 0 else scenario_median
            )
            lag_feature_map["covers_roll_mean_1d"].append(
                float(roll_1d.mean()) if len(roll_1d) > 0 else scenario_median
            )
            lag_feature_map["covers_roll_mean_3d"].append(roll_mean_3d)
            lag_feature_map["covers_roll_mean_7d"].append(
                float(roll_7d.mean()) if len(roll_7d) > 0 else scenario_median
            )
            lag_feature_map["covers_roll_std_1d"].append(
                float(roll_1d.std()) if len(roll_1d) > 1 else scenario_std
            )
            lag_feature_map["covers_roll_std_3d"].append(roll_std_3d)
            lag_feature_map["covers_roll_max_1d"].append(
                float(roll_1d.max()) if len(roll_1d) > 0 else scenario_median
            )
            lag_feature_map["covers_roll_min_1d"].append(
                float(roll_1d.min()) if len(roll_1d) > 0 else scenario_median
            )
            lag_feature_map["covers_lag_prev_same_period_weektype"].append(
                _latest_cover(same_period_weektype_df)
                if not same_period_weektype_df.empty
                else scenario_median
            )
            lag_feature_map["covers_lag_prev_same_period_season"].append(
                _latest_cover(same_period_season_df)
                if not same_period_season_df.empty
                else scenario_median
            )
            lag_feature_map["covers_volatility_3d"].append(
                roll_std_3d / (roll_mean_3d + 1)
                if roll_mean_3d >= 0
                else 0.15
            )

        self.logger.info(
            "Scenario-aware lag features prepared from %s through %s.",
            history_df["timestamp"].min(),
            history_df["timestamp"].max(),
        )
        return {col: np.asarray(values, dtype=float) for col, values in lag_feature_map.items()}

    def train(
        self,
        X_train,
        y_train,
        sample_weight=None,
        X_eval=None,
        y_eval=None,
        eval_sample_weight=None,
    ):
        y_train_log = np.log1p(y_train)

        use_early_stopping = (
            X_eval is not None and y_eval is not None and len(X_eval) > 0
        )
        model_params = {
            "n_estimators": 800,
            "max_depth": 4,
            "learning_rate": 0.025,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "min_child_weight": 2,
            "gamma": 0.05,
            "reg_alpha": 0.25,
            "reg_lambda": 5.0,
            "tree_method": "hist",
            "objective": "reg:squarederror",
            "random_state": 42,
            "n_jobs": 1,
            "early_stopping_rounds": self.early_stopping_rounds if use_early_stopping else None,
        }
        model_params.update(self.model_params)
        self.logger.info(
            "Training forecaster on %s row(s) with %s feature(s); eval_rows=%s.",
            len(X_train),
            len(X_train.columns),
            len(X_eval) if X_eval is not None else 0,
        )
        self.model = XGBRegressor(**model_params)
        fit_kwargs = {
            "X": X_train,
            "y": y_train_log,
            "sample_weight": sample_weight,
            "verbose": False,
        }
        if X_eval is not None and y_eval is not None and len(X_eval) > 0:
            fit_kwargs["eval_set"] = [(X_eval, np.log1p(y_eval))]
            if eval_sample_weight is not None:
                fit_kwargs["sample_weight_eval_set"] = [eval_sample_weight]

        self.model.fit(**fit_kwargs)
        self.logger.info("Forecaster training finished.")

    def predict(self, X):
        self.logger.info("Generating predictions for %s row(s).", len(X))
        pred_log = self.model.predict(X)
        pred = np.expm1(pred_log)
        pred = np.clip(pred, 0, None)
        return clip_round_positive(pred)

    def evaluate(self, X_test, y_test):
        pred = self.predict(X_test)
        error = pred - y_test
        mae = mean_absolute_error(y_test, pred)
        rmse = np.sqrt(mean_squared_error(y_test, pred))
        wape = np.sum(np.abs(y_test - pred)) / np.sum(np.abs(y_test))
        mape = np.mean(
            np.where(np.asarray(y_test) > 0, np.abs(error) / np.asarray(y_test), 0.0)
        )
        bias = float(np.mean(error))
        metrics = {
            "mae": mae,
            "rmse": rmse,
            "wape": wape,
            "mape": float(mape),
            "bias": bias,
        }
        self.logger.info(
            "Evaluation complete: mae=%.4f rmse=%.4f wape=%.4f mape=%.4f bias=%.4f.",
            metrics["mae"],
            metrics["rmse"],
            metrics["wape"],
            metrics["mape"],
            metrics["bias"],
        )
        return metrics


def run_single_xgb_forecast(covers_df, holdout_hours=7 * 15):
    covers_df = covers_df.copy()
    covers_df["timestamp"] = pd.to_datetime(covers_df["timestamp"])
    covers_df = covers_df.sort_values("timestamp").reset_index(drop=True)

    forecaster = RestaurantForecaster()

    full_enhanced, feature_cols = forecaster.prepare_model_frame(covers_df.copy())

    train_df = full_enhanced.iloc[:-holdout_hours].copy()
    test_df = full_enhanced.iloc[-holdout_hours:].copy()

    X_train = train_df[feature_cols]
    y_train = train_df["covers"]
    X_test = test_df[feature_cols]
    y_test = test_df["covers"]

    forecaster.feature_cols = feature_cols
    forecaster.train(X_train, y_train)

    pred = forecaster.predict(X_test)
    metrics = forecaster.evaluate(X_test, y_test)

    print("\n" + "=" * 80)
    print("SINGLE XGBOOST FORECAST RESULTS")
    print("=" * 80)
    print(f"Features used: {len(feature_cols)}")
    print(f"Train rows: {len(train_df)}")
    print(f"Test rows: {len(test_df)}")
    print(f"MAE : {metrics['mae']:.3f}")
    print(f"RMSE: {metrics['rmse']:.3f}")
    print(f"WAPE: {metrics['wape']:.3f}")
    print("=" * 80)

    return forecaster, train_df, test_df, pred, metrics
