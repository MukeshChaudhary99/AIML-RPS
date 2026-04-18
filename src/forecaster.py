import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


class RestaurantForecaster:
    """
    Single-model restaurant forecaster using XGBoost.
    Includes:
    - feature engineering
    - point forecast
    - scenario testing support
    """

    def __init__(self, feature_cols=None):
        self.feature_cols = feature_cols or []
        self.model = None

    def build_enhanced_features(self, df, covers_df=None, train_end_ts=None):
        df = df.copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values("timestamp").reset_index(drop=True)

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
        def get_service_period(hour):
            if hour < 8:
                return "pre_service"
            elif 8 <= hour <= 11:
                return "breakfast"
            elif 12 <= hour <= 15:
                return "lunch"
            elif 16 <= hour <= 18:
                return "afternoon"
            elif 19 <= hour <= 22:
                return "dinner"
            else:
                return "late_night"

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

        feature_cols = [c for c in full_enhanced.columns if c not in exclude]
        full_enhanced = full_enhanced.dropna(subset=feature_cols).reset_index(drop=True)
        return full_enhanced, feature_cols

    def prepare_future_frame(self, future_df, covers_history_df):
        covers_history_df = covers_history_df.copy()
        covers_history_df["timestamp"] = pd.to_datetime(covers_history_df["timestamp"])
        train_end_ts = covers_history_df["timestamp"].max()

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
        return future_features

    def _get_scenario_aware_lags(self, df, covers_df, train_end_ts):
        row = df.iloc[0]
        mask = pd.Series(True, index=covers_df.index)

        if "holiday_flag" in row and row["holiday_flag"]:
            mask &= covers_df["holiday_flag"] == 1
        if "event_flag" in row and row["event_flag"]:
            mask &= covers_df["event_flag"] == 1
        if "promotion_flag" in row and row["promotion_flag"]:
            mask &= covers_df["promotion_flag"] == 1
        if "rain_mm" in row and row["rain_mm"] >= 15:
            mask &= covers_df["rain_mm"] >= 15
        if "temp_c" in row and row["temp_c"] >= 35:
            mask &= covers_df["temp_c"] >= 35

        subset = covers_df[mask & (covers_df["timestamp"] <= train_end_ts)].copy()
        if len(subset) < 50:
            subset = covers_df[covers_df["timestamp"] <= train_end_ts].copy()

        subset = subset.sort_values("timestamp").reset_index(drop=True)

        return {
            "covers_lag_1h": subset["covers"].median(),
            "covers_lag_1d": subset["covers"].median(),
            "covers_lag_7d": subset["covers"].median(),
            "covers_lag_1w_same_hour": subset["covers"].median(),
            "covers_lag_1d_same_period": subset["covers"].median(),
            "covers_roll_mean_3h": subset["covers"].median(),
            "covers_roll_mean_6h": subset["covers"].median(),
            "covers_roll_mean_1d": subset["covers"].median(),
            "covers_roll_mean_3d": subset["covers"].median(),
            "covers_roll_mean_7d": subset["covers"].median(),
            "covers_roll_std_1d": subset["covers"].std(),
            "covers_roll_std_3d": subset["covers"].std(),
            "covers_roll_max_1d": float(np.nanpercentile(subset["covers"], 75)),
            "covers_roll_min_1d": float(np.nanpercentile(subset["covers"], 25)),
            "covers_lag_prev_same_period_weektype": subset["covers"].median(),
            "covers_lag_prev_same_period_season": subset["covers"].median(),
            "covers_volatility_3d": 0.15,
        }

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
        self.model = XGBRegressor(
            n_estimators=800,
            max_depth=4,
            learning_rate=0.025,
            subsample=0.9,
            colsample_bytree=0.9,
            min_child_weight=2,
            gamma=0.05,
            reg_alpha=0.25,
            reg_lambda=5.0,
            tree_method="hist",
            objective="reg:squarederror",
            random_state=42,
            n_jobs=-1,
            early_stopping_rounds=50 if use_early_stopping else None,
        )
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

    def predict(self, X):
        pred_log = self.model.predict(X)
        pred = np.expm1(pred_log).clip(min=0)
        return np.rint(pred).astype(int)

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
        return {
            "mae": mae,
            "rmse": rmse,
            "wape": wape,
            "mape": float(mape),
            "bias": bias,
        }


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
