from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils import get_service_period


DATA_DIR = Path("V2Data")
START_DATE = datetime(2025, 1, 1)
END_DATE = datetime(2025, 12, 31)
OPEN_HOUR = 8
CLOSE_HOUR = 22
RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)


def build_hourly_timestamps(
    start_date: datetime,
    end_date: datetime,
    open_hour: int = OPEN_HOUR,
    close_hour: int = CLOSE_HOUR,
) -> list[datetime]:
    timestamps: list[datetime] = []
    current = start_date
    while current <= end_date:
        for hour in range(open_hour, close_hour + 1):
            timestamps.append(datetime(current.year, current.month, current.day, hour))
        current += timedelta(days=1)
    return timestamps


def get_season(month: int) -> str:
    if month in [3, 4, 5]:
        return "summer"
    if month in [6, 7, 8, 9]:
        return "monsoon"
    if month in [10, 11]:
        return "festive"
    return "winter"


def _build_campaign_days(
    all_dates: list[datetime],
    season_map: dict[str, str],
) -> set[str]:
    campaign_days: set[str] = set()

    candidate_indices = list(range(0, len(all_dates) - 4, 18))
    for idx in candidate_indices:
        start = all_dates[idx]
        length = int(np.random.choice([2, 3, 4], p=[0.35, 0.45, 0.20]))
        for offset in range(length):
            if idx + offset < len(all_dates):
                campaign_days.add(all_dates[idx + offset].strftime("%Y-%m-%d"))

    for current in all_dates:
        date_str = current.strftime("%Y-%m-%d")
        season = season_map[date_str]
        if season == "festive" and current.weekday() >= 4:
            campaign_days.add(date_str)
        if current.day >= 26 and current.weekday() < 5:
            campaign_days.add(date_str)

    return campaign_days


def _build_daily_context(start_date: datetime, end_date: datetime) -> pd.DataFrame:
    holiday_map = {
        "2025-01-01": "New Year",
        "2025-01-26": "Republic Day",
        "2025-08-15": "Independence Day",
        "2025-10-02": "Gandhi Jayanti",
        "2025-12-25": "Christmas",
    }
    fixed_event_map = {
        "2025-03-14": ("Holi Crowd", "lunch_dinner"),
        "2025-08-27": ("Ganesh Festival", "dinner"),
        "2025-10-20": ("Diwali Rush", "dinner"),
        "2025-11-05": ("Wedding Season Event", "lunch_dinner"),
    }

    all_dates = pd.date_range(start_date, end_date, freq="D").to_pydatetime().tolist()
    season_map = {d.strftime("%Y-%m-%d"): get_season(d.month) for d in all_dates}
    campaign_days = _build_campaign_days(all_dates, season_map)

    event_templates = [
        ("Corporate Lunch Booking", "lunch"),
        ("College Event Crowd", "afternoon"),
        ("Cricket Match", "dinner"),
        ("Local Fair", "lunch_dinner"),
        ("Birthday Party", "dinner"),
        ("Conference Catering", "lunch"),
    ]

    rows: list[dict] = []
    for current in all_dates:
        date_str = current.strftime("%Y-%m-%d")
        season = season_map[date_str]
        is_weekend = int(current.weekday() >= 5)

        holiday_name = holiday_map.get(date_str, "")
        holiday_flag = int(bool(holiday_name))

        if date_str in fixed_event_map:
            event_name, event_window = fixed_event_map[date_str]
        else:
            event_prob = 0.04
            if season == "festive":
                event_prob += 0.03
            if is_weekend:
                event_prob += 0.03
            if np.random.rand() < event_prob:
                event_name, event_window = event_templates[
                    np.random.randint(0, len(event_templates))
                ]
            else:
                event_name, event_window = "", ""

        promotion_flag = int(date_str in campaign_days)

        base_temp = {
            "summer": 33,
            "monsoon": 28,
            "festive": 29,
            "winter": 20,
        }[season]
        daily_temp_offset = np.random.normal(0, 1.8)

        if season == "monsoon":
            weather_regime = np.random.choice(
                ["dry", "light_rain", "steady_rain", "heavy_spell"],
                p=[0.28, 0.26, 0.30, 0.16],
            )
        else:
            weather_regime = np.random.choice(
                ["dry", "light_rain", "steady_rain"],
                p=[0.74, 0.20, 0.06],
            )

        rain_start = int(np.random.choice([8, 10, 12, 15, 17, 19]))
        rain_duration = int(np.random.choice([2, 3, 4, 5]))

        rows.append(
            {
                "date": current.date(),
                "date_str": date_str,
                "season": season,
                "is_weekend": is_weekend,
                "holiday_flag": holiday_flag,
                "holiday_name": holiday_name,
                "event_name": event_name,
                "event_window": event_window,
                "event_flag": int(bool(event_name)),
                "promotion_flag": promotion_flag,
                "daily_temp_base": round(float(base_temp + daily_temp_offset), 2),
                "weather_regime": weather_regime,
                "rain_start": rain_start,
                "rain_duration": rain_duration,
            }
        )

    return pd.DataFrame(rows)


def _event_active_for_hour(event_window: str, hour: int) -> int:
    if event_window == "breakfast":
        return int(8 <= hour <= 11)
    if event_window == "lunch":
        return int(12 <= hour <= 15)
    if event_window == "afternoon":
        return int(16 <= hour <= 18)
    if event_window == "dinner":
        return int(19 <= hour <= 22)
    if event_window == "lunch_dinner":
        return int(12 <= hour <= 15 or 19 <= hour <= 22)
    return 0


def gen_external(ts_list: list[datetime]) -> pd.DataFrame:
    daily_context_df = _build_daily_context(START_DATE, END_DATE)
    daily_context_map = {
        row["date_str"]: row for _, row in daily_context_df.iterrows()
    }

    rows: list[dict] = []
    previous_rain = 0.0

    for timestamp in ts_list:
        date_str = timestamp.strftime("%Y-%m-%d")
        daily_row = daily_context_map[date_str]
        season = daily_row["season"]
        hour = timestamp.hour

        hour_heat = 3 if 12 <= hour <= 16 else 0
        morning_cool = -2 if 8 <= hour <= 10 else 0
        temp_c = np.random.normal(
            daily_row["daily_temp_base"] + hour_heat + morning_cool,
            1.5,
        )

        rain_mm = 0.0
        weather_regime = daily_row["weather_regime"]
        rain_start = int(daily_row["rain_start"])
        rain_end = rain_start + int(daily_row["rain_duration"])
        if weather_regime == "light_rain" and rain_start <= hour <= rain_end:
            rain_mm = float(np.random.choice([1, 2, 3, 5], p=[0.25, 0.35, 0.25, 0.15]))
        elif weather_regime == "steady_rain" and rain_start <= hour <= rain_end:
            rain_mm = float(np.random.choice([5, 8, 10, 12], p=[0.25, 0.30, 0.25, 0.20]))
        elif weather_regime == "heavy_spell" and rain_start <= hour <= rain_end:
            rain_mm = float(np.random.choice([10, 15, 20, 35], p=[0.20, 0.35, 0.30, 0.15]))

        if previous_rain > 0 and rain_mm == 0 and np.random.rand() < 0.35:
            rain_mm = float(max(1, previous_rain * np.random.uniform(0.15, 0.35)))
        previous_rain = rain_mm

        rows.append(
            {
                "timestamp": timestamp,
                "holiday_flag": int(daily_row["holiday_flag"]),
                "event_flag": _event_active_for_hour(daily_row["event_window"], hour),
                "holiday_name": daily_row["holiday_name"],
                "event_name": daily_row["event_name"],
                "temp_c": round(float(temp_c), 1),
                "rain_mm": round(float(rain_mm), 1),
                "is_weekend": int(daily_row["is_weekend"]),
                "season": season,
            }
        )

    return pd.DataFrame(rows)


def gen_sales(df_ext: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    previous_covers = 26.0
    previous_hour_by_hour: dict[int, float] = {}

    for _, row in df_ext.sort_values("timestamp").iterrows():
        timestamp = row["timestamp"]
        hour = int(timestamp.hour)
        season = row["season"]
        date_progress = (timestamp.timetuple().tm_yday - 1) / 364.0
        service_period = get_service_period(hour)

        if service_period == "breakfast":
            reservation_base = 5.0
            walkin_base = 10.0
        elif service_period == "lunch":
            reservation_base = 10.5
            walkin_base = 18.0
        elif service_period == "afternoon":
            reservation_base = 3.0
            walkin_base = 16.0
        else:
            reservation_base = 13.0
            walkin_base = 19.5

        if row["is_weekend"]:
            reservation_base += 1.0
            walkin_base += 6.0
        if row["holiday_flag"]:
            reservation_base += 5.0
            walkin_base += 9.0
        if row["event_flag"]:
            reservation_base += 3.0
            walkin_base += 11.0
        if season == "festive":
            reservation_base += 1.5
            walkin_base += 2.0
        if season == "winter" and service_period == "dinner":
            walkin_base += 2.5
        if season == "monsoon" and service_period == "afternoon":
            walkin_base += 2.0

        promotion_flag = int(row["is_weekend"] and np.random.rand() < 0.12)
        if service_period == "lunch" and np.random.rand() < 0.08:
            promotion_flag = 1
        if service_period == "breakfast" and season == "festive" and np.random.rand() < 0.05:
            promotion_flag = 1

        if promotion_flag:
            reservation_base += 1.0
            walkin_base += 5.0

        if row["rain_mm"] >= 15:
            if service_period == "breakfast":
                reservation_base += 0.5
                walkin_base -= 2.0
            elif service_period == "afternoon":
                walkin_base += 3.0
            else:
                walkin_base -= 5.5
        elif row["rain_mm"] > 0:
            if service_period in {"breakfast", "afternoon"}:
                walkin_base += 1.0
            else:
                walkin_base -= 1.5

        growth_multiplier = 1.0 + (0.08 * date_progress)
        reservation_base *= growth_multiplier
        walkin_base *= growth_multiplier

        prev_same_hour = previous_hour_by_hour.get(hour, walkin_base)
        walkin_mean = (
            walkin_base
            + 0.10 * previous_covers
            + 0.18 * prev_same_hour
            + np.random.normal(0, 2.2)
        )
        reservation_mean = reservation_base + np.random.normal(0, 1.4)

        reservations = int(max(0, np.random.poisson(max(1.0, reservation_mean))))
        walk_ins = int(max(0, np.random.poisson(max(1.0, walkin_mean))))
        covers = max(4, reservations + walk_ins)

        previous_covers = float(covers)
        previous_hour_by_hour[hour] = float(covers)

        if service_period == "breakfast":
            avg_ticket = np.random.normal(7.5, 0.9)
        elif service_period == "lunch":
            avg_ticket = np.random.normal(13.0, 1.6)
        elif service_period == "afternoon":
            avg_ticket = np.random.normal(8.5, 1.1)
        else:
            avg_ticket = np.random.normal(15.0, 2.0)

        if row["event_flag"]:
            avg_ticket += 1.5
        if promotion_flag:
            avg_ticket -= 0.8

        avg_ticket = max(4.5, round(float(avg_ticket), 2))
        num_orders = max(1, int(round(covers * np.random.uniform(0.84, 0.98))))
        gross_sales = round(covers * avg_ticket, 2)

        rows.append(
            {
                "timestamp": timestamp,
                "covers": int(covers),
                "gross_sales": gross_sales,
                "num_orders": num_orders,
                "avg_ticket_size": avg_ticket,
                "reservations": int(reservations),
                "walk_ins": int(walk_ins),
                "promotion_flag": promotion_flag,
            }
        )

    return pd.DataFrame(rows)


def choose_mix_by_context(hour: int, rain_mm: float, is_weekend: int, event_flag: int) -> dict[str, float]:
    if 8 <= hour <= 11:
        mix = {
            "Breakfast": 0.48,
            "Drinks": 0.25,
            "Snacks": 0.15,
            "Main": 0.02,
            "Starter": 0.00,
            "Salad": 0.02,
            "Bread": 0.03,
            "Dessert": 0.05,
        }
    elif 12 <= hour <= 15:
        mix = {
            "Breakfast": 0.00,
            "Drinks": 0.12,
            "Snacks": 0.10,
            "Main": 0.48,
            "Starter": 0.10,
            "Salad": 0.05,
            "Bread": 0.10,
            "Dessert": 0.05,
        }
    elif 16 <= hour <= 18:
        mix = {
            "Breakfast": 0.00,
            "Drinks": 0.28,
            "Snacks": 0.42,
            "Main": 0.10,
            "Starter": 0.06,
            "Salad": 0.02,
            "Bread": 0.00,
            "Dessert": 0.12,
        }
    else:
        mix = {
            "Breakfast": 0.00,
            "Drinks": 0.10,
            "Snacks": 0.08,
            "Main": 0.52,
            "Starter": 0.12,
            "Salad": 0.05,
            "Bread": 0.08,
            "Dessert": 0.05,
        }

    if rain_mm >= 10:
        mix["Drinks"] += 0.10
        mix["Snacks"] += 0.08
        mix["Main"] = max(0.10, mix["Main"] - 0.12)
    if is_weekend:
        mix["Starter"] += 0.03
        mix["Dessert"] += 0.03
        mix["Main"] += 0.04
    if event_flag:
        mix["Main"] += 0.05
        mix["Starter"] += 0.04
        mix["Drinks"] += 0.01

    total = sum(mix.values())
    return {key: value / total for key, value in mix.items()}


def gen_historical_menu_sales(
    df_sales: pd.DataFrame,
    df_ext: pd.DataFrame,
    menu_df: pd.DataFrame,
) -> pd.DataFrame:
    merged = df_sales.merge(
        df_ext[["timestamp", "rain_mm", "is_weekend", "event_flag"]],
        on="timestamp",
        how="left",
    )
    rows: list[dict] = []

    for _, row in merged.iterrows():
        timestamp = row["timestamp"]
        covers = int(row["covers"])
        hour = int(timestamp.hour)

        mix = choose_mix_by_context(
            hour=hour,
            rain_mm=float(row["rain_mm"]),
            is_weekend=int(row["is_weekend"]),
            event_flag=int(row["event_flag"]),
        )

        if 8 <= hour <= 11:
            item_factor = np.random.uniform(1.00, 1.15)
        elif 12 <= hour <= 15:
            item_factor = np.random.uniform(1.15, 1.35)
        elif 16 <= hour <= 18:
            item_factor = np.random.uniform(1.05, 1.25)
        else:
            item_factor = np.random.uniform(1.15, 1.35)

        total_item_qty = max(covers, int(round(covers * item_factor)))

        category_qty: dict[str, int] = {}
        allocated = 0
        categories = list(mix.keys())
        for idx, category in enumerate(categories):
            if idx == len(categories) - 1:
                qty = total_item_qty - allocated
            else:
                qty = int(round(total_item_qty * mix[category]))
                allocated += qty
            category_qty[category] = max(0, qty)

        def item_allowed(item_service_period: str, current_hour: int) -> bool:
            if item_service_period == "all_day":
                return True
            if item_service_period == "breakfast":
                return 8 <= current_hour <= 11
            if item_service_period == "evening":
                return 16 <= current_hour <= 18
            if item_service_period == "lunch_dinner":
                return 12 <= current_hour <= 15 or 19 <= current_hour <= 22
            return True

        for category, qty in category_qty.items():
            if qty <= 0:
                continue

            eligible = menu_df[menu_df["category"] == category].copy()
            eligible = eligible[eligible["service_period"].apply(lambda x: item_allowed(x, hour))]
            if eligible.empty:
                continue

            weights = np.ones(len(eligible), dtype=float)
            for idx, item_name in enumerate(eligible["menu_item_name"]):
                item_name_lower = item_name.lower()
                if row["rain_mm"] >= 10:
                    if any(keyword in item_name_lower for keyword in ["tea", "coffee", "pakora", "samosa"]):
                        weights[idx] *= 2.5
                    if any(keyword in item_name_lower for keyword in ["biryani", "pizza"]):
                        weights[idx] *= 0.8

            weights = weights / weights.sum()
            item_counts = np.random.multinomial(qty, weights)

            for (_, item_row), item_qty in zip(eligible.iterrows(), item_counts):
                if item_qty > 0:
                    rows.append(
                        {
                            "timestamp": timestamp,
                            "menu_item_id": item_row["menu_item_id"],
                            "qty_sold": int(item_qty),
                        }
                    )

    return pd.DataFrame(rows)


def main() -> None:
    timestamps = build_hourly_timestamps(
        start_date=START_DATE,
        end_date=END_DATE,
        open_hour=OPEN_HOUR,
        close_hour=CLOSE_HOUR,
    )
    menu_df = pd.read_csv(DATA_DIR / "menu_items_master.csv")

    external_df = gen_external(timestamps)
    sales_df = gen_sales(external_df)
    historical_menu_sales_df = gen_historical_menu_sales(sales_df, external_df, menu_df)

    external_df.to_csv(DATA_DIR / "external_features.csv", index=False)
    sales_df.to_csv(DATA_DIR / "historical_sales.csv", index=False)
    historical_menu_sales_df.to_csv(DATA_DIR / "historical_menu_sales.csv", index=False)

    print("Generated files:")
    print("-", DATA_DIR / "external_features.csv")
    print("-", DATA_DIR / "historical_sales.csv")
    print("-", DATA_DIR / "historical_menu_sales.csv")


if __name__ == "__main__":
    main()
