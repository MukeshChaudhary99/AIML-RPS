# Restaurant Resource Planning System

This project is a production-style prototype for a restaurant planning system that predicts:

- hourly covers
- staff requirements by role and station
- ingredient demand and purchase recommendations
- forecast corrections through a feedback loop

The system is built around an XGBoost covers forecaster, operational planning modules, and a FastAPI server with a startup dashboard.

For the detailed feedback loop flow, see [FeedBackWorkflow.md](/home/mukeshc/PP/RPS/FeedBackWorkflow.md).

## What The System Does

The planning flow is:

1. forecast hourly covers
2. convert covers into station workload and staffing
3. convert covers into menu demand, then ingredient demand
4. capture actual outcomes and manager corrections
5. retrain and adjust future predictions over time

The current API server also renders a dashboard at the root route so the startup validation summary is easy to inspect.

## Project Structure

- [src/forecaster.py](/home/mukeshc/PP/RPS/src/forecaster.py): XGBoost covers model and feature engineering
- [src/feedback_loop.py](/home/mukeshc/PP/RPS/src/feedback_loop.py): logging, correction logic, retraining loop
- [src/staff_planner.py](/home/mukeshc/PP/RPS/src/staff_planner.py): staffing and shift planning
- [src/ingredient_planner.py](/home/mukeshc/PP/RPS/src/ingredient_planner.py): ingredient demand and purchase planning
- [src/service.py](/home/mukeshc/PP/RPS/src/service.py): orchestration layer and dashboard HTML
- [src/api.py](/home/mukeshc/PP/RPS/src/api.py): FastAPI routes
- [main.py](/home/mukeshc/PP/RPS/main.py): server entry point
- [api_test_scenarios.json](/home/mukeshc/PP/RPS/api_test_scenarios.json): ready-to-use request bodies for API testing
- [V2Data](/home/mukeshc/PP/RPS/V2Data): source CSV data used by the system

## Main Data Files

The system reads its core inputs from `V2Data/`:

- `historical_sales.csv`
  Contains hourly covers, reservations, walk-ins, sales, and order counts.
- `external_features.csv`
  Contains weather, holiday, event, weekend, and season context by hour.
- `historical_menu_sales.csv`
  Contains historical sold quantity by menu item and hour.
- `menu_items_master.csv`
  Contains menu metadata such as category, service period, station, and prep minutes.
- `menu_ingredients.csv`
  Maps menu items to ingredients.
- `ingredient_master.csv`
  Contains shelf life, lead time, stock, safety stock, and minimum order quantity.
- `staff_roles.csv`
  Contains staffing capacity and cost by role and station.

If you want to refresh the synthetic dataset, run:

```bash
uv run python src/data_generator.py
```

This rebuilds the main hourly sales, external context, and historical menu sales files inside `V2Data/`.

## Startup Behavior

When the server starts:

1. the covers model is trained automatically
2. a startup validation split is created using diverse recent scenarios
3. summary metrics are computed
4. the dashboard becomes available at `/`

Startup validation is shown directly in the dashboard. It is not meant to generate separate startup CSV artifacts anymore.

## Dashboard

Open the server root route:

```bash
http://localhost:8000/
```

The dashboard shows:

- startup model summary
- current model status
- scenario coverage used in startup validation
- startup prediction preview table
- startup scenario error summary

This is the easiest place to inspect model quality after boot.

## API Endpoints

Core endpoints:

- `GET /`
  Startup dashboard in HTML.
- `GET /health`
  Health and model status summary.
- `POST /forecast/covers`
  Predict hourly covers.
- `POST /plan/staff`
  Predict covers and generate staffing outputs.
- `POST /plan/ingredients`
  Predict covers and generate ingredient outputs.
- `POST /plan/full`
  Predict covers, staffing, and ingredients together.
- `POST /feedback`
  Submit one or many hourly actuals and manager corrections.

Optional feedback inspection endpoints still exist:

- `GET /feedback/scenario-corrections`
- `GET /feedback/summary`

## API Request Shape

The hourly planning endpoints accept a list of hourly context rows. Example:

```json
{
  "hours": [
    {
      "timestamp": "2026-04-21T12:00:00",
      "reservations": 14,
      "holiday_flag": 0,
      "event_flag": 0,
      "temp_c": 31.0,
      "rain_mm": 0.0,
      "is_weekend": 0,
      "season": "summer",
      "promotion_flag": 0
    }
  ],
  "auto_retrain": true
}
```

## Feedback Loop

The feedback endpoint accepts one or many hourly feedback rows.

Example:

```json
{
  "entries": [
    {
      "timestamp": "2026-07-15T20:00:00",
      "predicted_covers": 34,
      "actual_covers": 26,
      "actual_reservations": 10,
      "actual_walk_ins": 16,
      "rain_mm": 14.0,
      "temp_c": 24.0,
      "is_weekend": 0,
      "season": "monsoon",
      "manager_note": "Heavy rain reduced walk-ins more than expected."
    }
  ],
  "retrain": true
}
```

The feedback loop does two things:

- short-term scenario correction based on observed error
- longer-term retraining on corrected actuals

## API Scenario Fixture File

Use [api_test_scenarios.json](./api_test_scenarios.json) for quick API testing.

It includes:

- common forecast/planning scenarios
- breakfast, winter, festive, rain, event, and promotion cases
- feedback examples for rain drops and event spikes

How it maps to the APIs:

- `forecast_scenarios[].request_body`
  Send to `POST /forecast/covers`, `POST /plan/staff`, `POST /plan/ingredients`, or `POST /plan/full`
- `feedback_examples[].request_body`
  Send to `POST /feedback`

## Future Enhancements

The current system accepts several planning and feedback context fields directly in the API input. A practical next step would be to reduce that manual input burden and fetch more of the context internally from connected data sources.

Examples:

- weather can be pulled from an external weather provider instead of always being passed manually
- holiday and event calendars can be maintained centrally and joined automatically
- recent restaurant history such as previous hour demand, previous day same hour demand, and short-term trends can be fetched internally from stored operational data
- promotion schedules can be read from a campaign table instead of being manually supplied in requests

This would make the API easier to use, reduce user-side errors, and make the production system more realistic.

## How To Start

This project uses `uv` and keeps its environment in `.venv`.

Recommended setup:

```bash
uv sync
```

If you want to activate the environment manually:

```bash
source .venv/bin/activate
```

You can also run commands through `uv`:

```bash
uv run python main.py
```

Then open:

```bash
http://localhost:8000/
```
