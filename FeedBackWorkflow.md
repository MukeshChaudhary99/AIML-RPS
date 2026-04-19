# Feedback Workflow

This document explains how feedback moves through the Restaurant Resource Planning System, what happens after a feedback request is submitted, and how that feedback affects future predictions.

## Purpose

The system does not treat forecasting as a one-time prediction task. It keeps comparing predictions with real outcomes, measures where it was wrong, and uses those mistakes to improve future forecasts.

In this project, feedback is mainly used for the covers forecasting system. Once covers improve, the staff and ingredient planning modules also improve because they depend on the covers forecast.

## What A Feedback Request Contains

A feedback request is sent to the `POST /feedback` API.

Each feedback row is tied to one hourly timestamp and can contain:

- `timestamp`
- `actual_covers`
- `predicted_covers` if available
- `actual_reservations`
- `actual_walk_ins`
- `holiday_flag`
- `event_flag`
- `promotion_flag`
- `rain_mm`
- `temp_c`
- `is_weekend`
- `season`
- `manager_note`

Important meaning:

- `predicted_covers` is what the system forecast earlier
- `actual_covers` is the final truth value the system should learn from


## Step By Step Flow

### Step 1: Feedback Is Submitted

The manager or test client sends one or more hourly rows to the feedback API.

Example:

```json
{
  "entries": [
    {
      "timestamp": "2026-07-15T12:00:00",
      "predicted_covers": 120,
      "actual_covers": 85,
      "rain_mm": 18.0,
      "temp_c": 24.0,
      "is_weekend": 0,
      "season": "monsoon",
      "manager_note": "Heavy rain reduced lunch walk-ins."
    }
  ],
  "retrain": true
}
```

### Step 2: Actuals Are Stored

The system stores the actual outcome in the actuals log.

This includes:

- `timestamp`
- `actual_covers`
- `actual_reservations`
- `actual_walk_ins`

This is the factual outcome table used for later comparison.

### Step 3: Feedback Context Is Stored

The system also stores the optional context that came with the feedback row.

This may include:

- `predicted_covers`
- weather
- event flags
- promotion flag
- weekend indicator
- season
- manager note

This gives the system enough information to understand what kind of scenario the error came from.

### Step 4: Prediction Log Is Checked

The system normally already has the original prediction stored, because every forecast request is logged when the prediction API is called.

If `predicted_covers` is included in the feedback payload, the system can also backfill the prediction log from the feedback row itself.

This is useful when:

- the original forecast was made earlier
- the prediction log is missing
- feedback is being tested manually
- another service produced the prediction

### Step 5: Feedback Frame Is Built

The system builds a combined table by joining:

- prediction log
- actuals log
- feedback context log

This combined table is the main feedback analysis table.

For each timestamp, it calculates:

- predicted value
- actual value
- error
- absolute error
- percentage error
- actual-to-predicted ratio

This is the table used for correction logic and retraining.

### Step 6: Error Is Measured

For each row:

- `error = predicted_covers - actual_covers`
- `abs_error = absolute value of error`

Interpretation:

- positive error means the model predicted too high
- negative error means the model predicted too low

Example:

- predicted `120`
- actual `85`
- error = `35`

This means the model overpredicted by 35 covers.

### Step 7: Scenario Rules Are Applied

The system checks whether the row belongs to common operating scenarios such as:

- `weekend`
- `holiday`
- `event`
- `promotion`
- `heavy_rain`
- `hot_weather`
- `breakfast`
- `lunch`
- `afternoon`
- `dinner`

The scenario is determined mainly from structured fields like:

- `event_flag`
- `promotion_flag`
- `rain_mm`
- `temp_c`
- `is_weekend`
- `season`


### Step 8: Scenario Correction Factors Are Recomputed

The system groups past feedback rows by scenario and checks whether there is a repeated bias.

Example:

- many rainy lunch rows show that actual covers are lower than predicted

From that, the system computes a correction factor such as:

- `0.92` meaning reduce future matching predictions by about 8 percent
- `1.08` meaning increase future matching predictions by about 8 percent

This is the short-term learning mechanism.

### Step 9: Threshold Is Checked

A scenario is not used immediately after one single feedback row.

The system waits until it has enough evidence.

Current logic:

- a scenario becomes actionable only after the minimum sample threshold is reached

In this project, the main threshold is controlled by configuration, and the current default is:

- `correction_min_samples = 8`

That means:

- feedback is always stored
- scenario metrics are always updated
- but prediction adjustment is applied only when enough similar examples exist

### Step 10: Retraining Decision Is Checked

After feedback is stored, the system checks whether retraining should happen.

Retraining does not happen blindly on every row.

The current logic checks recent error. If recent forecasting performance is poor enough, retraining is triggered.

This is controlled by:

- `retrain_error_threshold`

So retraining is also threshold-based.

### Step 11: If Retraining Happens, Training History Is Updated

When retraining starts, the system rebuilds the training dataset.

It takes:

- historical base data
- feedback rows with actual outcomes

Then it replaces the target covers for those feedback timestamps with the submitted `actual_covers`.

This is the long-term learning mechanism.

It means the next XGBoost model is trained on corrected truth, not only on old historical data.

### Step 12: Future Predictions Use Feedback

Future predictions can use feedback in two ways.

First way:

- the raw model prediction is adjusted using the scenario correction table

Second way:

- after retraining, the model itself has learned from the updated history

So feedback affects future forecasting both before and after retraining.

## What Happens Immediately After One Feedback Row

If you submit only one row:

- it is stored
- it contributes to error tracking
- it may not yet change future predictions much

This is because one row is usually not enough to make a scenario actionable.

So the immediate effect is limited, but the row is still useful because it builds evidence.

## When Feedback Becomes Useful For Future Predictions

Feedback becomes useful in different ways:

- immediately for storage and monitoring
- after enough similar rows for scenario correction
- after retraining for model learning

So the answer is:

- feedback is always remembered
- feedback is not always applied immediately
- stronger influence happens after enough support or retraining

## Meaning Of Feedback Outputs

When feedback is submitted, the API returns a summary and related tables.

### `summary`

This is the main high-level output.

Important fields:

- `feedback_rows`
  Total number of rows now available in the feedback frame.

- `mean_abs_error`
  Average absolute forecasting error across feedback rows.
  Lower is better.

- `mean_error_pct`
  Average signed error percentage relative to the prediction.
  Positive usually means the system tends to overpredict.
  Negative usually means the system tends to underpredict.

- `actionable_scenarios`
  Number of scenarios that now have enough support to be used for live correction.

- `retrained`
  Indicates whether model retraining was triggered during this feedback request.

If retraining happens, the response may also include model metrics such as:

- `mae`
- `rmse`
- `wape`

These represent model quality after retraining.

### `scenario_corrections`

This table shows the learned correction rule for each scenario.

Important fields:

- `scenario`
  Name of the scenario, such as `heavy_rain` or `event`

- `observations`
  Number of feedback rows supporting that scenario

- `support_weight`
  Confidence weight derived from the amount of evidence

- `mean_error`
  Average signed error in that scenario

- `median_actual_to_pred_ratio`
  How actual covers compare to predicted covers

- `correction_factor`
  Multiplier applied to future matching predictions

- `recommended_adjustment_pct`
  Same correction expressed as a percentage

- `is_actionable`
  Whether the scenario has enough support to be actively used

Interpretation:

- `correction_factor < 1.0` means future predictions will be reduced
- `correction_factor > 1.0` means future predictions will be increased

### `feedback_frame`

This is the combined detailed table built from:

- predictions
- actuals
- feedback context

It is the internal evidence table used for analysis, correction, and retraining.

It is useful when you want to inspect exactly how the system matched one prediction with one actual outcome.

## Small Example

Suppose the system predicted:

- `2026-07-15 12:00` -> `120 covers`

Then feedback says:

- actual = `85`
- rain was heavy
- season = monsoon

The system will:

1. store the actual
2. store the scenario context
3. calculate that the model overpredicted by 35
4. add this row to rainy lunch evidence
5. later reduce future rainy lunch forecasts if enough similar rows exist
6. eventually retrain the model using this corrected truth

