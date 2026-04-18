from __future__ import annotations

from contextlib import asynccontextmanager

import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse

from src.api_models import (
    FeedbackRequest,
    ForecastRequest,
    FullPlanRequest,
    IngredientPlanRequest,
    StaffPlanRequest,
)
from src.config import AppConfig, load_app_config
from src.service import RestaurantPlanningService
from src.utils import dataframe_to_records, get_logger


def _records_to_dataframe(records: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(records)


def create_app(app_config: AppConfig | None = None) -> FastAPI:
    config = app_config or load_app_config()
    logger = get_logger("rps.api", config.api.log_level)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        logger.info("Starting Restaurant Resource Planning API")
        app.state.config = config
        app.state.logger = logger
        app.state.service = RestaurantPlanningService(config)
        app.state.startup_model_status = None
        try:
            app.state.startup_model_status = app.state.service.load_model_on_startup()
            logger.info(
                "Model loaded on startup: metrics=%s loaded=%s training_rows=%s feature_count=%s",
                app.state.startup_model_status.get("metrics"),
                app.state.startup_model_status.get("loaded"),
                app.state.startup_model_status.get("training_rows"),
                app.state.startup_model_status.get("feature_count"),
            )
        except Exception:
            logger.exception("Model load failed during startup")
        yield
        logger.info("Stopping Restaurant Resource Planning API")

    app = FastAPI(
        title=config.api.title,
        version=config.api.version,
        lifespan=lifespan,
    )

    @app.get("/", response_class=HTMLResponse)
    def dashboard(request: Request):
        service: RestaurantPlanningService = request.app.state.service
        startup_status = request.app.state.startup_model_status
        return service.render_dashboard_html(startup_status)

    @app.get("/health")
    def health(request: Request):
        service: RestaurantPlanningService = request.app.state.service
        model_status = service.get_model_status()
        return {
            "status": "ok",
            "data_dir": str(service.config.data.data_dir),
            "feedback_dir": str(service.config.data.feedback_dir),
            "model_status": model_status,
            "startup_model_status": request.app.state.startup_model_status,
        }

    @app.get("/config")
    def get_config(request: Request):
        config: AppConfig = request.app.state.config
        return {
            "data_dir": str(config.data.data_dir),
            "feedback_dir": str(config.data.feedback_dir),
            "model_version": config.forecast.model_version,
            "auto_retrain": config.forecast.auto_retrain,
            "api_title": config.api.title,
            "api_version": config.api.version,
        }

    @app.post("/forecast/covers")
    def forecast_covers(request_body: ForecastRequest, request: Request):
        try:
            service: RestaurantPlanningService = request.app.state.service
            hours_df = _records_to_dataframe(
                [row.model_dump() for row in request_body.hours]
            )
            result_df = service.forecast_covers(
                hours_df,
                auto_retrain=request_body.auto_retrain,
                model_version=request_body.model_version,
            )
            return {
                "count": len(result_df),
                "predictions": dataframe_to_records(result_df),
            }
        except Exception as exc:
            request.app.state.logger.exception("forecast_covers failed")
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/plan/staff")
    def plan_staff(request_body: StaffPlanRequest, request: Request):
        try:
            service: RestaurantPlanningService = request.app.state.service
            hours_df = _records_to_dataframe(
                [row.model_dump() for row in request_body.hours]
            )
            result = service.plan_staff(
                hours_df,
                auto_retrain=request_body.auto_retrain,
            )
            return {
                "count": len(result["hourly_staff_plan"]),
                "covers_forecast": dataframe_to_records(result["covers_forecast"]),
                "station_workload": dataframe_to_records(result["station_workload"]),
                "hourly_staff_plan": dataframe_to_records(result["hourly_staff_plan"]),
                "shift_schedule": dataframe_to_records(result["shift_schedule"]),
            }
        except Exception as exc:
            request.app.state.logger.exception("plan_staff failed")
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/plan/ingredients")
    def plan_ingredients(request_body: IngredientPlanRequest, request: Request):
        try:
            service: RestaurantPlanningService = request.app.state.service
            hours_df = _records_to_dataframe(
                [row.model_dump() for row in request_body.hours]
            )
            result = service.plan_ingredients(
                hours_df,
                auto_retrain=request_body.auto_retrain,
            )
            return {
                "count": len(result["purchase_recommendation"]),
                "covers_forecast": dataframe_to_records(result["covers_forecast"]),
                "predicted_menu_demand": dataframe_to_records(
                    result["predicted_menu_demand"]
                ),
                "daily_ingredient_demand": dataframe_to_records(
                    result["daily_ingredient_demand"]
                ),
                "purchase_recommendation": dataframe_to_records(
                    result["purchase_recommendation"]
                ),
            }
        except Exception as exc:
            request.app.state.logger.exception("plan_ingredients failed")
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/plan/full")
    def plan_full(request_body: FullPlanRequest, request: Request):
        try:
            service: RestaurantPlanningService = request.app.state.service
            hours_df = _records_to_dataframe(
                [row.model_dump() for row in request_body.hours]
            )
            result = service.plan_full(
                hours_df,
                auto_retrain=request_body.auto_retrain,
            )
            return {
                "covers_forecast": dataframe_to_records(result["covers_forecast"]),
                "station_workload": dataframe_to_records(result["station_workload"]),
                "hourly_staff_plan": dataframe_to_records(result["hourly_staff_plan"]),
                "shift_schedule": dataframe_to_records(result["shift_schedule"]),
                "predicted_menu_demand": dataframe_to_records(
                    result["predicted_menu_demand"]
                ),
                "daily_ingredient_demand": dataframe_to_records(
                    result["daily_ingredient_demand"]
                ),
                "purchase_recommendation": dataframe_to_records(
                    result["purchase_recommendation"]
                ),
            }
        except Exception as exc:
            request.app.state.logger.exception("plan_full failed")
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/feedback")
    def submit_feedback(request_body: FeedbackRequest, request: Request):
        try:
            service: RestaurantPlanningService = request.app.state.service
            feedback_df = _records_to_dataframe(
                [row.model_dump() for row in request_body.entries]
            )
            result = service.submit_feedback(
                feedback_df,
                retrain=request_body.retrain,
            )
            return {
                "summary": result["summary"],
                "scenario_corrections": dataframe_to_records(
                    result["scenario_corrections"]
                ),
                "feedback_frame": dataframe_to_records(result["feedback_frame"]),
            }
        except Exception as exc:
            request.app.state.logger.exception("submit_feedback failed")
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/feedback/scenario-corrections")
    def get_scenario_corrections(request: Request):
        try:
            service: RestaurantPlanningService = request.app.state.service
            result_df = service.get_scenario_corrections()
            return {
                "count": len(result_df),
                "scenario_corrections": dataframe_to_records(result_df),
            }
        except Exception as exc:
            request.app.state.logger.exception("get_scenario_corrections failed")
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/feedback/summary")
    def get_feedback_summary(request: Request):
        try:
            service: RestaurantPlanningService = request.app.state.service
            return service.get_feedback_summary()
        except Exception as exc:
            request.app.state.logger.exception("get_feedback_summary failed")
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    return app


app = create_app()
