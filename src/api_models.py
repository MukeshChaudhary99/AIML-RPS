from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


class HourContextInput(BaseModel):
    timestamp: datetime
    reservations: Optional[float] = None
    holiday_flag: int = 0
    event_flag: int = 0
    holiday_name: Optional[str] = None
    event_name: Optional[str] = None
    temp_c: Optional[float] = None
    rain_mm: Optional[float] = None
    is_weekend: Optional[int] = None
    season: Optional[str] = None
    promotion_flag: int = 0


class ForecastRequest(BaseModel):
    hours: list[HourContextInput] = Field(min_length=1)
    auto_retrain: bool = True
    model_version: Optional[str] = None


class StaffPlanRequest(BaseModel):
    hours: list[HourContextInput] = Field(min_length=1)
    auto_retrain: bool = True


class IngredientPlanRequest(BaseModel):
    hours: list[HourContextInput] = Field(min_length=1)
    auto_retrain: bool = True


class FullPlanRequest(BaseModel):
    hours: list[HourContextInput] = Field(min_length=1)
    auto_retrain: bool = True


class FeedbackEntryInput(BaseModel):
    timestamp: datetime
    actual_covers: float
    actual_reservations: Optional[float] = None
    actual_walk_ins: Optional[float] = None
    predicted_covers: Optional[float] = None
    holiday_flag: Optional[int] = None
    event_flag: Optional[int] = None
    promotion_flag: Optional[int] = None
    rain_mm: Optional[float] = None
    temp_c: Optional[float] = None
    is_weekend: Optional[int] = None
    season: Optional[str] = None
    manager_note: Optional[str] = None


class FeedbackRequest(BaseModel):
    entries: list[FeedbackEntryInput] = Field(min_length=1)
    retrain: bool = True
