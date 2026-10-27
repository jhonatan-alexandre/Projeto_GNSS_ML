"""Frozen scientific and temporal contract; timestamps are UTC without a zone."""
from dataclasses import asdict, dataclass, fields
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class Protocol:
    time_resolution: str = "15min"
    forecast_minutes: int = 15
    forecast_horizon_steps: int = 1
    study_start: str = "2023-12-01 00:00:00"
    train_end: str = "2024-02-16 23:59:59"
    context_start: str = "2024-02-17 00:00:00"
    context_end: str = "2024-02-17 23:59:59"
    test_start: str = "2024-02-18 00:00:00"
    test_end: str = "2024-02-18 23:59:59"
    fit_end: str = "2024-02-02 23:59:59"
    early_stopping_start: str = "2024-02-03 00:00:00"
    early_stopping_end: str = "2024-02-09 23:59:59"
    selection_start: str = "2024-02-10 00:00:00"
    selection_end: str = "2024-02-16 23:59:59"
    piecewise_look_back_days: int = 15
    s4_threshold: float = 0.5
    temporal_gap: str = "24h"
    retrospective_interpolation: bool = True

    def __post_init__(self):
        if (pd.Timedelta(self.time_resolution) != pd.Timedelta("15min")
                or self.forecast_minutes != 15 or self.forecast_horizon_steps != 1):
            raise ValueError("This protocol requires 15min resolution and a one-step 15min horizon.")
        dates = [pd.Timestamp(getattr(self, k)) for k in (
            "study_start", "fit_end", "early_stopping_start", "early_stopping_end",
            "selection_start", "selection_end", "train_end", "context_start",
            "context_end", "test_start", "test_end")]
        if any(a > b for a, b in zip(dates, dates[1:])):
            raise ValueError("Invalid chronological periods.")
        if (self.piecewise_look_back_days != 15 or self.s4_threshold != .5
                or pd.Timedelta(self.temporal_gap) != pd.Timedelta("24h")
                or self.retrospective_interpolation is not True):
            raise ValueError("Unsupported scientific protocol.")

    @property
    def resolution_minutes(self):
        return int(pd.Timedelta(self.time_resolution).total_seconds()/60)

    @property
    def slot_count(self):
        return 1440 // self.resolution_minutes

    @property
    def rolling_window_steps(self):
        return 180 // self.resolution_minutes

    @property
    def daily_lag_steps(self):
        return self.slot_count

    @property
    def look_back(self):
        return self.slot_count

    @property
    def development_available_until(self):
        """All nominal development bins finish at the start of the context day."""
        return pd.Timestamp(self.train_end).floor(self.time_resolution) + pd.Timedelta(self.time_resolution)

    def partition(self, nominal_bin_start):
        t = pd.Timestamp(nominal_bin_start)
        if pd.Timestamp(self.study_start) <= t <= pd.Timestamp(self.train_end):
            return "development"
        if pd.Timestamp(self.context_start) <= t <= pd.Timestamp(self.context_end):
            return "context"
        if pd.Timestamp(self.test_start) <= t <= pd.Timestamp(self.test_end):
            return "test"
        return "outside"

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "resolution_minutes": self.resolution_minutes,
                "slot_count": self.slot_count, "rolling_window_steps": self.rolling_window_steps,
                "daily_lag_steps": self.daily_lag_steps, "look_back": self.look_back}

    @classmethod
    def from_dict(cls, data: dict[str, Any]):
        valid = {f.name for f in fields(cls)}
        derived = {"resolution_minutes", "slot_count", "rolling_window_steps", "daily_lag_steps", "look_back"}
        unknown = set(data) - valid - derived
        if unknown:
            raise ValueError("Unknown protocol fields: " + ", ".join(sorted(unknown)))
        result = cls(**{k: v for k, v in data.items() if k in valid})
        for k in derived & data.keys():
            if data[k] != getattr(result, k):
                raise ValueError(f"Protocol mismatch for {k}.")
        return result
