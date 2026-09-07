from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from pydantic import BaseModel
from pydantic import Field


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class CounterSnapshot(BaseModel):
    total: int = Field(default=0, ge=0)
    success: int = Field(default=0, ge=0)
    failed: int = Field(default=0, ge=0)


class RunLogEvent(BaseModel):
    event_at: str = Field(min_length=1)
    run_id: str = Field(min_length=1)
    region: str = Field(min_length=1)
    phase: str = Field(min_length=1)
    stage: str = Field(min_length=1)
    status: str = Field(min_length=1)
    metrics: dict[str, CounterSnapshot]
    detail: dict[str, object] = Field(default_factory=dict)


class RunLogStore:
    def __init__(
        self,
        log_root: Path,
        region: str,
        phase: str,
        run_id: str | None,
    ):
        self.run_id = run_id or f"{datetime.now().strftime('%Y%m%d%H%M%S')}_{uuid4().hex[:8]}"
        self._region = region
        self._phase = phase
        self._path = log_root / f"{self.run_id}_{region}_{phase}.jsonl"
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._metrics: dict[str, CounterSnapshot] = {
            "list_page": CounterSnapshot(),
            "prop": CounterSnapshot(),
            "community": CounterSnapshot(),
            "checkpoint": CounterSnapshot(),
        }

    @property
    def path(self) -> Path:
        return self._path

    def inc(
        self,
        metric: str,
        total: int,
        success: int,
        failed: int,
    ) -> None:
        if metric not in self._metrics:
            self._metrics[metric] = CounterSnapshot()
        current = self._metrics[metric]
        self._metrics[metric] = CounterSnapshot(
            total=current.total + total,
            success=current.success + success,
            failed=current.failed + failed,
        )

    def event(self, stage: str, status: str, detail: dict[str, object] | None) -> None:
        payload = RunLogEvent(
            event_at=_now_iso(),
            run_id=self.run_id,
            region=self._region,
            phase=self._phase,
            stage=stage,
            status=status,
            metrics=self._metrics,
            detail=detail or {},
        )
        with self._path.open("a", encoding="utf-8") as output_file:
            output_file.write(json.dumps(payload.model_dump(mode="json"), ensure_ascii=False))
            output_file.write("\n")
