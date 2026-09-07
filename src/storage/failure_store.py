from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from pydantic import BaseModel
from pydantic import Field


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class FailureRecord(BaseModel):
    record_type: str = Field(default="failure")
    failure_id: str = Field(min_length=1)
    run_id: str | None = Field(default=None)
    region: str = Field(min_length=1)
    phase: str = Field(min_length=1)
    target_type: str = Field(min_length=1, description="prop/community")
    target_url: str = Field(min_length=1)
    page_no: int = Field(ge=0)
    house_id: str | None = Field(default=None)
    community_id: str | None = Field(default=None)
    error_type: str = Field(min_length=1)
    error_message: str = Field(min_length=1)
    failed_at: str = Field(min_length=1)
    retry_count: int = Field(default=0, ge=0)
    resolved: bool = Field(default=False)
    resolved_at: str | None = Field(default=None)


class RetryUpdateRecord(BaseModel):
    record_type: str = Field(default="retry_update")
    failure_id: str = Field(min_length=1)
    retried_at: str = Field(min_length=1)
    success: bool
    retry_count: int = Field(ge=0)
    error_type: str | None = Field(default=None)
    error_message: str | None = Field(default=None)
    resolved: bool
    resolved_at: str | None = Field(default=None)


class FailureStore:
    def __init__(self, data_dir: Path, region: str):
        self._region = region
        self._base_dir = data_dir / "failures"
        self._base_dir.mkdir(parents=True, exist_ok=True)
        self.prop_failed_file = self._base_dir / f"{region}_prop_failed.jsonl"
        self.community_failed_file = self._base_dir / f"{region}_community_failed.jsonl"

    def _append(self, path: Path, payload: dict[str, object]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as output_file:
            output_file.write(json.dumps(payload, ensure_ascii=False))
            output_file.write("\n")

    def append_prop_failure(
        self,
        run_id: str | None,
        phase: str,
        page_no: int,
        target_url: str,
        house_id: str | None,
        error_type: str,
        error_message: str,
        community_id: str | None,
    ) -> str:
        failure_id = uuid4().hex
        record = FailureRecord(
            failure_id=failure_id,
            run_id=run_id,
            region=self._region,
            phase=phase,
            target_type="prop",
            target_url=target_url,
            page_no=page_no,
            house_id=house_id,
            community_id=community_id,
            error_type=error_type,
            error_message=error_message,
            failed_at=_now_iso(),
        )
        self._append(self.prop_failed_file, record.model_dump())
        return failure_id

    def append_community_failure(
        self,
        run_id: str | None,
        phase: str,
        page_no: int,
        target_url: str,
        house_id: str | None,
        community_id: str | None,
        error_type: str,
        error_message: str,
    ) -> str:
        failure_id = uuid4().hex
        record = FailureRecord(
            failure_id=failure_id,
            run_id=run_id,
            region=self._region,
            phase=phase,
            target_type="community",
            target_url=target_url,
            page_no=page_no,
            house_id=house_id,
            community_id=community_id,
            error_type=error_type,
            error_message=error_message,
            failed_at=_now_iso(),
        )
        self._append(self.community_failed_file, record.model_dump())
        return failure_id

    @staticmethod
    def append_retry_update(
        failure_file: Path,
        failure_id: str,
        retry_count: int,
        success: bool,
        error_type: str | None,
        error_message: str | None,
    ) -> None:
        resolved_at = _now_iso() if success else None
        update = RetryUpdateRecord(
            failure_id=failure_id,
            retried_at=_now_iso(),
            success=success,
            retry_count=retry_count,
            error_type=error_type,
            error_message=error_message,
            resolved=success,
            resolved_at=resolved_at,
        )
        failure_file.parent.mkdir(parents=True, exist_ok=True)
        with failure_file.open("a", encoding="utf-8") as output_file:
            output_file.write(json.dumps(update.model_dump(), ensure_ascii=False))
            output_file.write("\n")


def load_retry_targets(
    retry_file: Path,
    retry_type: str,
    retry_limit: int | None,
) -> list[FailureRecord]:
    if not retry_file.exists():
        raise FileNotFoundError(f"失败清单文件不存在: {retry_file}")

    latest_records: dict[str, FailureRecord] = {}
    retry_updates: dict[str, RetryUpdateRecord] = {}
    with retry_file.open("r", encoding="utf-8") as input_file:
        for index, raw_line in enumerate(input_file, start=1):
            line = raw_line.strip()
            if line == "":
                continue
            parsed = json.loads(line)
            if not isinstance(parsed, dict):
                continue
            record_type_raw = parsed.get("record_type")
            record_type = str(record_type_raw) if record_type_raw is not None else "failure"
            if record_type == "failure":
                payload = dict(parsed)
                failure_id_raw = payload.get("failure_id")
                failure_id = str(failure_id_raw) if failure_id_raw is not None else f"legacy_{index}"
                payload["failure_id"] = failure_id
                if payload.get("failed_at") is None:
                    payload["failed_at"] = _now_iso()
                if payload.get("target_type") is None:
                    payload["target_type"] = "prop"
                if payload.get("retry_count") is None:
                    payload["retry_count"] = 0
                if payload.get("resolved") is None:
                    payload["resolved"] = False
                latest_records[failure_id] = FailureRecord.model_validate(payload)
                continue
            if record_type == "retry_update":
                update = RetryUpdateRecord.model_validate(parsed)
                retry_updates[update.failure_id] = update

    unresolved: list[FailureRecord] = []
    for failure_id, record in latest_records.items():
        update = retry_updates.get(failure_id)
        if update is not None and update.resolved:
            continue
        if retry_type != "auto" and record.target_type != retry_type:
            continue
        unresolved.append(record)

    if retry_limit is None:
        return unresolved
    return unresolved[:retry_limit]
