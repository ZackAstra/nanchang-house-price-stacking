from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path

from playwright.sync_api import sync_playwright
from pydantic import BaseModel
from pydantic import Field

from src.auth import AuthManager
from src.auth import LoginOptions
from src.parsers.comm_parser import parse_community_page
from src.parsers.poi_parser import extract_poi_details
from src.parsers.poi_parser import extract_poi_summary
from src.parsers.prop_parser import parse_prop_page
from src.scheduler import M0RunConfig
from src.scheduler import _build_poi_summary_from_details
from src.scheduler import _collect_poi_details_with_ui
from src.scheduler import _goto_and_get_html
from src.scheduler import _normalize_community_url
from src.storage.failure_store import FailureStore
from src.storage.failure_store import load_retry_targets
from src.storage.jsonl_storage import append_jsonl
from src.storage.run_log_store import RunLogStore


class RetryRunConfig(BaseModel):
    region_slug: str = Field(min_length=1)
    data_dir: Path
    retry_file: Path
    retry_type: str = Field(default="auto")
    retry_limit: int | None = Field(default=None, ge=1)
    wait_ms: int = Field(default=2500, ge=500, le=20000)
    visible: bool = Field(default=True)
    manual_login: bool = Field(default=True)
    manual_login_timeout_sec: int = Field(default=180, ge=30, le=1800)
    save_html_on_error: bool = Field(default=True)
    error_html_dir: Path = Field(default=Path("data/debug/error_html"))
    auto_login: bool = Field(default=False)
    username: str = Field(default=os.environ.get("ANJUKE_USERNAME", ""))
    password: str = Field(default=os.environ.get("ANJUKE_PASSWORD", ""))
    run_id: str | None = Field(default=None)


def _load_existing_ids(output_file: Path, id_key: str) -> set[str]:
    if not output_file.exists():
        return set()
    ids: set[str] = set()
    with output_file.open("r", encoding="utf-8") as input_file:
        for raw_line in input_file:
            line = raw_line.strip()
            if line == "":
                continue
            parsed = json.loads(line)
            if not isinstance(parsed, dict):
                continue
            value = parsed.get(id_key)
            if value is None:
                continue
            value_text = str(value).strip()
            if value_text != "":
                ids.add(value_text)
    return ids


def run_retry_failures(config: RetryRunConfig, region_map: dict[str, str]) -> dict[str, int]:
    if config.region_slug not in region_map:
        raise ValueError(f"区域不存在于config/regions.json: {config.region_slug}")

    retry_targets = load_retry_targets(
        retry_file=config.retry_file,
        retry_type=config.retry_type,
        retry_limit=config.retry_limit,
    )
    if len(retry_targets) == 0:
        return {
            "total": 0,
            "success": 0,
            "failed": 0,
            "house_added": 0,
            "community_added": 0,
        }

    output_dir = config.data_dir / "output"
    houses_output = output_dir / f"{config.region_slug}_houses.jsonl"
    communities_output = output_dir / f"{config.region_slug}_communities.jsonl"
    existing_house_ids = _load_existing_ids(houses_output, "house_id")
    existing_community_ids = _load_existing_ids(communities_output, "community_id")

    run_log = RunLogStore(
        log_root=Path("logs") / "runs",
        region=config.region_slug,
        phase="retry-failures",
        run_id=config.run_id,
    )
    failure_store = FailureStore(config.data_dir, config.region_slug)
    run_log.event(
        stage="retry_start",
        status="started",
        detail={
            "retry_file": str(config.retry_file),
            "retry_type": config.retry_type,
            "retry_targets": len(retry_targets),
        },
    )

    m0_config = M0RunConfig(
        region_slug=config.region_slug,
        pages=1,
        max_props=1,
        wait_ms=config.wait_ms,
        visible=config.visible,
        manual_login=config.manual_login,
        manual_login_timeout_sec=config.manual_login_timeout_sec,
        save_html_on_error=config.save_html_on_error,
        error_html_dir=config.error_html_dir,
        output_path=config.data_dir / "temp_retry.jsonl",
        auto_login=config.auto_login,
        username=config.username,
        password=config.password,
        run_id=run_log.run_id,
    )

    total = len(retry_targets)
    success = 0
    failed = 0
    house_added = 0
    community_added = 0

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=not config.visible)
        context = browser.new_context(locale="zh-CN")
        page = context.new_page()
        if config.auto_login:
            login_options = LoginOptions(
                username=config.username,
                password=config.password,
                timeout_sec=config.manual_login_timeout_sec,
                visible=config.visible,
            )
            login_result = AuthManager.perform_login(page, login_options)
            if not login_result.success:
                raise RuntimeError(f"失败重试前自动登录失败: {login_result.state.error_message}")
        for target in retry_targets:
            if target.target_type == "prop":
                run_log.inc("prop", total=1, success=0, failed=0)
                try:
                    html = _goto_and_get_html(
                        page=page,
                        url=target.target_url,
                        config=m0_config,
                        page_no=target.page_no,
                        stage="retry_prop",
                    )
                    house = parse_prop_page(target.target_url, html, config.region_slug)
                    house_data = house.model_dump(mode="json")
                    house_data["list_page_no"] = target.page_no
                    house_data["extracted_at"] = datetime.now().isoformat()
                    house_id_text = str(house_data.get("house_id"))
                    if house_id_text not in existing_house_ids:
                        append_jsonl([house_data], houses_output)
                        existing_house_ids.add(house_id_text)
                        house_added += 1
                    success += 1
                    run_log.inc("prop", total=0, success=1, failed=0)
                    FailureStore.append_retry_update(
                        failure_file=config.retry_file,
                        failure_id=target.failure_id,
                        retry_count=target.retry_count + 1,
                        success=True,
                        error_type=None,
                        error_message=None,
                    )
                except Exception as error:
                    failed += 1
                    run_log.inc("prop", total=0, success=0, failed=1)
                    FailureStore.append_retry_update(
                        failure_file=config.retry_file,
                        failure_id=target.failure_id,
                        retry_count=target.retry_count + 1,
                        success=False,
                        error_type="retry_prop_failed",
                        error_message=str(error),
                    )
                    failure_store.append_prop_failure(
                        run_id=run_log.run_id,
                        phase="retry-failures",
                        page_no=target.page_no,
                        target_url=target.target_url,
                        house_id=target.house_id,
                        error_type="retry_prop_failed",
                        error_message=str(error),
                        community_id=target.community_id,
                    )
                continue

            run_log.inc("community", total=1, success=0, failed=0)
            try:
                normalized_url = _normalize_community_url(target.target_url)
                html = _goto_and_get_html(
                    page=page,
                    url=normalized_url,
                    config=m0_config,
                    page_no=target.page_no,
                    stage="retry_community",
                )
                tab_item_count = page.locator("div.tabBox li").count()
                poi_details = _collect_poi_details_with_ui(page)
                poi_summary = _build_poi_summary_from_details(poi_details)
                if sum(poi_summary.values()) == 0:
                    poi_details = extract_poi_details(html)
                    poi_summary = extract_poi_summary(html)
                community = parse_community_page(
                    normalized_url,
                    html,
                    tab_item_count,
                    poi_summary,
                    poi_details,
                )
                community_data = community.model_dump(mode="json")
                community_data["extracted_at"] = datetime.now().isoformat()
                community_id_text = str(community_data.get("community_id"))
                if community_id_text not in existing_community_ids:
                    append_jsonl([community_data], communities_output)
                    existing_community_ids.add(community_id_text)
                    community_added += 1
                success += 1
                run_log.inc("community", total=0, success=1, failed=0)
                FailureStore.append_retry_update(
                    failure_file=config.retry_file,
                    failure_id=target.failure_id,
                    retry_count=target.retry_count + 1,
                    success=True,
                    error_type=None,
                    error_message=None,
                )
            except Exception as error:
                failed += 1
                run_log.inc("community", total=0, success=0, failed=1)
                FailureStore.append_retry_update(
                    failure_file=config.retry_file,
                    failure_id=target.failure_id,
                    retry_count=target.retry_count + 1,
                    success=False,
                    error_type="retry_community_failed",
                    error_message=str(error),
                )
                failure_store.append_community_failure(
                    run_id=run_log.run_id,
                    phase="retry-failures",
                    page_no=target.page_no,
                    target_url=target.target_url,
                    house_id=target.house_id,
                    community_id=target.community_id,
                    error_type="retry_community_failed",
                    error_message=str(error),
                )

        context.close()
        browser.close()

    run_log.event(
        stage="retry_finish",
        status="completed",
        detail={
            "total": total,
            "success": success,
            "failed": failed,
            "house_added": house_added,
            "community_added": community_added,
        },
    )
    return {
        "total": total,
        "success": success,
        "failed": failed,
        "house_added": house_added,
        "community_added": community_added,
    }
