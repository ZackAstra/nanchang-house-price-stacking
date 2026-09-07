from __future__ import annotations

import json
import importlib
import os
import sys
import threading
from collections.abc import Callable
from concurrent.futures import Future
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from time import monotonic

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.exceptions import NetworkError


DATE_TAG: str = "20260326"
MAX_PROPS_PER_PAGE: int = 60
MAX_WORKERS_LIMIT: int = 3
MIN_SUCCESS_RATE: float = 0.9
DEFAULT_WAIT_MS: int = 2500
MAX_REGION_ATTEMPTS: int = 3
PROFILE_RUN_TAG: str = os.getenv("M2_PROFILE_RUN_TAG", str(int(monotonic() * 1000)))
ALL_REGIONS: tuple[str, ...] = (
    "honggutan",
    "qingshanhu",
    "donghu",
    "xihu",
    "qingyunpu",
    "xinjian",
    "nanchangxian",
    "jinxian",
    "anyi",
    "gaoxinkaifaqu",
    "changbeijingjikaifaqu",
    "wanli",
    "xiaolanjingjikaifaqu",
    "xianghu",
)
APPEND_JSONL_PATCH_LOCK = threading.Lock()


@dataclass(frozen=True)
class RegionRunSummary:
    region: str
    written_records: int
    prop_failed: int
    community_failed: int
    success_rate: float
    field_coverage_rate: float
    output_path: Path
    community_output_path: Path
    last_failed_prop_url: str | None


def _resolve_proxy_pool() -> list[str]:
    raw_proxy_list = os.getenv("M2_PROXY_LIST")
    if raw_proxy_list is None:
        return []
    proxies = [item.strip() for item in raw_proxy_list.split(",") if item.strip() != ""]
    return proxies


def _pick_sticky_proxy(region: str, proxy_pool: list[str]) -> str | None:
    if len(proxy_pool) == 0:
        return None
    index = abs(hash(region)) % len(proxy_pool)
    return proxy_pool[index]


def _pick_proxy_for_attempt(region: str, proxy_pool: list[str], attempt_no: int) -> str | None:
    if len(proxy_pool) == 0:
        return None
    base_index = abs(hash(region)) % len(proxy_pool)
    index = (base_index + attempt_no) % len(proxy_pool)
    return proxy_pool[index]


def _build_profile_dir(region: str) -> Path:
    profile_dir = Path("data") / "browser_profiles" / f"m2_{region}_{PROFILE_RUN_TAG}"
    profile_dir.mkdir(parents=True, exist_ok=True)
    return profile_dir


def _is_login_or_captcha_page(html: str) -> bool:
    return (
        "请输入验证码" in html
        or "/antibot/" in html
        or "geetest" in html
        or "微信登录" in html
        or "antispam-block" in html
    )


def _wait_for_manual_login(page_html_reader: Callable[[], str], page_waiter: Callable[[int], None], timeout_sec: int) -> None:
    deadline = monotonic() + timeout_sec
    while monotonic() < deadline:
        page_waiter(2000)
        if not _is_login_or_captcha_page(page_html_reader()):
            return
    raise NetworkError(f"人工验证超时, timeout_sec={timeout_sec}")


def _is_callback_antibot_url(url: str) -> bool:
    return "callback.58.com/antibot/verifycode" in url


def _auto_click_callback_verify(page: object) -> bool:
    rounds = 6
    for _ in range(rounds):
        clicked = False
        verify_button = page.locator("#btnSubmit")
        if verify_button.count() > 0:
            try:
                verify_button.first.click(timeout=5000)
                clicked = True
            except Exception:
                clicked = False
        if not clicked:
            text_button = page.locator("text=点击按钮进行验证")
            if text_button.count() > 0:
                try:
                    text_button.first.click(timeout=5000)
                    clicked = True
                except Exception:
                    clicked = False
        page.wait_for_timeout(1500)
        current_html = _safe_page_content(page)
        current_url = page.url
        if not _is_login_or_captcha_page(current_html) and "antispam-block" not in current_url:
            return True
        if not _is_callback_antibot_url(current_url) and not clicked:
            break
    return False


def _safe_page_content(page: object) -> str:
    for _ in range(6):
        try:
            return page.content()
        except Exception:
            page.wait_for_timeout(400)
    return page.content()


def _pick_or_create_active_page(context: object) -> object:
    for existing_page in context.pages:
        if existing_page.url != "about:blank":
            return existing_page
    return context.new_page()


def _auto_handle_antibot_transition(page: object, target_url: str) -> bool:
    for _ in range(6):
        current_url = page.url
        current_html = _safe_page_content(page)
        if _is_callback_antibot_url(current_url):
            callback_ok = _auto_click_callback_verify(page)
            if callback_ok:
                page.goto(target_url, wait_until="domcontentloaded", timeout=60000)
                page.wait_for_timeout(DEFAULT_WAIT_MS)
                continue
        if "antispam-block" in current_url:
            clicked = False
            verify_button = page.locator("#btnSubmit")
            if verify_button.count() > 0:
                try:
                    verify_button.first.click(timeout=5000)
                    clicked = True
                except Exception:
                    clicked = False
            if not clicked:
                text_button = page.locator("text=点击按钮进行验证")
                if text_button.count() > 0:
                    try:
                        text_button.first.click(timeout=5000)
                        clicked = True
                    except Exception:
                        clicked = False
            if clicked:
                page.wait_for_timeout(1500)
                continue
            page.goto(target_url, wait_until="domcontentloaded", timeout=60000)
            page.wait_for_timeout(DEFAULT_WAIT_MS)
            continue
        if not _is_login_or_captcha_page(current_html):
            return True
        page.wait_for_timeout(1200)
    final_html = _safe_page_content(page)
    final_url = page.url
    return not _is_login_or_captcha_page(final_html) and "antispam-block" not in final_url


def _http_probe_region(region_url: str, timeout_sec: int, proxy_server: str | None) -> tuple[bool, str]:
    try:
        import httpx  # type: ignore
    except Exception:
        return False, "httpx_unavailable"
    try:
        proxy_value = None
        if proxy_server is not None and proxy_server.strip() != "":
            proxy_value = proxy_server.strip()
        with httpx.Client(timeout=timeout_sec, proxy=proxy_value) as client:
            response = client.get(region_url, follow_redirects=True)
            if response.status_code != 200:
                return False, f"http_status_{response.status_code}"
            if "/sale/" not in str(response.url):
                return False, f"unexpected_url_{response.url}"
            return True, "ok"
    except Exception as error:
        return False, f"http_probe_error:{error}"


def _run_playwright_diagnostic(region: str, target_url: str, timeout_sec: int, profile_dir: Path, proxy_server: str | None) -> None:
    from playwright.sync_api import sync_playwright

    screenshot_dir = Path("logs/debug_screenshots")
    html_dir = Path("data/debug/error_html")
    screenshot_dir.mkdir(parents=True, exist_ok=True)
    html_dir.mkdir(parents=True, exist_ok=True)
    screenshot_path = screenshot_dir / f"m2_{region}_{DATE_TAG}_diagnostic.png"
    html_path = html_dir / f"m2_{region}_{DATE_TAG}_diagnostic.html"

    with sync_playwright() as playwright:
        proxy_settings: dict[str, str] | None = None
        if proxy_server is not None and proxy_server.strip() != "":
            proxy_settings = {"server": proxy_server.strip()}
        context = playwright.chromium.launch_persistent_context(
            user_data_dir=str(profile_dir),
            headless=False,
            locale="zh-CN",
            proxy=proxy_settings,
        )
        page = _pick_or_create_active_page(context)
        page.goto(target_url, wait_until="domcontentloaded", timeout=timeout_sec * 1000)
        page.wait_for_timeout(DEFAULT_WAIT_MS)
        html = _safe_page_content(page)
        if _is_login_or_captcha_page(html):
            auto_verified = _auto_handle_antibot_transition(page, target_url)
            if not auto_verified:
                verify_button = page.locator("#btnSubmit")
                if verify_button.count() > 0:
                    try:
                        verify_button.first.click(timeout=5000)
                    except Exception:
                        pass
                _wait_for_manual_login(page.content, page.wait_for_timeout, timeout_sec)
            page.wait_for_timeout(DEFAULT_WAIT_MS)
            html = _safe_page_content(page)
            if _is_callback_antibot_url(page.url):
                page.goto(target_url, wait_until="domcontentloaded", timeout=timeout_sec * 1000)
                page.wait_for_timeout(DEFAULT_WAIT_MS)
                html = _safe_page_content(page)
        page.screenshot(path=str(screenshot_path), full_page=True)
        html_path.write_text(html, encoding="utf-8")
        context.close()


def _read_regions(region_option: str, region_map: dict[str, str]) -> list[str]:
    if region_option == "all":
        return [region for region in ALL_REGIONS if region in region_map]
    raw_regions = [item.strip() for item in region_option.split(",") if item.strip() != ""]
    invalid_regions = [region for region in raw_regions if region not in region_map]
    if len(invalid_regions) > 0:
        raise ValueError(f"无效区域参数: {invalid_regions}")
    return raw_regions


def _checkpoint_path() -> Path:
    return Path("data") / f"m2_checkpoint_{DATE_TAG}.json"


def _load_checkpoint() -> dict[str, dict[str, object]]:
    checkpoint_file = _checkpoint_path()
    if not checkpoint_file.exists():
        return {}
    raw = json.loads(checkpoint_file.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        return {}
    normalized: dict[str, dict[str, object]] = {}
    for key, value in raw.items():
        if isinstance(value, dict):
            normalized[key] = value
    return normalized


def _save_checkpoint(checkpoint: dict[str, dict[str, object]]) -> None:
    checkpoint_file = _checkpoint_path()
    checkpoint_file.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_file.write_text(json.dumps(checkpoint, ensure_ascii=False, indent=2), encoding="utf-8")


def _calc_success_rate(written_records: int, prop_failed: int) -> float:
    if written_records <= 0:
        return 0.0
    success_records = written_records - prop_failed
    if success_records < 0:
        return 0.0
    return success_records / written_records


def _extract_total_price(house: dict[str, object]) -> float | int | None:
    total_price = house.get("total_price")
    if isinstance(total_price, (int, float)):
        return total_price
    total_price_wan = house.get("total_price_wan")
    if isinstance(total_price_wan, (int, float)):
        return total_price_wan
    return None


def _calc_field_coverage_rate(output_path: Path) -> float:
    required_fields: tuple[str, str, str, str] = ("house_id", "region", "total_price", "area_sqm")
    total_required = 0
    hit_required = 0
    with output_path.open("r", encoding="utf-8") as input_file:
        for raw_line in input_file:
            line = raw_line.strip()
            if line == "":
                continue
            record = json.loads(line)
            if not isinstance(record, dict):
                continue
            region = record.get("region")
            house_raw = record.get("house")
            house = house_raw if isinstance(house_raw, dict) else {}
            flat: dict[str, object] = {
                "house_id": house.get("house_id"),
                "region": region,
                "total_price": _extract_total_price(house),
                "area_sqm": house.get("area_sqm"),
            }
            for field_name in required_fields:
                total_required += 1
                value = flat[field_name]
                if value is None:
                    continue
                if isinstance(value, str) and value.strip() == "":
                    continue
                hit_required += 1
    if total_required == 0:
        return 0.0
    return hit_required / total_required


def _find_last_failed_prop_url(output_path: Path) -> str | None:
    last_failed_url: str | None = None
    with output_path.open("r", encoding="utf-8") as input_file:
        for raw_line in input_file:
            line = raw_line.strip()
            if line == "":
                continue
            record = json.loads(line)
            if not isinstance(record, dict):
                continue
            house_raw = record.get("house")
            house = house_raw if isinstance(house_raw, dict) else {}
            error_message_raw = house.get("error_message")
            error_message = error_message_raw if isinstance(error_message_raw, str) else ""
            if not error_message.startswith("prop_page_failed"):
                continue
            prop_url_raw = record.get("prop_url")
            if not isinstance(prop_url_raw, str):
                continue
            if prop_url_raw.strip() == "":
                continue
            last_failed_url = prop_url_raw.strip()
    return last_failed_url


def _region_output_path(region: str) -> Path:
    return Path("data") / f"m2_{region}_{DATE_TAG}.jsonl"


def _region_log_path(region: str) -> Path:
    return Path("logs") / f"m2_{region}_progress.log"


def _append_region_log(region: str, text: str) -> None:
    log_path = _region_log_path(region)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log_file:
        log_file.write(text)
        log_file.write("\n")


def _run_single_region(
    region: str,
    region_url: str,
    pages: int,
    timeout_sec: int,
    proxy_pool: list[str],
) -> RegionRunSummary:
    scheduler_module = importlib.import_module("src.scheduler")
    from src.scheduler import M0RunConfig
    from src.scheduler import run_m0

    _append_region_log(region, f"start region={region}, url={region_url}, pages={pages}")
    output_path = _region_output_path(region)
    sticky_proxy = _pick_sticky_proxy(region, proxy_pool)
    max_attempts = MAX_REGION_ATTEMPTS if len(proxy_pool) == 0 else min(MAX_REGION_ATTEMPTS, len(proxy_pool))
    manual_login_timeout_sec = min(timeout_sec, int(os.getenv("M2_MANUAL_LOGIN_TIMEOUT_SEC", "60")))
    visible = os.getenv("M2_VISIBLE", "false").strip().lower() == "true"
    profile_root_dir = _build_profile_dir(region)
    _append_region_log(region, f"sticky_profile region={region}, dir={profile_root_dir}")
    if sticky_proxy is not None:
        _append_region_log(region, f"sticky_proxy region={region}, proxy={sticky_proxy}")

    original_append_jsonl = scheduler_module.append_jsonl
    progress_state: dict[str, float | int] = {
        "records": 0,
        "start_ts": monotonic(),
        "prop_failed": 0,
    }

    def _progress_append_jsonl(records: list[dict[str, object]], target_output_path: Path) -> None:
        original_append_jsonl(records, target_output_path)
        if target_output_path != output_path:
            return
        progress_state["records"] = int(progress_state["records"]) + len(records)
        for record in records:
            house_raw = record.get("house")
            house = house_raw if isinstance(house_raw, dict) else {}
            error_message_raw = house.get("error_message")
            if isinstance(error_message_raw, str) and error_message_raw.startswith("prop_page_failed"):
                progress_state["prop_failed"] = int(progress_state["prop_failed"]) + 1
        records_count = int(progress_state["records"])
        if records_count == 0 or records_count % 10 != 0:
            return
        elapsed = monotonic() - float(progress_state["start_ts"])
        avg_cost = elapsed / records_count
        success_estimate = _calc_success_rate(records_count, int(progress_state["prop_failed"]))
        progress_text = (
            f"progress region={region}, records={records_count}, "
            f"success_rate_estimate={success_estimate:.4f}, avg_cost_sec={avg_cost:.2f}"
        )
        _append_region_log(region, progress_text)
        print(progress_text)

    result: dict[str, object] | None = None
    last_error: Exception | None = None
    for attempt_no in range(max_attempts):
        attempt_proxy = _pick_proxy_for_attempt(region, proxy_pool, attempt_no) if len(proxy_pool) > 0 else None
        attempt_profile_dir = profile_root_dir / f"attempt_{attempt_no + 1}"
        attempt_profile_dir.mkdir(parents=True, exist_ok=True)
        attempt_wait_ms = DEFAULT_WAIT_MS + attempt_no * 1000
        run_config = M0RunConfig(
            region_slug=region,
            pages=pages,
            max_props=MAX_PROPS_PER_PAGE,
            wait_ms=attempt_wait_ms,
            visible=visible,
            manual_login=True,
            manual_login_timeout_sec=manual_login_timeout_sec,
            save_html_on_error=True,
            error_html_dir=Path("data/debug/error_html"),
            output_path=output_path,
            persistent_user_data_dir=attempt_profile_dir,
            proxy_server=attempt_proxy,
        )
        _append_region_log(
            region,
            (
                f"attempt_start region={region}, attempt={attempt_no + 1}/{max_attempts}, "
                f"proxy={attempt_proxy if attempt_proxy is not None else 'none'}, "
                f"wait_ms={attempt_wait_ms}, manual_login_timeout_sec={manual_login_timeout_sec}"
            ),
        )
        with APPEND_JSONL_PATCH_LOCK:
            scheduler_module.append_jsonl = _progress_append_jsonl
            try:
                result = run_m0(run_config, {region: region_url})
                last_error = None
                break
            except Exception as error:
                last_error = error
                _append_region_log(region, f"attempt_failed region={region}, attempt={attempt_no + 1}, error={error}")
            finally:
                scheduler_module.append_jsonl = original_append_jsonl
    if result is None:
        if last_error is not None:
            raise last_error
        raise AssertionError(f"区域运行失败且无异常信息, region={region}")

    written_records = int(result["written_records"])
    prop_failed = int(result["prop_failed"])
    community_failed = int(result["community_failed"])
    success_rate = _calc_success_rate(written_records, prop_failed)
    field_coverage_rate = _calc_field_coverage_rate(output_path)
    community_output_path = Path(str(result["community_output_path"]))
    last_failed_prop_url = _find_last_failed_prop_url(output_path)

    _append_region_log(
        region,
        (
            f"finish region={region}, written={written_records}, prop_failed={prop_failed}, "
            f"community_failed={community_failed}, success_rate={success_rate:.4f}, "
            f"field_coverage_rate={field_coverage_rate:.4f}"
        ),
    )

    if success_rate < MIN_SUCCESS_RATE:
        _append_region_log(region, f"success_rate_below_threshold region={region}, threshold={MIN_SUCCESS_RATE}")
        diagnostic_url = last_failed_prop_url if last_failed_prop_url is not None else region_url
        _append_region_log(region, f"diagnostic_url region={region}, url={diagnostic_url}")
        raise AssertionError(
            f"区域成功率不足90%, region={region}, success_rate={success_rate:.4f}, "
            f"最后失败链接={diagnostic_url}"
        )

    if field_coverage_rate < 0.95:
        raise AssertionError(
            f"字段覆盖率不足95%, region={region}, field_coverage_rate={field_coverage_rate:.4f}"
        )

    return RegionRunSummary(
        region=region,
        written_records=written_records,
        prop_failed=prop_failed,
        community_failed=community_failed,
        success_rate=success_rate,
        field_coverage_rate=field_coverage_rate,
        output_path=output_path,
        community_output_path=community_output_path,
        last_failed_prop_url=last_failed_prop_url,
    )


def test_m2_full_run(request: pytest.FixtureRequest) -> None:
    region_option = str(request.config.getoption("--regions"))
    pages = int(request.config.getoption("--pages"))
    workers = int(request.config.getoption("--workers"))
    timeout_sec = int(request.config.getoption("--timeout"))
    resume = bool(request.config.getoption("--resume"))
    checkpoint_per_region = bool(request.config.getoption("--checkpoint-per-region"))
    proxy_pool = _resolve_proxy_pool()
    degrade_workers_on_antispam = os.getenv("M2_DEGRADE_WORKERS_ON_ANTISPAM", "true").strip().lower() == "true"

    if pages <= 0:
        raise ValueError(f"pages必须大于0, pages={pages}")
    if workers <= 0 or workers > MAX_WORKERS_LIMIT:
        raise ValueError(f"workers必须在1-{MAX_WORKERS_LIMIT}之间, workers={workers}")
    if timeout_sec < 30:
        raise ValueError(f"timeout建议>=30秒, timeout={timeout_sec}")

    region_map_raw = json.loads(Path("config/regions.json").read_text(encoding="utf-8"))
    if not isinstance(region_map_raw, dict):
        raise ValueError("config/regions.json 格式非法")
    region_map: dict[str, str] = {}
    for key, value in region_map_raw.items():
        if isinstance(key, str) and isinstance(value, str):
            region_map[key] = value

    planned_regions = _read_regions(region_option, region_map)
    checkpoint = _load_checkpoint() if resume else {}
    pending_regions: list[str] = []
    for region in planned_regions:
        if resume and region in checkpoint and bool(checkpoint[region].get("done", False)):
            continue
        pending_regions.append(region)

    checkpoint_lock = threading.Lock()
    summaries: list[RegionRunSummary] = []
    errors: list[str] = []

    effective_workers = workers
    if degrade_workers_on_antispam and len(proxy_pool) == 0:
        effective_workers = 1
    with ThreadPoolExecutor(max_workers=effective_workers) as executor:
        future_map: dict[Future[RegionRunSummary], str] = {}
        for region in pending_regions:
            future = executor.submit(_run_single_region, region, region_map[region], pages, timeout_sec, proxy_pool)
            future_map[future] = region
        for future, region in future_map.items():
            try:
                summary = future.result()
                summaries.append(summary)
                with checkpoint_lock:
                    checkpoint[region] = {
                        "done": True,
                        "written_records": summary.written_records,
                        "prop_failed": summary.prop_failed,
                        "community_failed": summary.community_failed,
                        "success_rate": summary.success_rate,
                        "output_path": str(summary.output_path),
                        "community_output_path": str(summary.community_output_path),
                        "field_coverage_rate": summary.field_coverage_rate,
                        "last_failed_prop_url": summary.last_failed_prop_url,
                    }
                    if checkpoint_per_region:
                        _save_checkpoint(checkpoint)
            except Exception as error:
                errors.append(f"region={region}, error={error}")

    for summary in summaries:
        assert summary.output_path.exists(), f"主表缺失: {summary.output_path}"
        assert summary.community_output_path.exists(), f"侧表缺失: {summary.community_output_path}"
        assert summary.written_records > 0, f"区域无写入记录: {summary.region}"

    if len(errors) > 0:
        raise AssertionError("M2全量测试存在失败区域:\n" + "\n".join(errors))

    if not checkpoint_per_region:
        _save_checkpoint(checkpoint)

    print("| 区域 | 写入记录 | 三级失败 | 四级失败 | 成功率 | 字段覆盖率 |")
    print("| --- | ---: | ---: | ---: | ---: | ---: |")
    for summary in sorted(summaries, key=lambda item: item.region):
        print(
            f"| {summary.region} | {summary.written_records} | {summary.prop_failed} | "
            f"{summary.community_failed} | {summary.success_rate:.2%} | {summary.field_coverage_rate:.2%} |"
        )
