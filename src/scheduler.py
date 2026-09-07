from __future__ import annotations

import os
from pathlib import Path
import re
from time import monotonic
from urllib.parse import quote
from urllib.parse import urlsplit

from playwright.sync_api import Page
from playwright.sync_api import sync_playwright
from pydantic import BaseModel
from pydantic import Field

# 登录凭据默认从环境变量读取，避免明文硬编码入库
_DEFAULT_USERNAME = os.environ.get("ANJUKE_USERNAME", "")
_DEFAULT_PASSWORD = os.environ.get("ANJUKE_PASSWORD", "")

from src.parsers.comm_parser import parse_community_page
from src.parsers.list_parser import parse_list_page
from src.parsers.poi_parser import extract_poi_details
from src.parsers.poi_parser import extract_poi_summary
from src.parsers.prop_parser import parse_prop_page
from src.storage.jsonl_storage import append_jsonl
from src.storage.jsonl_storage import write_jsonl
from src.storage.failure_store import FailureStore
from src.storage.run_log_store import RunLogStore
from src.exceptions import BanError
from src.exceptions import PageCrashError
from src.auth import AuthManager, LoginOptions


class M0RunConfig(BaseModel):
    region_slug: str = Field(min_length=1, description="区域英文slug")
    pages: int = Field(ge=1, description="每个区域采集页数")
    max_props: int = Field(ge=1, description="每页最多处理房源数")
    wait_ms: int = Field(ge=500, le=20000, description="页面加载后等待毫秒数")
    visible: bool = Field(description="是否显示浏览器窗口")
    manual_login: bool = Field(description="是否允许人工扫码登录")
    manual_login_timeout_sec: int = Field(ge=30, le=1800, description="人工扫码登录超时秒数")
    save_html_on_error: bool = Field(description="异常页面是否保存HTML")
    error_html_dir: Path = Field(description="异常页面HTML保存目录")
    output_path: Path = Field(description="输出JSONL路径")
    persistent_user_data_dir: Path | None = Field(default=None, description="Playwright持久化会话目录")
    proxy_server: str | None = Field(default=None, description="Playwright代理地址，格式如http://user:pass@host:port")
    # 登录相关配置
    auto_login: bool = Field(default=True, description="是否在爬取前自动登录")
    username: str = Field(default=_DEFAULT_USERNAME, description="登录账号")
    password: str = Field(default=_DEFAULT_PASSWORD, description="登录密码")
    run_id: str | None = Field(default=None, description="任务运行ID（可选）")


def _build_page_url(region_base_url: str, page_no: int) -> str:
    if page_no == 1:
        return region_base_url
    return f"{region_base_url.rstrip('/')}/p{page_no}/"


def _is_login_or_captcha_page(html: str) -> bool:
    return (
        "请输入验证码" in html
        or "/antibot/" in html
        or "geetest" in html
        or "微信登录" in html
        or "antispam-block" in html
    )


# =============================================================================
# 页面崩溃检测与恢复 (Out of Memory处理)
# =============================================================================

def _is_crash_page(html: str) -> bool:
    """检测是否出现浏览器崩溃页面 (Out of Memory)"""
    crash_indicators = [
        "喔唷，崩溃啦",
        "Out of Memory",
        "显示此网页时出了点问题",
        "错误代码：Out of Memory",
        "重新加载",
    ]
    return any(indicator in html for indicator in crash_indicators)


def _try_click_reload_button(page: Page) -> bool:
    """尝试点击"重新加载"按钮"""
    try:
        # 尝试多种可能的选择器
        reload_selectors = [
            "text=重新加载",
            "button:has-text('重新加载')",
            "[id*='reload']",
            "[class*='reload']",
        ]
        for selector in reload_selectors:
            reload_btn = page.locator(selector)
            if reload_btn.count() > 0:
                reload_btn.first.click(timeout=5000)
                page.wait_for_load_state("domcontentloaded", timeout=30000)
                return True
    except Exception:
        pass
    return False


def _handle_crash_and_reload(
    page: Page,
    target_url: str,
    config: M0RunConfig,
    page_no: int,
    max_retries: int = 3,
) -> str:
    """
    处理页面崩溃并尝试恢复
    
    Args:
        page: Playwright Page实例
        target_url: 目标URL
        config: 运行配置
        page_no: 当前页码
        max_retries: 最大重试次数
    
    Returns:
        恢复后的页面HTML
    
    Raises:
        PageCrashError: 重试后仍无法恢复
    """
    for attempt in range(1, max_retries + 1):
        print(f"[崩溃恢复] 第{attempt}次尝试恢复页面: {target_url}")
        
        # 尝试点击重新加载按钮
        if _try_click_reload_button(page):
            page.wait_for_timeout(config.wait_ms)
            html = _safe_page_content(page)
            if not _is_crash_page(html):
                print(f"[崩溃恢复] 页面已成功恢复")
                return html
        
        # 备选：直接重新导航
        try:
            page.goto(target_url, wait_until="domcontentloaded", timeout=60000)
            page.wait_for_timeout(config.wait_ms)
            html = _safe_page_content(page)
            if not _is_crash_page(html):
                print(f"[崩溃恢复] 通过重新导航恢复页面")
                return html
        except Exception as e:
            print(f"[崩溃恢复] 重新导航失败: {e}")
        
        # 指数退避等待
        wait_time = 2000 * attempt
        print(f"[崩溃恢复] 等待{wait_time}ms后重试...")
        page.wait_for_timeout(wait_time)
    
    # 保存错误HTML并抛出异常
    final_html = _safe_page_content(page)
    _save_error_html(
        config=config,
        stage="page_crash_out_of_memory",
        page_no=page_no,
        current_url=page.url,
        html=final_html,
    )
    raise PageCrashError(f"页面崩溃(Out of Memory)且{max_retries}次重试后仍无法恢复: {target_url}")


def _slug_text(raw_text: str) -> str:
    normalized = re.sub(r"[^a-zA-Z0-9_-]+", "_", raw_text)
    return normalized.strip("_")[:80] or "unknown"


def _save_error_html(config: M0RunConfig, stage: str, page_no: int, current_url: str, html: str) -> None:
    if not config.save_html_on_error:
        return
    config.error_html_dir.mkdir(parents=True, exist_ok=True)
    safe_stage = _slug_text(stage)
    safe_region = _slug_text(config.region_slug)
    safe_url = _slug_text(quote(current_url, safe=""))
    target_file = config.error_html_dir / f"{safe_region}_p{page_no}_{safe_stage}_{safe_url}.html"
    target_file.write_text(html, encoding="utf-8")
    print(f"已保存异常页面HTML: {target_file}")


def _try_click_human_verify_button(page: Page) -> bool:
    verify_button = page.locator("#btnSubmit")
    if verify_button.count() > 0:
        try:
            verify_button.first.click(timeout=5000)
            return True
        except Exception:
            return False
    text_button = page.locator("text=点击按钮进行验证")
    if text_button.count() > 0:
        try:
            text_button.first.click(timeout=5000)
            return True
        except Exception:
            return False
    return False


def _is_callback_antibot_url(url: str) -> bool:
    return "callback.58.com/antibot/verifycode" in url


def _perform_login(
    page: Page,
    username: str = _DEFAULT_USERNAME,
    password: str = _DEFAULT_PASSWORD,
    visible: bool = True,
    manual_login_timeout_sec: int = 180,
) -> bool:
    """
    执行安居客自动登录流程
    
    步骤：
    1. 导航到安居客首页
    2. 点击右上角登录按钮
    3. 点击"账号密码登录"选项卡
    4. 填写账号和密码
    5. 点击登录按钮
    6. 处理协议确认弹窗（如果出现）
    7. 等待登录完成跳转
    
    Args:
        page: Playwright Page 实例
        username: 登录账号
        password: 登录密码
        visible: 是否可视化模式
        manual_login_timeout_sec: 人工登录超时时间
    
    Returns:
        bool: 登录是否成功
    """
    from time import monotonic
    
    print("=" * 60)
    print("开始执行安居客自动登录")
    print("=" * 60)

    if not username or not password:
        print("[登录] 未提供账号密码（可设置环境变量 ANJUKE_USERNAME/ANJUKE_PASSWORD 或通过 --username/--password 传入），跳过自动登录")
        return False

    try:
        # 1. 导航到安居客首页
        print("[登录] 步骤 1/7: 导航到安居客南昌站...")
        page.goto("https://nc.anjuke.com/", wait_until="domcontentloaded", timeout=60000)
        page.wait_for_timeout(2000)
        
        # 检查是否已经是登录状态
        html = page.content()
        if "退出" in html or "个人中心" in html or "我的安居客" in html:
            print("[登录] 检测到已处于登录状态，跳过登录流程")
            return True
        
        # 2. 点击右上角登录按钮
        print("[登录] 步骤 2/7: 点击登录按钮...")
        login_link = page.locator("a:has-text('登录')").first
        if login_link.count() == 0:
            # 尝试其他选择器
            login_link = page.locator("[data-testid='login-button'], .login-btn, a[href*='login']").first
        if login_link.count() == 0:
            print("[登录] 错误: 未找到登录按钮")
            return False
        login_link.click()
        
        # 等待登录页加载
        page.wait_for_url("**/login.anjuke.com/**", timeout=15000)
        print(f"[登录] 已跳转到登录页: {page.url}")
        page.wait_for_timeout(1500)
        
        # 3. 点击"账号密码登录"选项卡
        print("[登录] 步骤 3/7: 点击'账号密码登录'选项卡...")
        account_tab = page.locator("text=账号密码登录").first
        if account_tab.count() > 0:
            account_tab.click()
            page.wait_for_timeout(800)
        
        # 4. 填写账号
        print("[登录] 步骤 4/7: 填写账号...")
        username_input = page.locator("input[placeholder*='用户名'], input[placeholder*='手机号'], input[name='username'], #username").first
        if username_input.count() == 0:
            username_input = page.locator("input[type='text']").first
        if username_input.count() > 0:
            username_input.fill(username)
        else:
            print("[登录] 警告: 未找到账号输入框")
        
        # 5. 填写密码
        print("[登录] 步骤 5/7: 填写密码...")
        password_input = page.locator("input[type='password'], input[name='password'], #password").first
        if password_input.count() > 0:
            password_input.fill(password)
        else:
            print("[登录] 警告: 未找到密码输入框")
        
        # 6. 点击登录按钮
        print("[登录] 步骤 6/7: 点击登录按钮...")
        # 先尝试查找checkbox并勾选（如果有）
        checkbox = page.locator("input[type='checkbox']").first
        if checkbox.count() > 0:
            checkbox.check()
            print("[登录] 已勾选同意协议checkbox")
        
        login_btn = page.locator("button[type='submit'], .login-submit-btn, text=登录").first
        if login_btn.count() > 0:
            login_btn.click()
        else:
            print("[登录] 错误: 未找到登录按钮")
            return False
        
        # 等待协议确认弹窗或跳转
        page.wait_for_timeout(1500)
        
        # 7. 处理协议确认弹窗（如果出现）
        print("[登录] 步骤 7/7: 处理协议确认...")
        agree_btn = page.locator("text=同意并继续").first
        if agree_btn.count() > 0:
            print("[登录] 检测到协议确认弹窗，点击'同意并继续'...")
            agree_btn.click()
        
        # 等待登录完成跳转
        print("[登录] 等待登录完成跳转...")
        deadline = monotonic() + manual_login_timeout_sec
        while monotonic() < deadline:
            current_url = page.url
            if "login" not in current_url and "anjuke.com" in current_url:
                print(f"[登录] 登录成功! 当前页面: {current_url}")
                print("=" * 60)
                return True
            # 检查是否需要人工介入
            if _is_login_or_captcha_page(page.content()):
                if not visible:
                    print("[登录] 错误: 触发验证码/验证页面但visible=false")
                    return False
                print("[登录] 检测到验证码/验证页面，等待人工处理...")
                # 尝试自动点击验证按钮
                if _try_click_human_verify_button(page):
                    print("[登录] 已自动点击验证按钮")
            page.wait_for_timeout(2000)
        
        print("[登录] 错误: 登录超时")
        return False
        
    except Exception as e:
        print(f"[登录] 登录过程中发生错误: {e}")
        return False


def _auto_click_callback_verify(page: Page) -> bool:
    rounds = 6
    for _ in range(rounds):
        clicked = _try_click_human_verify_button(page)
        if not clicked:
            text_button = page.locator("text=点击按钮进行验证")
            if text_button.count() > 0:
                text_button.first.click(timeout=5000)
                clicked = True
        page.wait_for_timeout(1500)
        current_html = _safe_page_content(page)
        current_url = page.url
        if not _is_login_or_captcha_page(current_html) and "antispam-block" not in current_url:
            return True
        if not _is_callback_antibot_url(current_url) and not clicked:
            break
    return False


def _pick_or_create_active_page(context: object) -> Page:
    for existing_page in context.pages:
        if existing_page.url != "about:blank":
            return existing_page
    return context.new_page()


def _auto_handle_antibot_transition(page: Page, target_url: str, wait_ms: int) -> bool:
    for _ in range(6):
        current_url = page.url
        current_html = _safe_page_content(page)
        if _is_callback_antibot_url(current_url):
            callback_ok = _auto_click_callback_verify(page)
            if callback_ok:
                page.goto(target_url, wait_until="domcontentloaded", timeout=60000)
                page.wait_for_timeout(wait_ms)
                continue
        if "antispam-block" in current_url:
            clicked = _try_click_human_verify_button(page)
            if clicked:
                page.wait_for_timeout(1500)
                continue
            page.goto(target_url, wait_until="domcontentloaded", timeout=60000)
            page.wait_for_timeout(wait_ms)
            continue
        if not _is_login_or_captcha_page(current_html):
            return True
        page.wait_for_timeout(1200)
    final_html = _safe_page_content(page)
    final_url = page.url
    return not _is_login_or_captcha_page(final_html) and "antispam-block" not in final_url


def _build_community_sidecar_path(output_path: Path) -> Path:
    return output_path.with_name(f"{output_path.stem}_community{output_path.suffix}")


def _normalize_community_url(raw_url: str) -> str:
    parsed = urlsplit(raw_url)
    return f"{parsed.scheme}://{parsed.netloc}{parsed.path}"


def _build_proxy_settings(proxy_server: str | None) -> dict[str, str] | None:
    if proxy_server is None:
        return None
    normalized_proxy = proxy_server.strip()
    if normalized_proxy == "":
        return None
    return {"server": normalized_proxy}


def _safe_page_content(page: Page) -> str:
    for _ in range(6):
        try:
            return page.content()
        except Exception:
            page.wait_for_timeout(400)
    return page.content()


def _dedupe_poi_items(items: list[dict[str, str | None]]) -> list[dict[str, str | None]]:
    deduped_items: list[dict[str, str | None]] = []
    seen_keys: set[tuple[str, str | None, str | None]] = set()
    for item in items:
        name = item.get("name")
        if name is None or name.strip() == "":
            continue
        dedupe_key = (
            name.strip(),
            (item.get("distance") or "").strip() or None,
            (item.get("sub_type") or "").strip() or None,
        )
        if dedupe_key in seen_keys:
            continue
        seen_keys.add(dedupe_key)
        deduped_items.append(item)
    return deduped_items


def _extract_school_sub_type(text: str) -> str | None:
    if "幼儿园" in text:
        return "幼儿园"
    if "小学" in text:
        return "小学"
    if "中学" in text:
        return "中学"
    if "大学" in text or "学院" in text or "高等" in text:
        return "高等院校"
    return None


def _wait_poi_list_render(page: Page, item_selector: str, timeout_ms: int) -> None:
    deadline = monotonic() + timeout_ms / 1000
    stable_rounds = 0
    previous_count = -1
    while monotonic() < deadline:
        item_count = page.locator(item_selector).count()
        loading_text_count = page.locator("div.tabBox").locator("text=加载中").count()
        if item_count > 0 and loading_text_count == 0:
            if item_count == previous_count:
                stable_rounds += 1
                if stable_rounds >= 2:
                    return
            else:
                stable_rounds = 0
        previous_count = item_count
        page.wait_for_timeout(400)


def _scroll_poi_list_until_stable(page: Page, container_selector: str, item_selector: str) -> None:
    stable_rounds = 0
    previous_count = -1
    for _ in range(10):
        page.evaluate(
            """
            (selector) => {
              const el = document.querySelector(selector);
              if (!el) return;
              el.scrollTop = el.scrollHeight;
            }
            """,
            container_selector,
        )
        page.wait_for_timeout(500)
        current_count = page.locator(item_selector).count()
        if current_count == previous_count:
            stable_rounds += 1
            if stable_rounds >= 2:
                break
        else:
            stable_rounds = 0
        previous_count = current_count


def _collect_poi_details_with_ui(page: Page) -> dict[str, list[dict[str, str | None]]]:
    tab_type_map: dict[str, str] = {
        "公交": "bus",
        "地铁": "subway",
        "学校": "school",
        "餐饮": "restaurant",
        "购物": "shop",
        "医院": "hospital",
        "银行": "bank",
    }
    details: dict[str, list[dict[str, str | None]]] = {value: [] for value in tab_type_map.values()}

    base_tab_selector = "div.tabBox ul.aroundTypeJs[data-index='0']"
    base_subword_selector = "div.tabBox div.subwordJs[data-index='0']"
    base_item_selector = "div.tabBox ul.itemBox.contanerBox[data-index='0'] > li.licontaner"
    base_item_container_selector = "div.tabBox ul.itemBox.contanerBox[data-index='0']"

    for tab_text, poi_type in tab_type_map.items():
        tab_locator = page.locator(f"{base_tab_selector} > li", has_text=tab_text)
        if tab_locator.count() == 0:
            continue
        tab_locator.first.click(timeout=5000)
        _wait_poi_list_render(page, base_item_selector, timeout_ms=5000)

        school_sub_types: list[str | None]
        if poi_type == "school":
            school_sub_types = ["幼儿园", "小学", "中学", "高等院校"]
        else:
            school_sub_types = [None]

        collected_items: list[dict[str, str | None]] = []
        for school_sub_type in school_sub_types:
            if school_sub_type is not None:
                school_sub_locator = page.locator(
                    f"{base_subword_selector} span, {base_subword_selector} li, {base_subword_selector} a, {base_subword_selector} div",
                    has_text=school_sub_type,
                )
                if school_sub_locator.count() == 0:
                    # 子类节点缺失时，退化为学校总类明细，避免学校数据整体为空。
                    school_sub_type = None
                else:
                    school_sub_locator.first.click(timeout=5000)
                    _wait_poi_list_render(page, base_item_selector, timeout_ms=4000)

            # 仅滚动POI列表容器，禁止滚动整页；滚动到列表条目数稳定后再采集。
            _scroll_poi_list_until_stable(page, base_item_container_selector, base_item_selector)
            _wait_poi_list_render(page, base_item_selector, timeout_ms=3000)

            item_locator = page.locator(base_item_selector)
            item_count = item_locator.count()
            for item_index in range(item_count):
                current_item = item_locator.nth(item_index)
                title_locator = current_item.locator(".itemContent .itemTitle")
                if title_locator.count() == 0:
                    continue
                item_title = title_locator.first.inner_text().strip()
                if item_title in {"公交", "地铁", "学校", "餐饮", "购物", "医院", "银行"}:
                    continue
                distance_locator = current_item.locator(".itemContent .itemdistance")
                item_distance_raw = ""
                if distance_locator.count() > 0:
                    item_distance_raw = distance_locator.first.inner_text().strip()
                item_distance = item_distance_raw if item_distance_raw != "" else None
                item_info_locator = current_item.locator(".itemInfo")
                item_info_raw = ""
                if item_info_locator.count() > 0:
                    item_info_raw = item_info_locator.first.inner_text().strip()
                item_info = item_info_raw if item_info_raw != "" else None
                coordinate = current_item.get_attribute("data-address")
                collected_items.append(
                    {
                        "type": poi_type,
                        "name": item_title,
                        "distance": item_distance,
                        "itemcontent": f"{item_title}{item_distance or ''}",
                        "iteminfo": item_info,
                        "coordinate": coordinate,
                        "address": item_info,
                        "sub_type": school_sub_type if poi_type == "school" else None,
                    }
                )
        details[poi_type] = _dedupe_poi_items(collected_items)
    return details


def _build_poi_summary_from_details(details: dict[str, list[dict[str, str | None]]]) -> dict[str, int]:
    summary: dict[str, int] = {}
    for poi_type in ["bank", "bus", "subway", "school", "restaurant", "shop", "hospital"]:
        summary[poi_type] = len(details.get(poi_type, []))
    return summary


def _wait_for_manual_login(page: Page, config: M0RunConfig, current_url: str, page_no: int) -> None:
    if not config.visible:
        raise BanError(f"触发登录/验证码页但visible=false, url={current_url}")
    if not config.manual_login:
        raise BanError(f"触发登录/验证码页且未启用manual-login, url={current_url}")

    if _is_callback_antibot_url(current_url):
        auto_verified = _auto_click_callback_verify(page)
        if auto_verified:
            print("检测到反爬回调页，已自动完成按钮验证，继续执行链路。")
            return

    clicked_verify_button = _try_click_human_verify_button(page)
    has_verify_text = page.locator("text=点击按钮进行验证").count() > 0
    has_geetest_widget = page.locator("[class*=geetest]").count() > 0
    current_html = _safe_page_content(page)
    hard_block_text = "系统检测到您正在使用网页抓取工具访问安居客网站"
    if hard_block_text in current_html:
        _save_error_html(
            config=config,
            stage="hard_antispam_block",
            page_no=page_no,
            current_url=current_url,
            html=current_html,
        )
        raise BanError(f"命中硬封禁页文本特征, url={current_url}")
    if "antispam-block" in current_url and not clicked_verify_button and not has_verify_text and not has_geetest_widget:
        _save_error_html(
            config=config,
            stage="hard_antispam_block",
            page_no=page_no,
            current_url=current_url,
            html=current_html,
        )
        raise BanError(f"命中硬封禁页且无可交互验证控件, url={current_url}")
    if clicked_verify_button:
        print("检测到验证码页面，已自动点击“点击按钮进行验证”。")
    print("检测到登录/验证码页面，请完成人工验证。")
    print(f"当前页面: {current_url}")
    print("验证完成后请保持页面不关闭，程序将自动检测并继续。")

    deadline = monotonic() + config.manual_login_timeout_sec
    while monotonic() < deadline:
        page.wait_for_timeout(2000)
        html = _safe_page_content(page)
        if not _is_login_or_captcha_page(html):
            print("人工登录验证完成，继续执行链路。")
            return
    _save_error_html(
        config=config,
        stage="manual_login_timeout",
        page_no=page_no,
        current_url=current_url,
        html=_safe_page_content(page),
    )
    raise BanError(f"人工扫码登录超时, timeout_sec={config.manual_login_timeout_sec}, url={current_url}")


def _goto_and_get_html(
    page: Page,
    url: str,
    config: M0RunConfig,
    page_no: int,
    stage: str,
) -> str:
    page.goto(url, wait_until="domcontentloaded", timeout=60000)
    page.wait_for_timeout(config.wait_ms)
    html = _safe_page_content(page)
    current_page_url = page.url
    
    # 检测并处理页面崩溃 (Out of Memory)
    if _is_crash_page(html):
        print(f"[警告] 检测到页面崩溃(Out of Memory): {current_page_url}")
        html = _handle_crash_and_reload(page, url, config, page_no)
        current_page_url = page.url
    
    if _is_login_or_captcha_page(html) or "antispam-block" in current_page_url:
        auto_recovered = _auto_handle_antibot_transition(page, url, config.wait_ms)
        if auto_recovered:
            return _safe_page_content(page)
        _save_error_html(
            config=config,
            stage=f"{stage}_login_or_captcha_before_manual",
            page_no=page_no,
            current_url=current_page_url,
            html=html,
        )
        _wait_for_manual_login(page, config, current_page_url, page_no)
        page.wait_for_timeout(config.wait_ms)
        html = _safe_page_content(page)
        current_page_url = page.url
        if _is_callback_antibot_url(current_page_url):
            page.goto(url, wait_until="domcontentloaded", timeout=60000)
            page.wait_for_timeout(config.wait_ms)
            html = _safe_page_content(page)
            current_page_url = page.url
        if _is_login_or_captcha_page(html) or "antispam-block" in current_page_url:
            _save_error_html(
                config=config,
                stage=f"{stage}_login_or_captcha_after_manual",
                page_no=page_no,
                current_url=current_page_url,
                html=html,
            )
            raise BanError(f"{stage}页面登录/验证码未解除, url={current_page_url}")
    return html


def run_m0(config: M0RunConfig, region_map: dict[str, str]) -> dict[str, int]:
    if config.region_slug not in region_map:
        raise ValueError(f"区域不存在于config/regions.json: {config.region_slug}")
    region_base_url = region_map[config.region_slug]
    written_records: int = 0
    community_failed: int = 0
    prop_failed: int = 0
    community_cache: dict[str, dict[str, object]] = {}
    failure_store = FailureStore(config.output_path.parent, config.region_slug)
    run_log = RunLogStore(
        log_root=Path("logs") / "runs",
        region=config.region_slug,
        phase="m0",
        run_id=config.run_id,
    )
    run_log.event(
        stage="m0_start",
        status="started",
        detail={
            "pages": config.pages,
            "max_props": config.max_props,
            "output_path": str(config.output_path),
        },
    )

    with sync_playwright() as playwright:
        proxy_settings = _build_proxy_settings(config.proxy_server)
        browser = None
        if config.persistent_user_data_dir is not None:
            config.persistent_user_data_dir.mkdir(parents=True, exist_ok=True)
            context = playwright.chromium.launch_persistent_context(
                user_data_dir=str(config.persistent_user_data_dir),
                headless=not config.visible,
                locale="zh-CN",
                proxy=proxy_settings,
            )
        else:
            browser = playwright.chromium.launch(headless=not config.visible, proxy=proxy_settings)
            context = browser.new_context(locale="zh-CN")
        page = _pick_or_create_active_page(context)

        # 登录流程：如果启用自动登录，则在爬取前执行登录
        if config.auto_login:
            print("\n[系统] 启用了自动登录，正在执行登录流程...")
            login_options = LoginOptions(
                username=config.username,
                password=config.password,
                timeout_sec=config.manual_login_timeout_sec,
                visible=config.visible,
            )
            login_result = AuthManager.perform_login(page, login_options)
            if not login_result.success:
                print(f"[系统] 警告: 自动登录失败: {login_result.state.error_message}")
                print("[系统] 将继续尝试以未登录状态爬取")
            else:
                print(f"[系统] 登录成功! 用户: {login_result.state.username}")
                print("[系统] 准备开始爬取任务\n")
            # 登录完成后，先访问目标区域的首页，确保会话有效
            region_base_url = region_map[config.region_slug]
            print(f"[系统] 正在跳转到目标区域: {region_base_url}")
            page.goto(region_base_url, wait_until="domcontentloaded", timeout=60000)
            page.wait_for_timeout(config.wait_ms)
        
        for page_no in range(1, config.pages + 1):
            run_log.inc("list_page", total=1, success=0, failed=0)
            list_url = _build_page_url(region_base_url, page_no)
            try:
                list_html = _goto_and_get_html(page=page, url=list_url, config=config, page_no=page_no, stage="list")
                parsed_list = parse_list_page(region_base_url, list_html, config.region_slug, page_no)
                run_log.inc("list_page", total=0, success=1, failed=0)
            except Exception as error:
                run_log.inc("list_page", total=0, success=0, failed=1)
                run_log.event(
                    stage="list_page_failed",
                    status="failed",
                    detail={"page_no": page_no, "list_url": list_url, "error": str(error)},
                )
                raise
            if len(parsed_list.prop_urls) == 0:
                _save_error_html(
                    config=config,
                    stage="list_no_prop_links",
                    page_no=page_no,
                    current_url=list_url,
                    html=list_html,
                )
                raise ValueError(f"列表页未解析到房源链接, region={config.region_slug}, page_no={page_no}")

            for prop_url in parsed_list.prop_urls[: config.max_props]:
                run_log.inc("prop", total=1, success=0, failed=0)
                record: dict[str, object] = {
                    "region": config.region_slug,
                    "list_page_no": page_no,
                    "prop_url": str(prop_url),
                }
                try:
                    prop_html = _goto_and_get_html(
                        page=page,
                        url=str(prop_url),
                        config=config,
                        page_no=page_no,
                        stage="prop",
                    )
                    house = parse_prop_page(str(prop_url), prop_html, config.region_slug)
                    record["house"] = house.model_dump(mode="json")
                except Exception as error:
                    prop_failed += 1
                    run_log.inc("prop", total=0, success=0, failed=1)
                    _save_error_html(
                        config=config,
                        stage="prop_parse_or_visit_error",
                        page_no=page_no,
                        current_url=str(prop_url),
                        html=_safe_page_content(page),
                    )
                    record["house"] = {
                        "is_valid": False,
                        "error_message": f"prop_page_failed: {error}",
                    }
                    record["community_ref"] = {"community_id": None, "status": "prop_failed"}
                    failure_store.append_prop_failure(
                        run_id=run_log.run_id,
                        phase="m0",
                        page_no=page_no,
                        target_url=str(prop_url),
                        house_id=None,
                        error_type="prop_parse_or_visit_error",
                        error_message=str(error),
                        community_id=None,
                    )
                    append_jsonl([record], config.output_path)
                    written_records += 1
                    continue
                run_log.inc("prop", total=0, success=1, failed=0)

                community_url = house.community_url
                community_id = house.community_id
                if community_url is None or community_id is None:
                    record["community_ref"] = {"community_id": community_id, "status": "missing_in_prop"}
                    append_jsonl([record], config.output_path)
                    written_records += 1
                    continue

                normalized_community_url = _normalize_community_url(str(community_url))
                if community_id in community_cache:
                    record["community_ref"] = {
                        "community_id": community_id,
                        "status": "linked_from_cache",
                    }
                    append_jsonl([record], config.output_path)
                    written_records += 1
                    continue

                run_log.inc("community", total=1, success=0, failed=0)
                try:
                    community_html = _goto_and_get_html(
                        page=page,
                        url=normalized_community_url,
                        config=config,
                        page_no=page_no,
                        stage="community",
                    )
                    tab_item_count = page.locator("div.tabBox li").count()
                    poi_details = _collect_poi_details_with_ui(page)
                    poi_summary = _build_poi_summary_from_details(poi_details)
                    if sum(poi_summary.values()) == 0:
                        # Fallback: 页面结构变更时退回HTML粗提取，避免空数据。
                        poi_details = extract_poi_details(community_html)
                        poi_summary = extract_poi_summary(community_html)
                    community = parse_community_page(
                        normalized_community_url,
                        community_html,
                        tab_item_count,
                        poi_summary,
                        poi_details,
                    )
                    community_data = community.model_dump(mode="json")
                    community_cache[community_id] = {
                        "community_id": community_id,
                        "community": {
                            "community_name": community_data["community_name"],
                            "community_url": community_data["community_url"],
                            "tab_item_count": community_data["tab_item_count"],
                        },
                        "poi": {
                            "summary": community_data["poi_summary"],
                            "details": community_data["poi_details"],
                        },
                    }
                    record["community_ref"] = {
                        "community_id": community_id,
                        "status": "linked_new_fetch",
                    }
                    run_log.inc("community", total=0, success=1, failed=0)
                except Exception as error:
                    community_failed += 1
                    run_log.inc("community", total=0, success=0, failed=1)
                    _save_error_html(
                        config=config,
                        stage="community_parse_or_visit_error",
                        page_no=page_no,
                        current_url=normalized_community_url,
                        html=_safe_page_content(page),
                    )
                    record["community_ref"] = {
                        "community_id": community_id,
                        "status": f"community_failed:{error}",
                    }
                    failure_store.append_community_failure(
                        run_id=run_log.run_id,
                        phase="m0",
                        page_no=page_no,
                        target_url=normalized_community_url,
                        house_id=house.house_id,
                        community_id=community_id,
                        error_type="community_parse_or_visit_error",
                        error_message=str(error),
                    )

                append_jsonl([record], config.output_path)
                written_records += 1

        context.close()
        if browser is not None:
            browser.close()

    community_output_path = _build_community_sidecar_path(config.output_path)
    write_jsonl(list(community_cache.values()), community_output_path)
    run_log.event(
        stage="m0_finish",
        status="completed",
        detail={
            "written_records": written_records,
            "prop_failed": prop_failed,
            "community_failed": community_failed,
            "unique_community_count": len(community_cache),
            "community_output_path": str(community_output_path),
        },
    )

    return {
        "written_records": written_records,
        "prop_failed": prop_failed,
        "community_failed": community_failed,
        "unique_community_count": len(community_cache),
        "community_output_path": str(community_output_path),
    }


# =============================================================================
# 分阶段运行入口 (M2优化)
# =============================================================================

class PhasedRunConfig(BaseModel):
    """分阶段运行配置"""
    region_slug: str = Field(min_length=1, description="区域英文slug")
    phase: str = Field(default="full", description="运行阶段: links, content, full")
    max_pages: int | None = Field(default=None, description="链接提取最大页数限制")
    resume: bool = Field(default=True, description="是否启用断点接续")
    wait_ms: int = Field(default=2500, ge=500, le=20000, description="页面加载后等待毫秒数")
    visible: bool = Field(default=True, description="是否显示浏览器窗口")
    manual_login: bool = Field(default=True, description="是否允许人工扫码登录")
    manual_login_timeout_sec: int = Field(default=180, ge=30, le=1800, description="人工扫码登录超时秒数")
    save_html_on_error: bool = Field(default=True, description="异常页面是否保存HTML")
    error_html_dir: Path = Field(default=Path("data/debug/error_html"), description="异常页面HTML保存目录")
    data_dir: Path = Field(default=Path("data"), description="数据根目录")
    auto_login: bool = Field(default=True, description="是否在爬取前自动登录")
    username: str = Field(default=_DEFAULT_USERNAME, description="登录账号")
    password: str = Field(default=_DEFAULT_PASSWORD, description="登录密码")
    run_id: str | None = Field(default=None, description="任务运行ID（可选）")


def run_phased(config: PhasedRunConfig, region_map: dict[str, str]) -> dict[str, Any]:
    """
    分阶段运行入口
    
    Args:
        config: 分阶段运行配置
        region_map: 区域映射表
    
    Returns:
        运行结果统计
    """
    from src.phased_extractor import LinkExtractor, ContentExtractor, _load_existing_links
    
    if config.region_slug not in region_map:
        raise ValueError(f"区域不存在于config/regions.json: {config.region_slug}")
    
    region_base_url = region_map[config.region_slug]
    result: dict[str, Any] = {"region": config.region_slug}
    run_log = RunLogStore(
        log_root=Path("logs") / "runs",
        region=config.region_slug,
        phase=config.phase,
        run_id=config.run_id,
    )
    failure_store = FailureStore(config.data_dir, config.region_slug)
    run_log.event(
        stage="phased_start",
        status="started",
        detail={"phase": config.phase, "data_dir": str(config.data_dir)},
    )
    
    # 确保目录存在
    config.error_html_dir.mkdir(parents=True, exist_ok=True)
    config.data_dir.mkdir(parents=True, exist_ok=True)
    
    # 构建M0RunConfig用于页面访问
    m0_config = M0RunConfig(
        region_slug=config.region_slug,
        pages=config.max_pages or 9999,
        max_props=60,
        wait_ms=config.wait_ms,
        visible=config.visible,
        manual_login=config.manual_login,
        manual_login_timeout_sec=config.manual_login_timeout_sec,
        save_html_on_error=config.save_html_on_error,
        error_html_dir=config.error_html_dir,
        output_path=config.data_dir / "temp.jsonl",
        auto_login=config.auto_login,
        username=config.username,
        password=config.password,
    )
    
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=not config.visible)
        context = browser.new_context(locale="zh-CN")
        page = context.new_page()
        
        # 自动登录
        if config.auto_login:
            print("\n[系统] 启用了自动登录，正在执行登录流程...")
            login_options = LoginOptions(
                username=config.username,
                password=config.password,
                timeout_sec=config.manual_login_timeout_sec,
                visible=config.visible,
            )
            login_result = AuthManager.perform_login(page, login_options)
            if login_result.success:
                print(f"[系统] 登录成功! 用户: {login_result.state.username}\n")
            else:
                print(f"[系统] 警告: 自动登录失败: {login_result.state.error_message}")
                print("[系统] 将继续尝试以未登录状态爬取\n")
        
        # Phase 1: 链接提取
        if config.phase in ("links", "full"):
            print("=" * 60)
            print(f"[Phase 1] 开始链接提取: {config.region_slug}")
            print("=" * 60)
            
            extractor = LinkExtractor(
                region=config.region_slug,
                base_url=region_base_url,
                data_dir=config.data_dir,
                config=m0_config,
            )
            link_result = extractor.extract(page, max_pages=config.max_pages)
            
            result["link_extraction"] = {
                "total_pages": link_result.total_pages,
                "total_links": link_result.total_links,
                "new_links": link_result.new_links,
            }
            run_log.event(
                stage="phase_links_finish",
                status="completed",
                detail=result["link_extraction"],
            )
            
            print(f"\n[Phase 1完成] 共遍历{link_result.total_pages}页，提取{link_result.total_links}条链接，新增{link_result.new_links}条")
        
        # Phase 2: 内容提取
        if config.phase in ("content", "full"):
            print("\n" + "=" * 60)
            print(f"[Phase 2] 开始内容提取: {config.region_slug}")
            print("=" * 60)
            
            # 加载链接
            links_file = config.data_dir / "links" / f"{config.region_slug}_links.json"
            if not links_file.exists():
                raise FileNotFoundError(f"链接文件不存在: {links_file}，请先运行Phase 1")
            
            existing_links = _load_existing_links(links_file)
            if not existing_links:
                raise ValueError(f"链接文件为空: {links_file}")
            
            print(f"[Phase 2] 加载到{len(existing_links)}条房源链接")
            
            extractor = ContentExtractor(
                region=config.region_slug,
                links=existing_links,
                data_dir=config.data_dir,
                config=m0_config,
                run_log=run_log,
                failure_store=failure_store,
                run_id=run_log.run_id,
            )
            content_result = extractor.extract(page, resume=config.resume)
            
            result["content_extraction"] = {
                "total_houses": content_result.total_houses,
                "processed": content_result.processed_count,
                "skipped": content_result.skipped_count,
                "communities": content_result.community_count,
                "failed": len(content_result.failed_houses),
            }
            run_log.event(
                stage="phase_content_finish",
                status="completed",
                detail=result["content_extraction"],
            )
            
            print(f"\n[Phase 2完成] 总房源{content_result.total_houses}，处理{content_result.processed_count}，跳过{content_result.skipped_count}")
            print(f"[Phase 2完成] 小区数{content_result.community_count}，失败{len(content_result.failed_houses)}")
        
        context.close()
        browser.close()
    
    return result
