from __future__ import annotations

import argparse
from pathlib import Path
from time import monotonic

from playwright.sync_api import Page
from playwright.sync_api import sync_playwright
from pydantic import BaseModel
from pydantic import Field
from pydantic import HttpUrl


class PlaywrightDebugConfig(BaseModel):
    target_url: HttpUrl = Field(description="待调试页面URL")
    wait_after_load_ms: int = Field(gt=0, le=60000, description="加载后等待毫秒数")
    visible: bool = Field(description="是否显示浏览器窗口")
    open_inspector: bool = Field(description="是否打开Playwright Inspector")
    manual_login: bool = Field(description="是否允许人工扫码登录")
    manual_login_timeout_sec: int = Field(ge=30, le=1800, description="人工扫码超时秒数")
    output_html_path: Path = Field(description="页面快照HTML输出路径")


def collect_page_metrics(page: Page) -> dict[str, int]:
    metrics: dict[str, int] = {
        "prop_link_count": page.locator("a[href*='/prop/view/']").count(),
        "community_link_count": page.locator("a[href*='/community/view/']").count(),
        "tab_box_count": page.locator("div.tabBox").count(),
        "tab_item_count": page.locator("div.tabBox li").count(),
    }
    return metrics


def _is_login_or_captcha_page(html: str) -> bool:
    return "请输入验证码" in html or "/antibot/" in html or "geetest" in html or "微信登录" in html


def _try_click_human_verify_button(page: Page) -> bool:
    verify_button = page.locator("#btnSubmit")
    if verify_button.count() == 0:
        return False
    verify_button.first.click(timeout=5000)
    return True


def _wait_for_manual_login(page: Page, config: PlaywrightDebugConfig) -> None:
    if not config.visible:
        raise RuntimeError("触发登录/验证码页但visible=false，无法人工扫码")
    if not config.manual_login:
        raise RuntimeError("触发登录/验证码页且manual-login=false")
    clicked_verify_button = _try_click_human_verify_button(page)
    if clicked_verify_button:
        print("已自动点击验证码页面“点击按钮进行验证”。")
    print("检测到登录/验证码页面，请完成人工验证。")
    deadline = monotonic() + config.manual_login_timeout_sec
    while monotonic() < deadline:
        page.wait_for_timeout(2000)
        current_html = page.content()
        if not _is_login_or_captcha_page(current_html):
            print("扫码验证完成，继续执行。")
            return
    raise RuntimeError(f"人工扫码超时，timeout_sec={config.manual_login_timeout_sec}")


def run_debug(config: PlaywrightDebugConfig) -> None:
    with sync_playwright() as playwright:
        browser = None
        launch_errors: list[str] = []
        try:
            browser = playwright.chromium.launch(headless=not config.visible)
        except Exception as error:
            launch_errors.append(f"chromium: {error}")
        if browser is None:
            try:
                browser = playwright.chromium.launch(channel="chrome", headless=not config.visible)
            except Exception as error:
                launch_errors.append(f"chrome: {error}")
        if browser is None:
            try:
                browser = playwright.chromium.launch(channel="msedge", headless=not config.visible)
            except Exception as error:
                launch_errors.append(f"msedge: {error}")
        if browser is None:
            error_text = "; ".join(launch_errors)
            raise RuntimeError(f"浏览器启动失败, errors={error_text}")
        context = browser.new_context(locale="zh-CN")
        page = context.new_page()
        page.goto(str(config.target_url), wait_until="domcontentloaded", timeout=60000)
        page.wait_for_timeout(config.wait_after_load_ms)
        page_html = page.content()
        if _is_login_or_captcha_page(page_html):
            _wait_for_manual_login(page, config)
            page.wait_for_timeout(config.wait_after_load_ms)
        if config.open_inspector:
            page.pause()

        metrics = collect_page_metrics(page)
        html_content = page.content()
        config.output_html_path.parent.mkdir(parents=True, exist_ok=True)
        config.output_html_path.write_text(html_content, encoding="utf-8")

        print("页面调试完成")
        print(f"URL: {config.target_url}")
        print(f"房源详情链接数量: {metrics['prop_link_count']}")
        print(f"小区详情链接数量: {metrics['community_link_count']}")
        print(f"tabBox容器数量: {metrics['tab_box_count']}")
        print(f"tabBox子项数量: {metrics['tab_item_count']}")
        print(f"页面HTML快照: {config.output_html_path}")

        context.close()
        browser.close()


def parse_cli_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Playwright页面调试工具")
    parser.add_argument("--url", required=True, help="目标URL，例如 https://nc.anjuke.com/sale/honggutan/")
    parser.add_argument("--wait-ms", required=True, type=int, help="页面加载后额外等待毫秒数，例如 2500")
    parser.add_argument("--visible", required=True, choices=["true", "false"], help="是否显示浏览器窗口")
    parser.add_argument("--inspector", required=True, choices=["true", "false"], help="是否打开Inspector")
    parser.add_argument("--manual-login", required=True, choices=["true", "false"], help="是否允许人工扫码登录")
    parser.add_argument("--manual-login-timeout-sec", required=True, type=int, help="人工扫码最大等待秒数")
    parser.add_argument("--output-html", required=True, help="页面HTML快照输出路径")
    args = parser.parse_args()
    return args


def main() -> None:
    args = parse_cli_args()
    config = PlaywrightDebugConfig(
        target_url=args.url,
        wait_after_load_ms=args.wait_ms,
        visible=args.visible == "true",
        open_inspector=args.inspector == "true",
        manual_login=args.manual_login == "true",
        manual_login_timeout_sec=args.manual_login_timeout_sec,
        output_html_path=Path(args.output_html),
    )
    run_debug(config)


if __name__ == "__main__":
    main()
