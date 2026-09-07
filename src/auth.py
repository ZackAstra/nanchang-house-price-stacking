"""
登录认证模块 - 可复用的安居客登录管理器

遵循奥卡姆剃刀原则：只做必要的事，保持简单
第一性原则：登录的本质是获取有效会话

设计目标：
1. 任何异常处理模块可独立调用
2. 不依赖全局配置，纯函数式接口
3. 清晰的输入(Page)和输出(SessionState)
"""

from __future__ import annotations

from datetime import datetime
import os
import re
from time import monotonic
from typing import Callable

from playwright.sync_api import Page

from src.models.auth import LoginOptions, LoginResult, SessionState


# =============================================================================
# 登录状态检测
# =============================================================================

# 已登录页面特征
_LOGGED_IN_INDICATORS = [
    "退出",
    "个人中心", 
    "我的安居客",
    "欢迎您",
    "注销",
]

# 登录页特征
_LOGIN_PAGE_INDICATORS = [
    "账号密码登录",
    "请输入用户名",
    "请输入密码",
    "login.anjuke.com",
]

# 验证码页特征
_CAPTCHA_INDICATORS = [
    "请输入验证码",
    "/antibot/",
    "geetest",
    "微信登录",
    "antispam-block",
    "点击按钮进行验证",
]


def check_login_state(page: Page) -> SessionState:
    """
    检查当前页面的登录状态（无副作用）
    
    Args:
        page: Playwright Page实例
        
    Returns:
        SessionState: 当前会话状态
    """
    try:
        html = page.content()
        url = page.url
        
        # 检查已登录特征
        for indicator in _LOGGED_IN_INDICATORS:
            if indicator in html:
                # 尝试提取用户名
                username = _extract_username_from_page(html)
                return SessionState(
                    is_logged_in=True,
                    username=username,
                    login_time=None,  # 不知道具体登录时间
                )
        
        # 检查是否在登录页
        for indicator in _LOGIN_PAGE_INDICATORS:
            if indicator in html or indicator in url:
                return SessionState(
                    is_logged_in=False,
                    error_message="当前在登录页面",
                )
        
        # 检查是否触发验证码
        for indicator in _CAPTCHA_INDICATORS:
            if indicator in html or indicator in url:
                return SessionState(
                    is_logged_in=False,
                    error_message=f"触发验证页面: {indicator}",
                )
        
        # 启发式判断：安居客站内且无登录入口，通常为已登录
        if "anjuke.com" in url and "login.anjuke.com" not in url:
            login_entry = page.locator("a[href*='login.anjuke.com']").first
            if login_entry.count() == 0:
                username = _extract_username_from_page(html)
                return SessionState(
                    is_logged_in=True,
                    username=username,
                    login_time=None,
                )

        # 默认状态
        return SessionState(
            is_logged_in=False,
            error_message="未知状态",
        )
        
    except Exception as e:
        return SessionState(
            is_logged_in=False,
            error_message=f"状态检查失败: {e}",
        )


def _extract_username_from_page(html: str) -> str | None:
    """从页面HTML提取用户名（简单启发式）"""
    # 尝试匹配常见的用户名称示方式
    patterns = [
        r'class=["\']user-name["\'][^>]*>([^<]+)',
        r'欢迎您[,，]\s*([^<\s]+)',
        r'用户[:：]\s*([^<\s]+)',
    ]
    for pattern in patterns:
        match = re.search(pattern, html)
        if match:
            return match.group(1).strip()
    return None


# =============================================================================
# 验证按钮点击
# =============================================================================

def _try_click_verify_button(page: Page) -> bool:
    """
    尝试点击验证按钮（验证码页面的"点击按钮进行验证"）
    
    Returns:
        bool: 是否成功点击
    """
    selectors = [
        "#btnSubmit",
        "text=点击按钮进行验证",
        "button:has-text('验证')",
    ]
    for selector in selectors:
        try:
            btn = page.locator(selector).first
            if btn.count() > 0:
                btn.click(timeout=5000)
                return True
        except Exception:
            continue
    return False


def _auto_handle_captcha(page: Page, options: LoginOptions) -> bool:
    """
    自动处理验证码/验证页面
    
    Returns:
        bool: 是否成功通过验证
    """
    html = page.content()
    url = page.url
    
    # 检查是否在验证页
    is_captcha_page = any(indicator in html or indicator in url 
                          for indicator in _CAPTCHA_INDICATORS)
    if not is_captcha_page:
        return True
    
    # 尝试自动点击验证按钮
    for attempt in range(6):
        if _try_click_verify_button(page):
            page.wait_for_timeout(2000)
            # 检查是否已通过
            html = page.content()
            url = page.url
            if not any(indicator in html or indicator in url 
                       for indicator in _CAPTCHA_INDICATORS):
                return True
        else:
            break
    
    # 需要人工介入
    if not options.visible:
        return False
    
    print("[登录] 需要人工完成验证，请操作页面...")
    deadline = monotonic() + options.timeout_sec
    while monotonic() < deadline:
        page.wait_for_timeout(2000)
        html = page.content()
        url = page.url
        if not any(indicator in html or indicator in url 
                   for indicator in _CAPTCHA_INDICATORS):
            print("[登录] 人工验证完成")
            return True
    
    return False


# =============================================================================
# 核心登录流程
# =============================================================================

def navigate_to_login_page(page: Page) -> bool:
    """
    导航到登录页面
    
    1. 如果当前在首页，点击登录按钮
    2. 如果当前在其他页面，先导航到首页
    
    Returns:
        bool: 是否成功导航到登录页
    """
    try:
        url = page.url
        html = page.content()

        # 如果已经在登录页，直接返回
        if "login.anjuke.com" in url or "账号密码登录" in html:
            return True

        # 严格从首页起步，避免在任意页面误点到非顶部登录元素
        page.goto("https://nc.anjuke.com/", wait_until="domcontentloaded", timeout=30000)
        page.wait_for_timeout(1500)

        # 点击登录按钮
        login_btn_selectors = [
            "a[href*='login.anjuke.com']",
            "a:has-text('登录')",
            "[data-testid='login-button']",
            ".login-btn",
        ]
        for selector in login_btn_selectors:
            try:
                btn = page.locator(selector).first
                if btn.count() > 0:
                    btn.click(timeout=10000)
                    page.wait_for_timeout(1000)
                    if "login.anjuke.com" in page.url:
                        return True
                    # 某些场景点击登录会新开页签，这里回收到当前page，避免出现多个窗口
                    login_pages = [
                        opened_page
                        for opened_page in page.context.pages
                        if "login.anjuke.com" in opened_page.url
                    ]
                    if len(login_pages) > 0:
                        login_url = login_pages[-1].url
                        page.goto(login_url, wait_until="domcontentloaded", timeout=15000)
                        for opened_page in login_pages:
                            if opened_page != page:
                                opened_page.close()
                        return "login.anjuke.com" in page.url
            except Exception:
                continue

        # 兜底：显式跳转登录页，避免因为首页元素结构变化导致卡住
        page.goto(
            "https://login.anjuke.com/login/form?history=aHR0cHM6Ly9uYy5hbmp1a2UuY29tLw==",
            wait_until="domcontentloaded",
            timeout=15000,
        )
        return "login.anjuke.com" in page.url

        return False
        
    except Exception as e:
        print(f"[登录] 导航到登录页失败: {e}")
        return False


def fill_login_form(page: Page, options: LoginOptions) -> bool:
    """
    填写登录表单
    
    Args:
        page: 当前应在登录页
        options: 登录选项
        
    Returns:
        bool: 是否成功填写
    """
    try:
        # 切换到账号密码登录选项卡（如果需要）
        account_tab = page.locator("text=账号密码登录").first
        if account_tab.count() > 0:
            account_tab.click()
            page.wait_for_timeout(500)
        
        # 填写账号
        username_input = page.locator(
            "input[placeholder*='用户名'], input[placeholder*='手机号'], "
            "input[name='username'], #username"
        ).first
        if username_input.count() == 0:
            username_input = page.locator("input[type='text']").first
        
        if username_input.count() > 0:
            username_input.fill(options.username)
        else:
            print("[登录] 警告: 未找到账号输入框")
            return False
        
        # 填写密码
        password_input = page.locator(
            "input[type='password'], input[name='password'], #password"
        ).first
        if password_input.count() > 0:
            password_input.fill(options.password)
        else:
            print("[登录] 警告: 未找到密码输入框")
            return False
        
        return True
        
    except Exception as e:
        print(f"[登录] 填写表单失败: {e}")
        return False


def submit_login_form(page: Page) -> bool:
    """
    提交登录表单
    
    Returns:
        bool: 是否成功提交
    """
    try:
        # 勾选协议checkbox（如有）
        privacy_checkbox = page.locator("div.privacy i.icon-checkbox").first
        if privacy_checkbox.count() > 0:
            privacy_checkbox.click()
        checkbox = page.locator("input[type='checkbox']").first
        if checkbox.count() > 0:
            checkbox.check()
        else:
            protocol_check = page.locator(
                "[role='checkbox'], .checkbox, .check, .agreement-checkbox"
            ).first
            if protocol_check.count() > 0:
                protocol_check.click()
        
        # 点击登录按钮（优先命中登录表单真实提交节点）
        login_btn = page.locator("div.submit").first
        if login_btn.count() == 0:
            login_btn = page.locator(
                "button[type='submit'], .login-submit-btn, button:has-text('登录')"
            ).first
        if login_btn.count() == 0:
            login_btn = page.locator("div:has-text('登录')").first
        if login_btn.count() > 0:
            login_btn.click(timeout=5000)
            return True
        
        print("[登录] 错误: 未找到登录按钮")
        return False
        
    except Exception as e:
        print(f"[登录] 提交表单失败: {e}")
        return False


def wait_for_login_complete(page: Page, options: LoginOptions) -> SessionState:
    """
    等待登录完成，处理协议弹窗/验证码等
    
    Returns:
        SessionState: 最终会话状态
    """
    start_time = monotonic()
    
    while monotonic() - start_time < options.timeout_sec:
        current_url = page.url
        html = page.content()

        # 协议确认弹窗可能在登录页出现，优先处理
        agree_btn = page.locator("text=同意并继续").first
        if agree_btn.count() > 0:
            try:
                agree_btn.click(timeout=2000)
            except Exception:
                page.wait_for_timeout(500)
            page.wait_for_timeout(1000)
            continue
        
        # 检查是否已登录成功
        if "login" not in current_url and "anjuke.com" in current_url:
            # 再次检查登录状态
            state = check_login_state(page)
            if state.is_logged_in:
                return SessionState(
                    is_logged_in=True,
                    username=options.username,
                    login_time=datetime.now(),
                )
        
        # 检查是否触发验证码
        if any(indicator in html or indicator in current_url 
               for indicator in _CAPTCHA_INDICATORS):
            if not _auto_handle_captcha(page, options):
                return SessionState(
                    is_logged_in=False,
                    error_message="验证码处理失败或超时",
                )
            continue
        
        page.wait_for_timeout(1000)
    
    return SessionState(
        is_logged_in=False,
        error_message=f"登录超时({options.timeout_sec}秒)",
    )


# =============================================================================
# 对外接口
# =============================================================================

class AuthManager:
    """
    可复用的登录管理器
    
    使用示例:
        # 1. 确保登录（推荐）
        options = LoginOptions(username="xxx", password="xxx")
        state = AuthManager.ensure_login(page, options)
        
        # 2. 仅检查状态
        state = AuthManager.check_login_state(page)
        
        # 3. 强制重新登录
        state = AuthManager.perform_login(page, options)
    """
    
    @staticmethod
    def check_login_state(page: Page) -> SessionState:
        """检查当前登录状态（无副作用）"""
        return check_login_state(page)
    
    @staticmethod
    def perform_login(
        page: Page,
        options: LoginOptions,
        force: bool = False
    ) -> LoginResult:
        """
        执行登录流程
        
        Args:
            page: Playwright Page实例
            options: 登录选项
            force: 是否强制重新登录（即使已登录）
            
        Returns:
            LoginResult: 登录结果
        """
        start_time = monotonic()
        retry_count = 0
        
        print("=" * 60)
        print("开始执行安居客登录")
        print(f"账号: {options.username}")
        print("=" * 60)
        
        # 检查是否已登录
        if not force:
            current_state = check_login_state(page)
            if current_state.is_logged_in:
                print(f"[登录] 检测到已处于登录状态，跳过登录流程")
                return LoginResult(
                    success=True,
                    state=current_state,
                    retry_count=0,
                    elapsed_sec=0.0,
                )
        
        # 导航到登录页
        print("[登录] 导航到登录页面...")
        if not navigate_to_login_page(page):
            return LoginResult(
                success=False,
                state=SessionState(
                    is_logged_in=False,
                    error_message="导航到登录页失败",
                ),
                retry_count=retry_count,
                elapsed_sec=monotonic() - start_time,
            )
        
        # 填写表单
        print("[登录] 填写登录表单...")
        if not fill_login_form(page, options):
            return LoginResult(
                success=False,
                state=SessionState(
                    is_logged_in=False,
                    error_message="填写登录表单失败",
                ),
                retry_count=retry_count,
                elapsed_sec=monotonic() - start_time,
            )
        
        # 提交表单
        print("[登录] 提交登录表单...")
        if not submit_login_form(page):
            return LoginResult(
                success=False,
                state=SessionState(
                    is_logged_in=False,
                    error_message="提交登录表单失败",
                ),
                retry_count=retry_count,
                elapsed_sec=monotonic() - start_time,
            )
        
        # 等待登录完成
        print("[登录] 等待登录完成...")
        final_state = wait_for_login_complete(page, options)
        
        elapsed = monotonic() - start_time
        
        if final_state.is_logged_in:
            print(f"[登录] 登录成功! 耗时: {elapsed:.1f}秒")
            print("=" * 60)
            return LoginResult(
                success=True,
                state=SessionState(
                    is_logged_in=True,
                    username=options.username,
                    login_time=datetime.now(),
                ),
                retry_count=retry_count,
                elapsed_sec=elapsed,
            )
        else:
            print(f"[登录] 登录失败: {final_state.error_message}")
            print("=" * 60)
            return LoginResult(
                success=False,
                state=final_state,
                retry_count=retry_count,
                elapsed_sec=elapsed,
            )
    
    @staticmethod
    def ensure_login(
        page: Page,
        options: LoginOptions,
        on_auth_error: Callable[[Page, LoginOptions], None] | None = None
    ) -> SessionState:
        """
        确保页面处于登录状态（最常用接口）
        
        如果未登录，自动执行登录流程
        
        Args:
            page: Playwright Page实例
            options: 登录选项
            on_auth_error: 认证失败时的回调函数
            
        Returns:
            SessionState: 最终会话状态
        """
        # 先检查状态
        state = check_login_state(page)
        if state.is_logged_in:
            return state
        
        # 需要登录
        result = AuthManager.perform_login(page, options)
        
        if not result.success and on_auth_error:
            on_auth_error(page, options)
        
        return result.state


# =============================================================================
# 便捷函数（向后兼容）
# =============================================================================

def perform_login(
    page: Page,
    username: str = "",
    password: str = "",
    visible: bool = True,
    manual_login_timeout_sec: int = 180,
) -> bool:
    """
    向后兼容的登录函数（供旧代码使用）
    
    内部调用 AuthManager.perform_login
    """
    options = LoginOptions(
        username=username or os.environ.get("ANJUKE_USERNAME", ""),
        password=password or os.environ.get("ANJUKE_PASSWORD", ""),
        timeout_sec=manual_login_timeout_sec,
        visible=visible,
    )
    result = AuthManager.perform_login(page, options)
    return result.success


# 为了兼容scheduler.py中的导入
def _is_login_or_captcha_page(html: str) -> bool:
    """检查是否是登录或验证码页面"""
    return any(indicator in html for indicator in _CAPTCHA_INDICATORS)


def _try_click_human_verify_button(page: Page) -> bool:
    """尝试点击人工验证按钮"""
    return _try_click_verify_button(page)
