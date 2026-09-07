"""
认证相关数据模型
"""

from __future__ import annotations

import os
from datetime import datetime
from pydantic import BaseModel, Field


class LoginOptions(BaseModel):
    """登录配置选项（账号密码默认从环境变量 ANJUKE_USERNAME/ANJUKE_PASSWORD 读取）"""
    username: str = Field(default=os.environ.get("ANJUKE_USERNAME", ""), description="登录账号")
    password: str = Field(default=os.environ.get("ANJUKE_PASSWORD", ""), description="登录密码")
    timeout_sec: int = Field(default=180, ge=30, le=1800, description="登录超时时间")
    visible: bool = Field(default=True, description="是否可视化模式（影响验证码处理）")
    
    class Config:
        frozen = True


class SessionState(BaseModel):
    """会话状态"""
    is_logged_in: bool = Field(default=False, description="是否已登录")
    username: str | None = Field(default=None, description="当前登录用户名")
    login_time: datetime | None = Field(default=None, description="登录时间")
    session_id: str | None = Field(default=None, description="会话标识（如有）")
    error_message: str | None = Field(default=None, description="错误信息")
    
    class Config:
        frozen = True


class LoginResult(BaseModel):
    """登录操作结果"""
    success: bool
    state: SessionState
    retry_count: int = 0
    elapsed_sec: float = 0.0
