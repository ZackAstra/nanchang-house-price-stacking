from __future__ import annotations

import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption("--regions", action="store", default="all", help="区域列表，all或逗号分隔")
    parser.addoption("--pages", action="store", type=int, default=10, help="每区域采集页数")
    parser.addoption("--workers", action="store", type=int, default=3, help="区域并发数，最大3")
    parser.addoption("--timeout", action="store", type=int, default=300, help="超时秒数")
    parser.addoption("--resume", action="store_true", default=False, help="是否从checkpoint断点续传")
    parser.addoption("--checkpoint-per-region", action="store_true", default=False, help="每个区域完成后立即保存checkpoint")
