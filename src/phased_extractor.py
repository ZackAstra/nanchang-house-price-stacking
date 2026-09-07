"""
分阶段提取器 - Phase 1: 链接提取, Phase 2: 内容提取

遵循奥卡姆剃刀原则：只做必要的事，保持简单
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import urljoin
from html import unescape

from playwright.sync_api import Page
from pydantic import BaseModel, Field

from src.parsers.list_parser import parse_list_page
from src.parsers.prop_parser import parse_prop_page
from src.parsers.comm_parser import parse_community_page
from src.parsers.poi_parser import extract_poi_details, extract_poi_summary
from src.scheduler import (
    M0RunConfig,
    _goto_and_get_html,
    _safe_page_content,
    _save_error_html,
    _normalize_community_url,
    _collect_poi_details_with_ui,
    _build_poi_summary_from_details,
)
from src.storage.failure_store import FailureStore
from src.storage.jsonl_storage import append_jsonl, write_jsonl
from src.storage.run_log_store import RunLogStore


# =============================================================================
# 数据模型
# =============================================================================

class HouseLink(BaseModel):
    """房源链接信息"""
    house_id: str
    url: str
    title: str | None = None
    list_page_no: int
    extracted_at: str = Field(default_factory=lambda: datetime.now().isoformat())


class LayoutInfo(BaseModel):
    """房源房型信息（从列表页提取）"""
    house_id: str
    rooms: int | None = None
    halls: int | None = None
    baths: int | None = None
    area: float | None = None
    price: float | None = None
    community_name: str | None = None


class LinkExtractionResult(BaseModel):
    """链接提取结果"""
    region: str
    total_pages: int
    total_links: int
    new_links: int
    links: list[HouseLink]
    layouts: dict[str, LayoutInfo] = Field(default_factory=dict)


class ContentCheckpoint(BaseModel):
    """内容提取断点"""
    region: str
    processed_houses: list[str] = Field(default_factory=list)
    processed_communities: list[str] = Field(default_factory=list)
    last_updated: str = Field(default_factory=lambda: datetime.now().isoformat())


class ContentExtractionResult(BaseModel):
    """内容提取结果"""
    region: str
    total_houses: int
    processed_count: int
    skipped_count: int
    community_count: int
    failed_houses: list[str] = Field(default_factory=list)


# =============================================================================
# 工具函数
# =============================================================================

def _ensure_dirs(base_dir: Path) -> tuple[Path, Path, Path]:
    """确保目录结构存在"""
    links_dir = base_dir / "links"
    checkpoints_dir = base_dir / "checkpoints"
    output_dir = base_dir / "output"
    
    links_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    return links_dir, checkpoints_dir, output_dir


def _extract_house_id_from_url(url: str) -> str | None:
    """从房源URL提取房源ID"""
    import re
    match = re.search(r"/prop/view/([^/?#]+)", url)
    return match.group(1) if match else None


def _load_existing_links(links_file: Path) -> list[HouseLink]:
    """加载已保存的链接"""
    if not links_file.exists():
        return []
    try:
        data = json.loads(links_file.read_text(encoding="utf-8"))
        return [HouseLink.model_validate(item) for item in data.get("links", [])]
    except Exception:
        return []


def _load_existing_layouts(layouts_file: Path) -> dict[str, LayoutInfo]:
    """加载已保存的房型信息"""
    if not layouts_file.exists():
        return {}
    try:
        data = json.loads(layouts_file.read_text(encoding="utf-8"))
        return {
            k: LayoutInfo.model_validate(v) 
            for k, v in data.get("layouts", {}).items()
        }
    except Exception:
        return {}


def _load_content_checkpoint(checkpoint_file: Path) -> ContentCheckpoint:
    """加载内容提取断点"""
    if not checkpoint_file.exists():
        return ContentCheckpoint(region="")
    try:
        data = json.loads(checkpoint_file.read_text(encoding="utf-8"))
        return ContentCheckpoint.model_validate(data)
    except Exception:
        return ContentCheckpoint(region="")


def _save_content_checkpoint(checkpoint: ContentCheckpoint, checkpoint_file: Path) -> None:
    """保存内容提取断点"""
    checkpoint.last_updated = datetime.now().isoformat()
    checkpoint_file.write_text(
        json.dumps(checkpoint.model_dump(), ensure_ascii=False, indent=2),
        encoding="utf-8"
    )


# =============================================================================
# Phase 1: 链接提取器
# =============================================================================

class LinkExtractor:
    """
    链接提取器 - 负责从二级页面提取所有房源链接
    
    特性：
    - 增量对比：与已有链接对比，仅保存新增
    - 实时保存：每页提取后立即保存
    - 支持断点：可从中断处继续（通过增量对比实现）
    """
    
    def __init__(
        self,
        region: str,
        base_url: str,
        data_dir: Path,
        config: M0RunConfig,
    ):
        self.region = region
        self.base_url = base_url
        self.config = config
        
        links_dir, _, _ = _ensure_dirs(data_dir)
        self.links_file = links_dir / f"{region}_links.json"
        self.layouts_file = links_dir / f"{region}_layouts.json"
    
    def extract(
        self,
        page: Page,
        max_pages: int | None = None,
    ) -> LinkExtractionResult:
        """
        提取链接主流程
        
        Args:
            page: Playwright Page实例
            max_pages: 最大页数限制，None表示无限制
        
        Returns:
            链接提取结果
        """
        # 1. 加载已有链接
        existing_links = _load_existing_links(self.links_file)
        existing_ids = {link.house_id for link in existing_links}
        existing_layouts = _load_existing_layouts(self.layouts_file)
        
        print(f"[链接提取] 已存在链接: {len(existing_links)}条")
        
        all_links: list[HouseLink] = existing_links.copy()
        all_layouts: dict[str, LayoutInfo] = existing_layouts.copy()
        new_count = 0
        page_no = 0
        
        while True:
            page_no += 1
            if max_pages and page_no > max_pages:
                print(f"[链接提取] 达到最大页数限制: {max_pages}")
                break
            
            # 构建页面URL
            if page_no == 1:
                page_url = self.base_url
            else:
                page_url = f"{self.base_url.rstrip('/')}/p{page_no}/"
            
            print(f"[链接提取] 正在处理第{page_no}页: {page_url}")
            
            try:
                # 获取页面HTML
                html = _goto_and_get_html(
                    page=page,
                    url=page_url,
                    config=self.config,
                    page_no=page_no,
                    stage="link_list",
                )
                
                # 解析列表页
                parsed = parse_list_page(self.base_url, html, self.region, page_no)
                
                if not parsed.prop_urls:
                    print(f"[链接提取] 第{page_no}页无房源链接，可能已到达最后一页")
                    break
                
                # 处理本页链接
                page_new_count = 0
                for url in parsed.prop_urls:
                    house_id = _extract_house_id_from_url(str(url))
                    if not house_id:
                        continue
                    
                    # 增量对比：只处理新链接
                    if house_id in existing_ids:
                        continue
                    
                    link = HouseLink(
                        house_id=house_id,
                        url=str(url),
                        list_page_no=page_no,
                    )
                    all_links.append(link)
                    existing_ids.add(house_id)
                    new_count += 1
                    page_new_count += 1
                
                print(f"[链接提取] 第{page_no}页新增: {page_new_count}条")
                
                # 实时保存（每页后保存）
                self._save_results(all_links, all_layouts)
                
                # 简单判断是否还有下一页（通过检查是否返回了60条，安居客每页60条）
                if len(parsed.prop_urls) < 60:
                    print(f"[链接提取] 本页仅{len(parsed.prop_urls)}条，可能为最后一页")
                    # 继续尝试下一页确认
                    
            except Exception as e:
                print(f"[链接提取] 第{page_no}页处理失败: {e}")
                # 保存当前进度后继续
                self._save_results(all_links, all_layouts)
                # 连续失败3次则停止
                if page_no > 3:
                    break
                continue
        
        # 最终保存
        self._save_results(all_links, all_layouts)
        
        return LinkExtractionResult(
            region=self.region,
            total_pages=page_no,
            total_links=len(all_links),
            new_links=new_count,
            links=all_links,
            layouts=all_layouts,
        )
    
    def _save_results(
        self,
        links: list[HouseLink],
        layouts: dict[str, LayoutInfo],
    ) -> None:
        """保存结果到JSON文件"""
        # 保存链接
        links_data = {
            "region": self.region,
            "total": len(links),
            "updated_at": datetime.now().isoformat(),
            "links": [link.model_dump() for link in links],
        }
        self.links_file.write_text(
            json.dumps(links_data, ensure_ascii=False, indent=2),
            encoding="utf-8"
        )
        
        # 保存房型信息
        layouts_data = {
            "region": self.region,
            "total": len(layouts),
            "updated_at": datetime.now().isoformat(),
            "layouts": {k: v.model_dump() for k, v in layouts.items()},
        }
        self.layouts_file.write_text(
            json.dumps(layouts_data, ensure_ascii=False, indent=2),
            encoding="utf-8"
        )


# =============================================================================
# Phase 2: 内容提取器
# =============================================================================

class ContentExtractor:
    """
    内容提取器 - 负责从三级和四级页面提取详细信息
    
    特性：
    - 断点接续：跳过已处理的房源
    - 小区去重：同一小区只解析一次
    - 实时保存：每处理N条后保存断点
    """
    
    def __init__(
        self,
        region: str,
        links: list[HouseLink],
        data_dir: Path,
        config: M0RunConfig,
        run_log: RunLogStore | None = None,
        failure_store: FailureStore | None = None,
        run_id: str | None = None,
    ):
        self.region = region
        self.links = links
        self.config = config
        self.run_log = run_log
        self.failure_store = failure_store
        self.run_id = run_id
        
        _, checkpoints_dir, output_dir = _ensure_dirs(data_dir)
        self.checkpoint_file = checkpoints_dir / f"{region}_content_checkpoint.json"
        self.houses_output = output_dir / f"{region}_houses.jsonl"
        self.communities_output = output_dir / f"{region}_communities.jsonl"
    
    def extract(
        self,
        page: Page,
        resume: bool = True,
        save_interval: int = 10,
    ) -> ContentExtractionResult:
        """
        提取内容主流程
        
        Args:
            page: Playwright Page实例
            resume: 是否启用断点接续
            save_interval: 每处理N条后保存断点
        
        Returns:
            内容提取结果
        """
        # 1. 加载断点
        checkpoint = _load_content_checkpoint(self.checkpoint_file)
        if checkpoint.region != self.region:
            checkpoint = ContentCheckpoint(region=self.region)
        
        processed_houses = set(checkpoint.processed_houses) if resume else set()
        processed_communities = set(checkpoint.processed_communities) if resume else set()
        
        print(f"[内容提取] 已处理房源: {len(processed_houses)}条")
        print(f"[内容提取] 已处理小区: {len(processed_communities)}个")
        
        failed_houses: list[str] = []
        processed_count = 0
        skipped_count = 0
        
        for idx, link in enumerate(self.links):
            house_id = link.house_id
            
            # 断点接续：跳过已处理
            if house_id in processed_houses:
                skipped_count += 1
                if skipped_count % 100 == 0:
                    print(f"[内容提取] 已跳过{skipped_count}条已处理房源")
                continue
            
            print(f"[内容提取] [{idx+1}/{len(self.links)}] 处理房源: {house_id}")
            if self.run_log is not None:
                self.run_log.inc("prop", total=1, success=0, failed=0)
            
            try:
                # 解析三级页面（房源详情）
                house_data = self._extract_house(page, link)
                
                if not house_data:
                    failed_houses.append(house_id)
                    if self.run_log is not None:
                        self.run_log.inc("prop", total=0, success=0, failed=1)
                    if self.failure_store is not None:
                        self.failure_store.append_prop_failure(
                            run_id=self.run_id,
                            phase="content",
                            page_no=link.list_page_no,
                            target_url=link.url,
                            house_id=house_id,
                            error_type="prop_extract_none",
                            error_message="房源解析返回空结果",
                            community_id=None,
                        )
                    continue
                
                # 提取小区信息
                community_id = house_data.get("community_id")
                community_url = house_data.get("community_url")
                
                if community_id and community_url:
                    # 检查小区是否已处理
                    if community_id not in processed_communities:
                        if self.run_log is not None:
                            self.run_log.inc("community", total=1, success=0, failed=0)
                        # 解析四级页面（小区详情）
                        community_data = self._extract_community(
                            page, community_id, community_url
                        )
                        if community_data:
                            # 保存小区信息
                            append_jsonl([community_data], self.communities_output)
                            processed_communities.add(community_id)
                            print(f"[内容提取] 新增小区: {community_id}")
                            if self.run_log is not None:
                                self.run_log.inc("community", total=0, success=1, failed=0)
                        else:
                            if self.run_log is not None:
                                self.run_log.inc("community", total=0, success=0, failed=1)
                            if self.failure_store is not None:
                                self.failure_store.append_community_failure(
                                    run_id=self.run_id,
                                    phase="content",
                                    page_no=link.list_page_no,
                                    target_url=community_url,
                                    house_id=house_id,
                                    community_id=community_id,
                                    error_type="community_extract_none",
                                    error_message="小区解析返回空结果",
                                )
                
                # 保存房源信息
                append_jsonl([house_data], self.houses_output)
                processed_houses.add(house_id)
                processed_count += 1
                if self.run_log is not None:
                    self.run_log.inc("prop", total=0, success=1, failed=0)
                
                # 定期保存断点
                if processed_count % save_interval == 0:
                    checkpoint.processed_houses = list(processed_houses)
                    checkpoint.processed_communities = list(processed_communities)
                    _save_content_checkpoint(checkpoint, self.checkpoint_file)
                    print(f"[内容提取] 已保存断点，当前进度: {processed_count}/{len(self.links)}")
                    if self.run_log is not None:
                        self.run_log.inc("checkpoint", total=1, success=1, failed=0)
                
            except Exception as e:
                print(f"[内容提取] 房源{house_id}处理失败: {e}")
                failed_houses.append(house_id)
                if self.run_log is not None:
                    self.run_log.inc("prop", total=0, success=0, failed=1)
                if self.failure_store is not None:
                    self.failure_store.append_prop_failure(
                        run_id=self.run_id,
                        phase="content",
                        page_no=link.list_page_no,
                        target_url=link.url,
                        house_id=house_id,
                        error_type="prop_extract_exception",
                        error_message=str(e),
                        community_id=None,
                    )
                continue
        
        # 最终保存断点
        checkpoint.processed_houses = list(processed_houses)
        checkpoint.processed_communities = list(processed_communities)
        _save_content_checkpoint(checkpoint, self.checkpoint_file)
        if self.run_log is not None:
            self.run_log.inc("checkpoint", total=1, success=1, failed=0)
        
        return ContentExtractionResult(
            region=self.region,
            total_houses=len(self.links),
            processed_count=processed_count,
            skipped_count=skipped_count,
            community_count=len(processed_communities),
            failed_houses=failed_houses,
        )
    
    def _extract_house(self, page: Page, link: HouseLink) -> dict[str, Any] | None:
        """提取单个房源信息（三级页面）"""
        try:
            html = _goto_and_get_html(
                page=page,
                url=link.url,
                config=self.config,
                page_no=link.list_page_no,
                stage="prop_detail",
            )
            
            house = parse_prop_page(link.url, html, self.region)
            data = house.model_dump(mode="json")
            data["list_page_no"] = link.list_page_no
            data["extracted_at"] = datetime.now().isoformat()
            
            return data
            
        except Exception as e:
            print(f"[内容提取] 房源解析失败 {link.house_id}: {e}")
            return None
    
    def _extract_community(
        self,
        page: Page,
        community_id: str,
        community_url: str,
    ) -> dict[str, Any] | None:
        """提取单个小区信息（四级页面）"""
        try:
            normalized_url = _normalize_community_url(community_url)
            
            html = _goto_and_get_html(
                page=page,
                url=normalized_url,
                config=self.config,
                page_no=0,  # 小区页面无页码概念
                stage="community_detail",
            )
            
            # 提取配套信息
            tab_item_count = page.locator("div.tabBox li").count()
            poi_details = _collect_poi_details_with_ui(page)
            poi_summary = _build_poi_summary_from_details(poi_details)
            
            # 如果UI提取为空，回退到HTML提取
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
            
            data = community.model_dump(mode="json")
            data["extracted_at"] = datetime.now().isoformat()
            
            return data
            
        except Exception as e:
            print(f"[内容提取] 小区解析失败 {community_id}: {e}")
            return None
