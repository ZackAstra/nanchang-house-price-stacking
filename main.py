"""
南昌安居客二手房数据采集系统 - 主入口

支持两种模式：
1. M0模式（原模式）: 单进程顺序采集
2. 分阶段模式（M2优化）: Phase 1链接提取 + Phase 2内容提取

默认使用分阶段模式，可通过 --mode 参数切换
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path

from src.scheduler import M0RunConfig, run_m0, PhasedRunConfig, run_phased
from src.retry_failed import RetryRunConfig, run_retry_failures


def parse_cli_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="南昌安居客二手房数据采集系统",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 分阶段模式 - 完整流程（默认）
  python main.py --region honggutan --mode phased --phase full
  
  # 分阶段模式 - 仅提取链接
  python main.py --region honggutan --mode phased --phase links --max-pages 10
  
  # 分阶段模式 - 仅提取内容（自动读取已有链接）
  python main.py --region honggutan --mode phased --phase content
  
  # M0模式 - 传统单进程采集
  python main.py --region honggutan --mode m0 --pages 1 --max-props 5
  
  # 禁用断点接续（重新全量采集）
  python main.py --region honggutan --no-resume
        """
    )
    
    # 通用参数
    parser.add_argument("--region", required=True, help="区域slug，如 honggutan")
    parser.add_argument("--mode", default="phased", choices=["m0", "phased"], 
                        help="运行模式: m0(传统), phased(分阶段，默认)")
    parser.add_argument("--visible", default="true", choices=["true", "false"], 
                        help="是否显示浏览器窗口")
    parser.add_argument("--wait-ms", default=2500, type=int, help="页面加载后等待毫秒数")
    
    # M0模式参数
    m0_group = parser.add_argument_group("M0模式参数")
    m0_group.add_argument("--pages", type=int, help="M0模式: 采集页数")
    m0_group.add_argument("--max-props", type=int, help="M0模式: 每页最多处理房源数")
    m0_group.add_argument("--output", help="M0模式: JSONL输出路径")
    
    # 分阶段模式参数
    phased_group = parser.add_argument_group("分阶段模式参数")
    phased_group.add_argument("--phase", default="full", choices=["links", "content", "full", "retry-failures"],
                             help="运行阶段: links(仅链接), content(仅内容), full(完整)")
    phased_group.add_argument("--max-pages", type=int, help="链接提取最大页数限制")
    phased_group.add_argument("--no-resume", action="store_true", 
                             help="禁用断点接续（默认启用）")
    phased_group.add_argument("--retry-file", help="失败清单路径（phase=retry-failures时必填）")
    phased_group.add_argument("--retry-type", default="auto", choices=["prop", "community", "auto"],
                             help="失败重试类型（phase=retry-failures）")
    phased_group.add_argument("--retry-limit", type=int, help="失败重试条数上限（phase=retry-failures）")
    
    # 登录参数
    login_group = parser.add_argument_group("登录参数")
    login_group.add_argument("--auto-login", default="true", choices=["true", "false"],
                            help="是否在爬取前自动登录 (default: true)")
    login_group.add_argument("--username", default=os.environ.get("ANJUKE_USERNAME", ""),
                            help="登录账号（默认读环境变量 ANJUKE_USERNAME）")
    login_group.add_argument("--password", default=os.environ.get("ANJUKE_PASSWORD", ""),
                            help="登录密码（默认读环境变量 ANJUKE_PASSWORD）")
    login_group.add_argument("--manual-login", default="true", choices=["true", "false"],
                            help="是否启用人工扫码登录兜底")
    login_group.add_argument("--manual-login-timeout-sec", default=180, type=int,
                            help="人工扫码登录最大等待秒数")
    
    # 调试参数
    debug_group = parser.add_argument_group("调试参数")
    debug_group.add_argument("--save-html-on-error", default="true", choices=["true", "false"],
                            help="异常页面是否保存HTML")
    debug_group.add_argument("--error-html-dir", default="data/debug/error_html",
                            help="异常页面HTML保存目录")
    debug_group.add_argument("--data-dir", default="data", help="数据根目录")
    debug_group.add_argument("--run-id", help="任务运行ID（可选，不填自动生成）")
    
    return parser.parse_args()


def _build_timestamped_output_path(raw_output_path: Path) -> Path:
    timestamp_text = datetime.now().strftime("%Y%m%d%H%M%S")
    return raw_output_path.with_name(f"{raw_output_path.stem}_{timestamp_text}{raw_output_path.suffix}")


def run_m0_mode(args: argparse.Namespace, region_map: dict[str, str]) -> None:
    """运行M0模式"""
    if not args.pages or not args.max_props:
        print("错误: M0模式需要 --pages 和 --max-props 参数")
        return
    
    output_path = _build_timestamped_output_path(
        Path(args.output or f"data/m0_{args.region}.jsonl")
    )
    
    run_config = M0RunConfig(
        region_slug=args.region,
        pages=args.pages,
        max_props=args.max_props,
        wait_ms=args.wait_ms,
        visible=args.visible == "true",
        manual_login=args.manual_login == "true",
        manual_login_timeout_sec=args.manual_login_timeout_sec,
        save_html_on_error=args.save_html_on_error == "true",
        error_html_dir=Path(args.error_html_dir),
        output_path=output_path,
        auto_login=args.auto_login == "true",
        username=args.username,
        password=args.password,
        run_id=args.run_id,
    )
    
    print("=" * 60)
    print("运行模式: M0 (传统单进程)")
    print(f"区域: {args.region}")
    print(f"页数: {args.pages}, 每页房源: {args.max_props}")
    print("=" * 60)
    
    result = run_m0(run_config, region_map)
    
    print("\n" + "=" * 60)
    print("M0链路执行完成")
    print(f"输出文件: {output_path}")
    print(f"小区文件: {result['community_output_path']}")
    print(f"输出记录数: {result['written_records']}")
    print(f"去重小区数: {result['unique_community_count']}")
    print(f"三级页面失败数: {result['prop_failed']}")
    print(f"四级页面失败数: {result['community_failed']}")
    print("=" * 60)


def run_phased_mode(args: argparse.Namespace, region_map: dict[str, str]) -> None:
    """运行分阶段模式"""
    config = PhasedRunConfig(
        region_slug=args.region,
        phase=args.phase,
        max_pages=args.max_pages,
        resume=not args.no_resume,  # 默认启用断点接续
        wait_ms=args.wait_ms,
        visible=args.visible == "true",
        manual_login=args.manual_login == "true",
        manual_login_timeout_sec=args.manual_login_timeout_sec,
        save_html_on_error=args.save_html_on_error == "true",
        error_html_dir=Path(args.error_html_dir),
        data_dir=Path(args.data_dir),
        auto_login=args.auto_login == "true",
        username=args.username,
        password=args.password,
        run_id=args.run_id,
    )
    
    print("=" * 60)
    print("运行模式: 分阶段 (Phased)")
    print(f"区域: {args.region}")
    print(f"阶段: {args.phase}")
    print(f"断点接续: {'启用' if config.resume else '禁用'}")
    print("=" * 60)
    
    if args.phase == "retry-failures":
        if args.retry_file is None:
            print("错误: phase=retry-failures 需要 --retry-file 参数")
            return
        retry_config = RetryRunConfig(
            region_slug=args.region,
            data_dir=Path(args.data_dir),
            retry_file=Path(args.retry_file),
            retry_type=args.retry_type,
            retry_limit=args.retry_limit,
            wait_ms=args.wait_ms,
            visible=args.visible == "true",
            manual_login=args.manual_login == "true",
            manual_login_timeout_sec=args.manual_login_timeout_sec,
            save_html_on_error=args.save_html_on_error == "true",
            error_html_dir=Path(args.error_html_dir),
            auto_login=args.auto_login == "true",
            username=args.username,
            password=args.password,
            run_id=args.run_id,
        )
        retry_result = run_retry_failures(retry_config, region_map)
        print("\n" + "=" * 60)
        print("失败清单重试完成")
        print(f"总条数: {retry_result['total']}")
        print(f"成功: {retry_result['success']}")
        print(f"失败: {retry_result['failed']}")
        print(f"新增房源: {retry_result['house_added']}")
        print(f"新增小区: {retry_result['community_added']}")
        print("=" * 60)
        return

    result = run_phased(config, region_map)
    
    print("\n" + "=" * 60)
    print("分阶段执行完成")
    
    if "link_extraction" in result:
        le = result["link_extraction"]
        print(f"\n[链接提取]")
        print(f"  遍历页数: {le['total_pages']}")
        print(f"  总链接数: {le['total_links']}")
        print(f"  新增链接: {le['new_links']}")
    
    if "content_extraction" in result:
        ce = result["content_extraction"]
        print(f"\n[内容提取]")
        print(f"  总房源数: {ce['total_houses']}")
        print(f"  成功处理: {ce['processed']}")
        print(f"  跳过(已处理): {ce['skipped']}")
        print(f"  小区数量: {ce['communities']}")
        print(f"  失败数量: {ce['failed']}")
    
    print(f"\n数据目录: {args.data_dir}")
    print("=" * 60)


def main() -> None:
    args = parse_cli_args()
    
    # 加载区域配置
    region_path = Path("config/regions.json")
    if not region_path.exists():
        print(f"错误: 区域配置文件不存在: {region_path}")
        return
    
    region_map: dict[str, str] = json.loads(region_path.read_text(encoding="utf-8"))
    
    if args.region not in region_map:
        print(f"错误: 区域 '{args.region}' 不存在于配置中")
        print(f"可用区域: {', '.join(region_map.keys())}")
        return
    
    # 根据模式执行
    if args.mode == "m0":
        run_m0_mode(args, region_map)
    else:
        run_phased_mode(args, region_map)


if __name__ == "__main__":
    main()
