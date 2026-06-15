"""把检查结果与性能记录渲染成交付用的 Markdown 报告。"""

from __future__ import annotations


def _fmt(v) -> str:
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:.1f}"
    return str(v)


def _cell(v) -> str:
    """格式化为表格单元格,转义会破坏 Markdown 表格的竖线。"""
    return _fmt(v).replace("|", "\\|")


def render_report(meta: dict, checks: list, perf: list) -> str:
    n_pass = sum(1 for c in checks if c.passed)
    n_total = len(checks)
    overall = "PASS" if n_pass == n_total else "FAIL"

    lines: list[str] = []
    lines.append("# feature_extractor 验证报告")
    lines.append("")
    lines.append(f"- 日期:{meta.get('date', '')}")
    lines.append(f"- Git commit:`{meta.get('commit', '')}`　主机:{meta.get('host', '')}")
    lines.append(f"- GPU:{meta.get('gpu', '')}　torch:{meta.get('torch', '')}"
                 f"　CUDA:{meta.get('cuda', '')}")
    lines.append(f"- 依赖:{meta.get('deps', '')}")
    lines.append("")

    # 摘要
    lines.append("## 摘要")
    lines.append("")
    lines.append(f"- 总体:**{overall}**　功能不变量 **{n_pass}/{n_total}** 通过")
    lines.append("")

    # 功能不变量
    lines.append("## 功能不变量")
    lines.append("")
    lines.append("| 分支 | 检查项 | 期望 | 实测 | 结果 |")
    lines.append("|------|--------|------|------|------|")
    for c in checks:
        result = "PASS" if c.passed else "FAIL"
        lines.append(f"| {_cell(c.branch)} | {_cell(c.name)} | {_cell(c.expected)} "
                     f"| {_cell(c.observed)} | {result} |")
    lines.append("")

    # 性能
    lines.append("## 性能")
    lines.append("")
    lines.append("| 视频 | 分支 | 帧数 | 耗时(s) | FPS | 峰值显存(MB) | 备注 |")
    lines.append("|------|------|------|---------|-----|--------------|------|")
    for p in perf:
        lines.append(
            f"| {_cell(p.video_id)} | {_cell(p.branch)} | {_cell(p.frames)} | "
            f"{_cell(p.seconds)} | {_cell(p.fps)} | {_cell(p.peak_mem_mb)} | {_cell(p.note)} |"
        )
    lines.append("")

    # 复现
    lines.append("## 复现")
    lines.append("")
    lines.append("```bash")
    lines.append(meta.get("command", ""))
    lines.append("```")
    lines.append("")
    return "\n".join(lines)
