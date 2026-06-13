from dataclasses import dataclass
from typing import Optional

from feature_extractor.validation.sanity import CheckResult
from feature_extractor.validation.report import render_report


# report.render_report 按属性名鸭子类型访问性能记录,故意不依赖 perf.py
# (真正的 PerfRecord 在 perf.py 定义,字段须与此 stand-in 保持一致)。
@dataclass
class _Perf:
    video_id: str
    branch: str
    frames: int
    seconds: float
    fps: Optional[float]
    peak_mem_mb: Optional[float]
    note: str = ""


def test_render_report_has_sections_and_counts():
    meta = {"date": "2026-06-13", "commit": "abc1234", "gpu": "TestGPU",
            "torch": "2.5.0", "command": "feature-validate --x"}
    checks = [
        CheckResult("dino", "embed_dim", "384", "384", True),
        CheckResult("pose", "pose0=identity", "~I", "max|.|=0.3", False),
    ]
    perf = [_Perf("vid0", "dino", 64, 4.0, 16.0, 1200.0)]
    md = render_report(meta, checks, perf)

    assert "# feature_extractor 验证报告" in md
    assert "1/2" in md                 # 通过计数
    assert "FAIL" in md and "PASS" in md
    assert "pose0=identity" in md      # 失败项可见
    assert "TestGPU" in md             # 环境
    assert "feature-validate --x" in md  # 复现命令
    assert "16.0" in md                # fps
    # 单元格内的竖线被转义,不破坏 Markdown 表格
    assert "max\\|.\\|=0.3" in md
    assert "max|.|=0.3" not in md
