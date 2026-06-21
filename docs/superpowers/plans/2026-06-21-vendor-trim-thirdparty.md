# third_party 去子模块化 + 裁剪 vendoring Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把 `third_party/` 的四个 git 子模块改为仓库内自包含的**精简源码**(只留运行时真正用到的文件),去子模块化,全功能保留。

**Architecture:** 经验式 import 追踪得到「保留清单」→ 在 staging 目录构建精简树并**先验证**(子模块未动)→ 验证通过后才去子模块、把精简树落回原路径 `third_party/<repo>/` → 干净状态下回归验证。完整性的最终保证是"精简代码能跑通三分支"。

**Tech Stack:** Python、`sys.modules` 运行时追踪、uv、pytest、feature-validate。GPU(CUDA_VISIBLE_DEVICES=7)+ 已就位权重。

**关键事实(已快照):**
- checkpoints 均为软链 → `/root/codes/egoWM/third_party/<repo>/checkpoints`(de-submodule 后需重建)。
- 上游 commit:dinov3 `31703e4`、VGGT `44b3afb`、Video-Depth-Anything `4f5ae23`、ml-depth-pro `9efe5c1`。
- LICENSE 文件:`dinov3/LICENSE.md`、`VGGT/LICENSE.txt`、`Video-Depth-Anything/LICENSE`、`ml-depth-pro/LICENSE`。

> **执行建议:Inline(executing-plans)**。本任务是探索性的(保留清单运行时才确定,需迭代),需会话连续性,不适合每任务派冷 subagent。

---

### Task 1: 记录溯源信息(非破坏)

**Files:**
- Create: `third_party/PROVENANCE.md`

- [ ] **Step 1: 写溯源文件**

`third_party/PROVENANCE.md`:
```markdown
# third_party 溯源

下列目录为从上游仓库**裁剪**后 vendoring 的精简源码(仅保留本项目运行时实际 import 的文件),
原以 git submodule 引用,为入库合规改为仓库内普通文件。各目录顶层保留上游 LICENSE。

| 目录 | 上游 | 裁剪自 commit |
|------|------|----------------|
| dinov3 | https://github.com/facebookresearch/dinov3.git | 31703e4cbf1ccb7c4a72daa1350405f86754b6d1 |
| VGGT | https://github.com/facebookresearch/vggt.git | 44b3afbd1869d8bde4894dd8ea1e293112dd5eba |
| Video-Depth-Anything | https://github.com/DepthAnything/Video-Depth-Anything.git | 4f5ae23172ba60fd7bc11ef671cca678842c7072 |
| ml-depth-pro | https://github.com/apple/ml-depth-pro.git | 9efe5c1def37a26c5367a71df664b18e1306c708 |

裁剪方法见 `scripts/trace_thirdparty_usage.py`(运行时 import 追踪)。升级上游后可复跑该脚本重新生成保留清单。
```

- [ ] **Step 2: 暂不提交(待 Task 5 一起提交)。仅确认文件写入。**

Run: `head -3 third_party/PROVENANCE.md`
Expected: 打印标题。

---

### Task 2: 写运行时追踪脚本,生成保留清单

**Files:**
- Create: `scripts/trace_thirdparty_usage.py`
- Create(产物): `scripts/keep/<repo>.txt`(运行后生成)

- [ ] **Step 1: 写追踪脚本**

`scripts/trace_thirdparty_usage.py`:
```python
#!/usr/bin/env python3
"""运行所有用到 third_party 的分支,记录 sys.modules 中真实加载的文件 → 保留清单。

用法: CUDA_VISIBLE_DEVICES=7 uv run python scripts/trace_thirdparty_usage.py
产物: scripts/keep/<repo>.txt(每行一个相对 third_party/<repo>/ 的文件路径)
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TP = (ROOT / "third_party").resolve()
REPOS = ["dinov3", "VGGT", "Video-Depth-Anything", "ml-depth-pro"]


def exercise_all_branches() -> None:
    import numpy as np  # noqa: F401
    from feature_extractor.extractors.dino import DINOExtractor
    from feature_extractor.extractors.depth import DepthExtractor
    from feature_extractor.extractors.pose import PoseExtractor
    from feature_extractor.validation.synthetic import make_gradient_video

    dev = "cuda"
    with tempfile.TemporaryDirectory() as td:
        vid = str(Path(td) / "g.mp4")
        make_gradient_video(vid, n_frames=6)
        idx = list(range(6))

        # DINO:两个 vits16* 变体都触发(都走本地 dinov3)
        dino = None
        for mn in ("dinov3_vits16plus", "dinov3_vits16"):
            dino = DINOExtractor(model_name=mn, device=dev)
            dino.extract_video(vid, frame_indices=idx)

        # Depth:VDA(内部带 DINOv2 编码器)+ depth_pro
        DepthExtractor(mode="video_depth_anything", device=dev,
                       dino_extractor=dino).extract_video(vid, frame_indices=idx)
        try:
            DepthExtractor(mode="depth_pro", device=dev,
                           dino_extractor=dino).extract_video(vid, frame_indices=idx)
        except Exception as e:  # depth_pro 权重缺失等不应阻断其余追踪
            print(f"[trace] depth_pro 跳过: {e}")

        # Pose:VGGT
        PoseExtractor(device=dev).extract_video(vid, frame_indices=idx)


def collect_used() -> dict[str, set[str]]:
    used = {r: set() for r in REPOS}
    for mod in list(sys.modules.values()):
        f = getattr(mod, "__file__", None)
        if not f:
            continue
        try:
            fp = Path(f).resolve()
        except Exception:
            continue
        for r in REPOS:
            base = (TP / r).resolve()
            prefix = str(base) + os.sep
            if str(fp).startswith(prefix):
                used[r].add(str(fp.relative_to(base)))
    return used


def main() -> int:
    exercise_all_branches()
    used = collect_used()
    out_dir = ROOT / "scripts" / "keep"
    out_dir.mkdir(parents=True, exist_ok=True)
    for r in REPOS:
        files = sorted(used[r])
        (out_dir / f"{r}.txt").write_text("\n".join(files) + "\n")
        print(f"{r}: {len(files)} files used")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 2: 运行追踪,生成保留清单**

Run: `CUDA_VISIBLE_DEVICES=7 uv run python scripts/trace_thirdparty_usage.py 2>&1 | grep -E "files used|跳过"`
Expected: 打印每个 repo 的 used 文件数(均 > 0,VGGT/dinov3 应有数十个;远小于其总文件数)。

- [ ] **Step 3: 人工抽查清单合理性**

Run: `for r in dinov3 VGGT Video-Depth-Anything ml-depth-pro; do echo "== $r =="; wc -l scripts/keep/$r.txt; head -5 scripts/keep/$r.txt; done`
Expected: 每个 repo 的清单含其 `__init__.py` 与模型/层定义路径(如 `vggt/models/vggt.py`、`dinov3/hub/backbones.py`)。

> 若某 repo used=0,说明该分支没真正加载它 → 停下排查(可能权重缺失导致提前失败),修好再继续。

---

### Task 3: 构建 staging 精简树并验证(子模块未动,可迭代)

**Files:**
- Create(临时): `/tmp/tp_trim/third_party/<repo>/...`

- [ ] **Step 1: 从保留清单复制精简树到 staging,并补 LICENSE + checkpoints 软链**

```bash
cd /root/codes/feature_extractor
rm -rf /tmp/tp_trim && mkdir -p /tmp/tp_trim/third_party
for r in dinov3 VGGT Video-Depth-Anything ml-depth-pro; do
  dst="/tmp/tp_trim/third_party/$r"
  while IFS= read -r f; do
    [ -z "$f" ] && continue
    mkdir -p "$dst/$(dirname "$f")"
    cp "third_party/$r/$f" "$dst/$f"
  done < "scripts/keep/$r.txt"
  # 保留 LICENSE(顶层任意 LICENSE*)
  cp third_party/$r/LICENSE* "$dst/" 2>/dev/null || true
  # checkpoints 软链复用 egoWM(验证需要权重)
  ln -sfn "/root/codes/egoWM/third_party/$r/checkpoints" "$dst/checkpoints"
done
echo "staged file counts:"; for r in dinov3 VGGT Video-Depth-Anything ml-depth-pro; do echo "$r: $(find /tmp/tp_trim/third_party/$r -name '*.py' | wc -l) py"; done
```
Expected: 每个 repo 的 staged .py 数 = 对应 keep 清单行数。

- [ ] **Step 2: 用 staging 精简树跑三分支(assets_root 指向 staging),验证完整性**

Run:
```bash
CUDA_VISIBLE_DEVICES=7 uv run feature-validate \
    --branches dino,depth,pose --depth_mode video_depth_anything --skip-perf \
    --assets_root /tmp/tp_trim --report /tmp/trim_val.md 2>&1 | grep -ivE "it/s\]|Svt\[" | tail -15
echo "EXIT: ${PIPESTATUS[0]}"; grep -c PASS /tmp/trim_val.md
```
Expected: 加载并跑通 DINO/VDA/depth_pro 的 DINOv2/VGGT,报告 `功能 N/N 通过`,无 `ModuleNotFoundError`/`FileNotFoundError`。

- [ ] **Step 3: 若有缺文件 → 补进保留清单,重做 Step 1–2(迭代直到全绿)**

排查命令(从报错里找缺的模块/文件):
```bash
# 例:报错 ModuleNotFoundError: No module named 'vggt.heads.xxx'
# 把对应文件加入清单:
echo "vggt/heads/xxx.py" >> scripts/keep/VGGT.txt
# 缺非 .py 资源(FileNotFoundError 指向 third_party/<repo>/...)同理把该资源相对路径加入清单。
```
重跑 Step 1、Step 2,直到 EXIT 0 且全 PASS。**这一步是完整性的最终保证。**

- [ ] **Step 4: 也验证 vits16 与可视化脚本走 staging**

Run:
```bash
CUDA_VISIBLE_DEVICES=7 uv run python -c "
from feature_extractor.extractors.dino import DINOExtractor
import os; os.environ['FEATURE_EXTRACTOR_ASSETS']='/tmp/tp_trim'
DINOExtractor(model_name='dinov3_vits16', device='cuda', assets_root='/tmp/tp_trim')
print('vits16 OK on staging')
"
```
Expected: `vits16 OK on staging`,无 ImportError。

- [ ] **Step 5: 提交追踪脚本 + 最终保留清单**

```bash
git add scripts/trace_thirdparty_usage.py scripts/keep/
git commit -m "chore: add third_party usage tracer + keep-lists"
```

---

### Task 4: 去子模块 + 落回精简树 + 重建 checkpoints/LICENSE/PROVENANCE

> ⚠️ 破坏性操作。务必 Task 3 已全绿。先全部 deinit/删除,再从 staging 落回。

- [ ] **Step 1: 逐个去子模块**

```bash
cd /root/codes/feature_extractor
for r in dinov3 VGGT Video-Depth-Anything ml-depth-pro; do
  git submodule deinit -f "third_party/$r"
  git rm -f "third_party/$r"
  rm -rf ".git/modules/third_party/$r"
done
rm -f .gitmodules
echo "submodules now:"; git submodule status; echo "(空即正确)"
```
Expected: `git submodule status` 无输出;`.gitmodules` 不存在;`third_party/` 下四个目录已被删除。

- [ ] **Step 2: 从 staging 落回精简源码到原路径**

```bash
for r in dinov3 VGGT Video-Depth-Anything ml-depth-pro; do
  mkdir -p "third_party/$r"
  # 复制精简代码(含 LICENSE),排除 checkpoints 软链(单独重建)
  rsync -a --exclude 'checkpoints' "/tmp/tp_trim/third_party/$r/" "third_party/$r/"
  # 重建 checkpoints 软链(本机权重复用 egoWM)
  ln -sfn "/root/codes/egoWM/third_party/$r/checkpoints" "third_party/$r/checkpoints"
done
echo "落回后 py 计数:"; for r in dinov3 VGGT Video-Depth-Anything ml-depth-pro; do echo "$r: $(find third_party/$r -name '*.py' | wc -l)"; done
```
Expected: 各 repo .py 数与 staging 一致;checkpoints 为软链。

- [ ] **Step 3: 确认 LICENSE 与 PROVENANCE 就位**

Run: `ls third_party/PROVENANCE.md third_party/*/LICENSE* 2>&1`
Expected: PROVENANCE 与四个 LICENSE 都在。

---

### Task 5: .gitignore 权重 + 提交精简源码

**Files:**
- Modify: `.gitignore`

- [ ] **Step 1: 忽略 checkpoints(只提交代码不提交权重)**

在 `.gitignore` 末尾追加:
```
# vendored third_party model weights (downloaded separately; see README)
third_party/*/checkpoints/
```

- [ ] **Step 2: 确认 checkpoints 不会被纳入,精简代码会被纳入**

Run: `git add -A && git status --short | grep -E "third_party/(dinov3|VGGT|Video-Depth-Anything|ml-depth-pro)/checkpoints" | head; echo "---"; git status --short | grep -c "third_party/"`
Expected: 第一段(checkpoints)**无输出**(被忽略);第二段计数为大量新增的精简源码文件。

- [ ] **Step 3: 提交**

```bash
git commit -m "build: de-submodule third_party; vendor trimmed sources

Replace the four git submodules (dinov3/VGGT/Video-Depth-Anything/ml-depth-pro)
with in-repo trimmed sources containing only the files loaded at runtime, for
self-contained check-in and reduced compliance-scan surface. LICENSE files and
PROVENANCE.md (upstream URL+commit) retained. Weights stay external (gitignored)."
```

---

### Task 6: 干净状态回归验证 + 度量

- [ ] **Step 1: 确认干净自包含状态**

Run:
```bash
git submodule status; echo "submodules ^(空=OK)"
test -f .gitmodules && echo ".gitmodules STILL EXISTS (BAD)" || echo ".gitmodules gone (OK)"
echo "third_party 纳管文件数: $(git ls-files third_party/ | wc -l)"
```
Expected: 无子模块;`.gitmodules` 不存在;纳管文件数为精简后的源码数(远小于原 ~290)。

- [ ] **Step 2: 全套单测**

Run: `CUDA_VISIBLE_DEVICES="" uv run --extra dev python -m pytest -q`
Expected: 全部通过(28+)。

- [ ] **Step 3: 三分支端到端(用默认 assets_root = 仓库内精简树)**

Run:
```bash
CUDA_VISIBLE_DEVICES=7 uv run feature-validate \
    --branches dino,depth,pose --depth_mode video_depth_anything --skip-perf \
    --report /tmp/final_val.md 2>&1 | grep -ivE "it/s\]|Svt\[" | tail -6
echo "EXIT: ${PIPESTATUS[0]}"; grep -E "总体|功能" /tmp/final_val.md
```
Expected: EXIT 0,功能 N/N 通过(证明精简后 DINO/VDA/depth_pro/VGGT 全可加载推理)。

- [ ] **Step 4: 可视化脚本**

Run:
```bash
CUDA_VISIBLE_DEVICES=7 uv run python src/feature_extractor/virtualization/export_dino_video.py \
  --video_path data/libero_10/videos/observation.images.image/chunk-000/file-000.mp4 \
  --output_mp4 /tmp/trim_vis.mp4 --dino_model dinov3_vits16 --render_scale 0.5 --max_frames 8 2>&1 | tail -3
ls -l /tmp/trim_vis.mp4 | awk '{print $5}'
```
Expected: 生成非空 mp4。

- [ ] **Step 5: 度量裁剪效果,记入 PROVENANCE(可选)**

Run: `for r in dinov3 VGGT Video-Depth-Anything ml-depth-pro; do echo "$r: $(find third_party/$r -name '*.py' | wc -l) py, $(du -sh --exclude=checkpoints third_party/$r | cut -f1)"; done`
Expected: 每个 repo 文件数/体积显著下降(对比原 174/69/35/12)。

- [ ] **Step 6: 最终提交(若 Step 5 更新了 PROVENANCE)**

```bash
git add third_party/PROVENANCE.md && git commit -m "docs: record third_party trim metrics" || echo "无改动可提交"
```
