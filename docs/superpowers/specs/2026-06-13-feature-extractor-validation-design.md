# feature_extractor 自验证设计(交付产品团队)

- 日期:2026-06-13
- 范围:为 `feature_extractor` 模块设计并实现自验证工具,从**功能(合理性/不变量)**与**性能(吞吐/显存/扩展性)**两个角度产出可交付的验证结论。

## 1. 目标与决策

| 维度 | 决策 |
|------|------|
| 功能正确性基准 | 合理性 / 不变量检查(**无金标准**,不与 egoWM 逐元素对齐) |
| 性能指标 | 吞吐 / 处理速度、峰值显存、规模扩展性 |
| 交付形态 | 可重跑脚本 + 自动生成 Markdown 报告 |
| 输入数据 | 功能用**合成视频**(确定、可移植);性能用**真实数据集**(代表性数字) |
| 覆盖范围 | 交付配置:`dino + depth(video_depth_anything) + pose` |


## 2. 架构与结构

新增子包 `src/feature_extractor/validation/`,文件职责单一:

```
src/feature_extractor/validation/
  __init__.py
  synthetic.py   # 造可控合成视频(复用 tests/test_video_io.py 的 ramp 生成器)
  sanity.py      # 各分支不变量检查 -> list[CheckResult]
  perf.py        # 真实数据基准 -> list[PerfRecord]
  report.py      # 渲染 Markdown
  cli.py         # 编排 + 入口(main)
```

注册 console entry:`feature-validate = feature_extractor.validation.cli:main`(`pyproject.toml`)。

合成视频生成器从 `tests/test_video_io.py` 抽取为 `synthetic.py` 的公共函数,测试改为引用它,避免重复。

### 数据结构

```python
@dataclass
class CheckResult:
    branch: str          # "dino" | "depth" | "pose" | "pipeline" | "storage"
    name: str            # 检查项名
    expected: str        # 期望(人读)
    observed: str        # 实测(人读)
    passed: bool

@dataclass
class PerfRecord:
    video_id: str
    branch: str          # 或 "model_load" / "decode" / "end_to_end"
    frames: int
    seconds: float
    fps: float           # frames / seconds(不适用时为 None)
    peak_mem_mb: float   # 峰值显存(不适用时为 None)
    note: str            # 如 "SKIPPED: weights not found"
```

## 3. CLI

```bash
uv run feature-validate \
    --data_root data/libero_10/videos \      # 性能用真实数据(--skip-perf 时可省)
    --report validation_report.md \          # 默认 validation_report.md
    --device cuda \
    --branches dino,depth,pose \             # 默认交付配置
    --depth_mode video_depth_anything \      # 默认
    --frames-sweep 16,32,64,128 \            # 扩展性扫描;超视频长度自动截断
    --perf-frames 64 \                       # 单视频吞吐基准帧数
    --assets_root <可选> \
    [--skip-perf] [--skip-sanity]
```

- 合理性检查用合成视频,**不需要** `--data_root`。
- 性能用 `--data_root`;`--skip-sanity` / `--skip-perf` 可单独跑某一段。
- 权重缺失:对应性能/功能项标 `SKIPPED`,不崩溃,报告如实记录。

## 4. 功能不变量(合成视频,逐分支)

原则:只选**与模型质量无关、构造上必然成立**的硬不变量,断言才稳。确定性用 `allclose`(容差),不强求逐 bit(cuDNN 可能非确定)。

### 通用 / 跨分支
- 形状:dino `(T, N+1, 384)`、depth `(T, H, W, 1)`、pose `(T, 9)`
- dtype = float32;全部有限(无 NaN/Inf)
- `T == len(frame_indices)`,三分支使用**同一组帧索引**(对齐保证)
- 存储往返:`write_* → read_all` 数值完全一致(lzf 无损)
- 写回 HDF5 的帧索引与请求采样一致

### DINO
- embed_dim = 384(`dinov3_vits16plus`);patch 数 N 与输入分辨率 / patch_size 吻合
- CLS 行与 patch 行不相同(非退化)
- 同输入两次运行 `allclose`(确定性)

### Depth(逆深度)
- 逆深度 ≥ 0;全部有限;非全常数(有信号)
- 同输入两次运行 `allclose`

### Pose(相对 SE3,9 维 = 平移 3 + 6D 旋转)
- `pose[0] ≈ [0,0,0, 1,0,0, 0,1,0]`(相对第 0 帧 = 单位变换;构造保证,最强硬不变量)
- 每帧 6D 旋转重建出的 R 正交且 `det ≈ +1`(6D→SO(3) 构造保证)
- 平移 / 旋转有限

## 5. 性能口径(真实数据,dino + VDA + pose)

- **环境捕获**:git commit、主机名、GPU 型号、CUDA/驱动、torch 版本、关键依赖(av / decord)版本
- **模型加载耗时**:一次性测量,单列,**不计入**吞吐
- **吞吐**:预热 1~2 帧(不计入),计时区用 `torch.cuda.synchronize()` 包裹;逐分支 `FPS = 帧数 / 秒`,并记端到端单视频耗时
- **峰值显存**:每分支前 `torch.cuda.reset_peak_memory_stats()`,后 `torch.cuda.max_memory_allocated()` → 逐分支峰值(MB)
- **解码 vs 推理拆分**:单独计 `video_io` 读帧耗时,与推理耗时对比(PyAV 回退后值得观察)
- **扩展性扫描**:在一条代表性真实视频上扫 `--frames-sweep`,记录每档耗时 / FPS / 峰值显存,据此判断时间是否近似线性、瓶颈位置(解码 vs 推理 vs 显存)

## 6. 报告版式(Markdown)

1. 标题 + 元数据(日期、git commit、环境摘要)
2. **摘要**:总体 PASS/FAIL、`X/Y` 项通过;头条性能(各分支 FPS、峰值显存)
3. **功能不变量**:逐分支表格(检查项 | 期望 | 实测 | 结果)
4. **性能**:模型加载表 / 逐分支吞吐 + 显存表 / 解码 vs 推理 / 扩展性扫描表
5. **环境明细**:依赖与版本
6. **复现**:原样附上本次运行命令

报告默认输出到 `--report` 指定路径(缺省 `validation_report.md`)。

## 7. 错误处理

- 权重缺失 / 某分支构造失败:标 `SKIPPED` 并附原因,继续其余检查,报告如实反映,进程不崩。
- 真实数据缺失且未 `--skip-perf`:明确报错并提示 `--data_root` 或 `--skip-perf`。
- 合成视频生成失败(编码器缺失):合理性段标 SKIPPED 并说明所缺编码器。

## 8. 测试

- `synthetic.py` 抽取后,`tests/test_video_io.py` 改为引用它(保持现有 10 个用例通过)。
- 为 `sanity.py` 的纯函数(形状/有限性/SE3 正交性/6D→R 重建)加少量单测,使用极小合成输入,不依赖重权重。
- `report.py` 渲染函数对给定的 `CheckResult` / `PerfRecord` 列表做快照式断言(纯字符串,不跑模型)。