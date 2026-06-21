# third_party 去子模块化 + 裁剪 vendoring 设计

- 日期:2026-06-21
- 范围:为代码入库合规,把 `third_party/` 的四个 git 子模块改为**仓库内自包含的精简源码**,
  只保留运行时真正用到的文件,删掉训练/评测/其他任务等无关代码。**全功能保留**
  (dino + pose + depth 几何 + depth 代理)。

## 1. 目标与约束

| 约束 | 来源 | 结论 |
|------|------|------|
| 不允许 submodule / 外部仓 | 入库合规 | 删除 `.gitmodules`、gitlink、`.git/modules`,改为提交普通源码 |
| 减少待扫描的第三方代码量 | 合规扫描负担 | 只 vendoring 被 import 到的文件,丢弃其余(~290 .py → 预计大幅下降) |
| 功能不变 | 用户 | dino/pose/depth(VDA)/depth_pro/dino_attention 全部仍可用 |

非目标:不换实现(不引 pip/transformers 重写);不动主管道逻辑;不动 checkpoints 获取方式。

## 2. 现状(实测)

`src` 实际用到的入口:
- `dinov3.hub.backbones.<builder>`(dino)
- `vggt.models.vggt.VGGT`、`vggt.utils.pose_enc.pose_encoding_to_extri_intri`(pose)
- `video_depth_anything.video_depth.VideoDepthAnything`、`video_depth_anything.util.transform`(depth=VDA)
- `depth_pro`(VDA 度量校正 / depth_pro 模式)

加载器 `_load_dinov3` 等通过 `sys.path.insert(root/local_repo)` + 包名 import;**vendored 后保留
`third_party/<repo>/` 原路径,加载逻辑零改动**。

体积(不含权重):VGGT 64M/69py、dinov3 19M/174py、VDA 8.1M/35py、ml-depth-pro 2.8M/12py。

## 3. 方法:经验追踪为主 + 静态兜底

### 3.1 运行时追踪(主)
写一个一次性脚本 `scripts/trace_thirdparty_usage.py`:
1. 构建并运行**所有分支**(复用 `feature_extractor.validation` 的合成视频 + 各 extractor:
   DINO(vits16plus 与 vits16 都触发一次)、Depth(`video_depth_anything`,内部会带出 VDA
   依赖的 DINOv2 编码器)、Depth(`depth_pro`)、Pose(VGGT));对合成 clip 各跑一次 `extract_video`。
2. 运行后遍历 `sys.modules`,凡 `__file__` 落在某 `third_party/<repo>/` 下的,记录其相对路径
   → 得到**运行时实际加载文件集**(自动覆盖 getattr/动态/内部依赖)。

### 3.2 静态闭包(兜底)
从上述入口模块用 AST 递归解析 `import`,求可达模块集,与运行时集**取并集**——补上某些
运行分支没覆盖到、但可能被其他参数路径用到的文件(如其他 dino_model 变体)。

### 3.3 保留清单 = 运行时集 ∪ 静态集 ∪ 必需附属
附属必须保留:各级 `__init__.py`(保证包可导入)、`LICENSE`/`NOTICE`/版权头文件、被代码
读取的非 .py 资源(若有,如 config yaml——追踪时一并记录被 open 的路径)。

## 4. Vendoring 执行(每个 repo)

1. 依据保留清单把文件复制到临时目录(保持包内相对路径)。
2. 去子模块:`git submodule deinit -f third_party/<repo>` → `git rm -f third_party/<repo>` →
   `rm -rf .git/modules/third_party/<repo>`;最终删空 `.gitmodules`。
3. 把精简文件放回 `third_party/<repo>/`(普通目录),`git add` 提交为仓库源码。
4. 每个 repo 顶层保留 `LICENSE`,新增 `PROVENANCE.md`(上游 URL + 原 commit + 裁剪说明),
   满足合规署名/溯源。

## 5. checkpoints(权重)

权重不是代码、不进合规扫描,**不 vendoring**。当前 `third_party/<repo>/checkpoints` 是软链到
egoWM;vendoring 后:
- 新增 `.gitignore` 规则 `third_party/*/checkpoints/`,确保**提交精简代码但不提交权重**。
- 软链/权重获取方式不变(README 已说明 OBS / 官方下载)。

## 6. 验证

在**干净状态**(无任何 submodule,`third_party/` 仅含精简后的普通文件)下:
1. `git submodule status` 为空;`.gitmodules` 不存在;`git ls-files third_party/ | wc -l` 显示已纳管精简源码。
2. 全套单测通过:`uv run --extra dev python -m pytest -q`。
3. 三分支端到端:`feature-validate --branches dino,depth,pose --depth_mode video_depth_anything --skip-perf`
   全 PASS(覆盖 DINO/VDA/depth_pro/VGGT 加载与推理);再跑 `--dino_model dinov3_vits16` 一次。
4. 可视化脚本 `export_dino_video.py --max_frames 8` 能出图。
5. 任何 `ImportError`/缺文件 → 把缺的文件加入保留清单,重跑(迭代直到全绿)。

## 7. 风险

- **漏动态导入文件 → 运行时 ImportError**:经验追踪 + 第 6 步实跑可暴露;迭代补齐。
- **裁剪后 VGGT 仍较大**:模型+head 是不可约的推理代码;但训练/数据/评测/示例(占大头)会被删。
- **上游许可**:DINOv3/VGGT 等可能为非商用许可——本设计保留 LICENSE 并溯源,但**是否允许入库由合规
  按许可证判定**,本任务不改变许可性质(若合规因许可证拒绝,则需回到「换实现/移除该分支」,超出本范围)。

## 8. 交付

- `third_party/<repo>/` 精简普通源码(含 LICENSE + PROVENANCE.md)。
- `.gitmodules` 删除;`.gitignore` 加 checkpoints 规则。
- 一次性追踪脚本 `scripts/trace_thirdparty_usage.py`(便于将来升级上游后复跑)。
- 验证记录(测试 + feature-validate 全绿)。
