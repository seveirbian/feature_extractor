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
