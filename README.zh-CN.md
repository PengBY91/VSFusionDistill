# ao-det — 退化鲁棒的声光融合检测（F1/F2）

[English](README.md) | 中文

F2 选题的代码库：相机退化下的声呐–相机融合检测——清晰视角特权蒸馏
与支路独立性
（`doc/topic-f2-degradation-robust-fusion-detection-distillation-2026-10.md`），
按 `doc/topic-f2-codebase-selection-and-setup-2026-10.md` 搭建。

基础栈：MMDetection 3.3.0（YOLOX 系）+ mmengine 0.10.7 + torch 2.1.2，
版本固定于 `env/`（本机安装 CPU 调试变体；训练机用 cu121 变体）。
不修改任何第三方仓库：`third_party/` 以固定 SHA 只读克隆
（`tools/clone_third_party.sh`、`third_party/PINNED.md`）。

## 目录结构

| 路径 | 内容 |
|---|---|
| `aodet/datasets/` | 统一配对格式（`paired_format.py`）、`PairedSonarCameraDataset`、转换器、数据变换（配对几何、相机退化 D0–D∞、fork & pack） |
| `aodet/models/` | `TwoStreamYOLOX`（三头）、融合模块、KD 损失（`branch_kd`、`cross_plane_mil`）、冻结教师、`PairedDataPreprocessor` |
| `aodet/evaluation/` | 逐支路 COCO AP、支路误差相关性（φ、ΔP_miss、Spearman、朴素贝叶斯融合、ECE） |
| `aodet/geometry/` | M₀ 相机→方位映射 + 翻转镜像 |
| `configs/` | `_base_`（模型/数据/调度/运行时）+ `f2/` S0-lite 配置 |
| `tools/` | `train.py`、`test.py`（逐支路 D0–D∞ 扫描）、`make_degraded_testsets.py`、`run_seeds.sh`、`collect_results.py`、`correlation_table.py`、第三方 pin、RSFusionDet 监视 |
| `tests/` | 78 项测试：退化物理、格式往返、几何、指标、模型、UXO 转换器与投影、端到端管线与工具 |
| `env/` | 安装脚本 + `ENVIRONMENT.md` 版本记录 |
| `doc/` | 选题方案、按日期的搭建日志（`doc/logs/`）、实验记录（每个实验：设置/结果/解读/结论，`doc/experiments.md`）、证据存档 |

## 快速开始

```bash
source .venv/bin/activate
python -m pytest tests/ -q          # 78 passed

# S0-lite（UXO sample 已转换至 data/paired/uxo/）
python tools/train.py configs/f2/s0_b0_sonar_only_yoloxs.py --seed 0
# mmengine 只写 epoch_N.pth + last_checkpoint（没有 latest.pth 软链）
ckpt=$(cat work_dirs/s0_b0_sonar_only_yoloxs_seed0/last_checkpoint)
python tools/test.py  configs/f2/s0_b0_sonar_only_yoloxs.py "$ckpt" \
    --out-dir work_dirs/s0_b0_sonar_only_yoloxs_seed0
python tools/collect_results.py work_dirs --out results/results.csv
```

每个实验的设置、结果表、解读与结论统一记录在 `doc/experiments.md`；
叙事型过程日志（当日踩坑、排查链）在 `doc/logs/`。

## 配对格式（所有源数据集共用一套布局）

```
<root>/camera.json  sonar.json  pairs.csv  frame_meta.csv
<root>/images/camera/...  images/sonar/...
```

`frame_meta.csv` 把每个 `sample_id` 键接到两个 COCO image id、split
与退化种子；`read_paired` 负责 CSV 类型归一。同步翻转仅当声呐方位轴
0° 居中时合法（`sonar_zero_centred=True`）。

## 退化协议

`aodet/datasets/transforms/camera_degradation.py`：物理有序算子
（衰减 → 波长吸收 → 散射雾幕 → 模糊），级别 D0–D4、D∞（全黑/传感器
缺失），按 (sample_id, level, seed) 确定性生成。训练按 p(s) 抽级别
（F2 §5.2），D∞ 物化为黑帧（`dinf_mode="zero"`，batch 保持稠密）；
测试期 D∞ 丢弃相机（`dinf_mode="drop"`，batch 1）。离线固定测试集：
`tools/make_degraded_testsets.py`（逐级别 sha256 manifest）。

## 评测

基于保存的预测文件按退化级别出逐支路 AP（`tools/test.py` →
`pred.<branch>.<level>.json` + `ap_summary.json`）；支路误差相关性
（φ、带 bootstrap 置信区间的条件漏检升幅 ΔP_miss、Spearman 置信
相关、朴素贝叶斯融合存在性 ECE）实现在
`aodet/evaluation/branch_correlation.py`，经 `tools/correlation_table.py`
对 run 目录输出 `correlation.csv`。

## 状态（2026-10-31）

- W1 环境安装并验证；COCO YOLOX-S 复现门在 val2017 全量（5000 图）
  通过（AP 40.30 vs 官方 40.5）；W2 配对格式/退化/变换/指标完成并有
  测试；W3 模型+配置+工具完成，S0-lite 在合成数据上端到端冒烟通过。
- UXO sample 核验（×3 标注规则、GoPro 对齐、外参）并转换：
  `data/paired/uxo/` 下 2573 对（recording 级划分，train 2199 /
  test 374）。声呐侧框 + 跨模态对应 π 完成（TODO 14.2 条目 4）：
  逐 recording 稳健标定 + 投影，`sonar.json` 已标注（blob 过门 1185 /
  几何投影 1388），`pairs.csv` 已写；误差统计与拼图见
  `data/paired/uxo/projection_report.json` / `projection_check.jpg`
  及 W1–W3 日志。
- 本机 GPU 环境就绪，可跑 S0-lite（RTX 5090 D v2 24 GB）：`.venv-gpu`
  = torch 2.7.1+cu128 + 源码 CPU-ops mmcv 2.1.0 + NMS CPU 回退
  （`aodet/cuda_compat.py`）——见 `env/ENVIRONMENT.md`。固定 cu121
  栈无法跑 Blackwell；CPU 调试 `.venv` 未动。
- S0-lite 首跑暴露 GT 语义缺陷：几何投影框在水池距离横跨半幅图而
  真实回波紧凑，训练虽收敛 AP 仍 ≈0。投影框已钳制到回波尺度
  （`_PROJ_BOX_MAX`），GT 语义统一定义为「可见回波区域」。B0 +
  cross-attn 在修正后的 GT 上重训中。
- RSFusionDet 仓库仍为空（路线 a 监视中）。
