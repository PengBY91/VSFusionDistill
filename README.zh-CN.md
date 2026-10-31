# ao-det — 退化鲁棒的声光融合检测（F2）

[English](README.md) | 中文

**一句话**：水下机器人同时带着前视声呐和相机出海。声呐不受浊度影响，
相机画质却随浊度与光照**连续退化**——从清水（D0）到重度退化（D4），
直到完全不可用（D∞）。本代码库在整条可靠性轴上训练与评测双流声光
检测器，回答两个问题：**训练期特权蒸馏**（教师看*清晰*相机，学生看
退化相机）是否比推理期特征融合更能保住声呐支路；以及这种"互增强"
的代价——两条支路的误差被拉相关了多少（下游把两路检出当独立证据用）。

完整研究方案：`doc/topic-f2-degradation-robust-fusion-detection-distillation-2026-10.md` ·
实验台账（每个实验：设置/结果/解读/结论）：`doc/experiments.md` ·
搭建过程日志：`doc/logs/`

基础栈：MMDetection 3.3.0（YOLOX 系）+ mmengine 0.10.7 + torch 2.1.2，
版本固定于 `env/`（本机另建 torch 2.7.1+cu128 的 `.venv-gpu` 跑
RTX 5090，见 `env/ENVIRONMENT.md`）。不修改任何第三方仓库：
`third_party/` 以固定 SHA 只读克隆（`tools/clone_third_party.sh`、
`third_party/PINNED.md`）。

## 1. 这份代码要解决什么问题

部署现实：ROV/AUV 上前视声呐与相机同时在线，相机可靠性是**连续
退化**，不是开关。现有方法只覆盖这条谱的两端——纯声呐蒸馏
（CMOLD/CMHD，D∞ 端专家，相机清晰时浪费信息）与清水条件下的推理期
融合（RSFusionDet 等，从未系统测过退化行为）。清水下融合对单支路的
增益只有约 1 AP，提升空间在退化区。同时，库内没有任何声光检测工作
报告过**支路间的误差相关性**，而下游（目标建图/关联）恰恰把逐模态
检出当条件独立证据使用——互增强换来的 AP 可能在下游变成过度自信。
完整论证见选题文档 §1。

本仓库用同一套框架（MMDetection/YOLOX）、同一份数据、同一个退化
协议做受控比较：

| 研究问题 | 一句话 | 在代码里的载体 |
|---|---|---|
| RQ-0（闸门） | 相机退化时，推理期交叉注意力融合会不会拖累声呐支路（互毒害）？ | `s0_cross_attn` vs `s0_b0` 的逐支路 D0–D∞ 扫描（`tools/test.py`） |
| RQ-1 | 对声呐支路，训练期蒸馏与推理期融合谁更有效？ | 方法空间 B1 vs B2/B3 vs M3（§2.4） |
| RQ-2 | 蒸馏收益里「容量收益」（大教小学生）与「特权收益」（教师看清晰、学生看退化）各占多少？ | 2×2 归因：同容量/大教师 × 退化/清晰输入 |
| RQ-3 | 跨平面方位窗蒸馏是否优于逐网格蒸馏？ | `aodet/models/losses/cross_plane_mil.py` |
| RQ-4 | 互增强让两支路误差相关了多少？代价多大？ | `aodet/evaluation/branch_correlation.py` |

## 2. 研究思路

### 2.1 学生模型：双流三头

所有方法共用一个学生骨架 `TwoStreamYOLOX`（YOLOX-S 量级，教师/
学生同家族，保证逐预测蒸馏对得上）：

```
camera image ──► cam backbone ──► cam neck ──┬────────────────► camera head          (L_det^c)
                                              │                  训练期被施加 D0–D∞ 退化
sonar  image ──► son backbone ──► son neck ──┼──► fusion ──────► fused sonar head    (L_det^s)
  （永不退化）                                │
                                              └────────────────► uncoupled sonar head (L_det^s0)
                                                                 · 只看融合前的声呐特征，从不接触相机
                                                                 · D∞ 回退头 + 未耦合记账，评测单独报告
```

- **camera head** 只看相机特征；**fused sonar head** 看融合特征（默认
  交叉注意力，声呐查询相机）；两个头各自用本模态 GT 监督。
- **uncoupled sonar head** 是设计关键（选题文档张力 1）：它给出
  "未被相机耦合污染"的声呐输出，D∞ 时天然回退，评测时单独记账。
- 去掉相机流（`camera_backbone=None` + `NoneFusion`）即 B0 纯声呐
  基线——同一框架内的最强单模态参照。

### 2.2 特权蒸馏

教师固定看**清晰**相机帧（几何增强师生同步施加），因此其输出可
离线缓存；当前为在线 `no_grad` 前向（`FrozenTeacher`），缓存启用前
须过等价测试。

```
clear camera frame ──► FrozenTeacher（更大或同容量双流 YOLOX，no_grad）
        │
        ├─ 头输出 objectness + 类别分布 ──► branch_kd ──► 蒸给学生 camera/sonar 头
        │                                   （只蒸 objectness 与类别，不蒸框回归）
        │
        └─ 相机检测框 ──► M₀ 名义几何（相机水平角 → 声呐方位窗）
                            └─► cross_plane_mil：窗内学生声呐头沿距离方向的
                                 最大 objectness 须与教师相符（多示例学习）
                                 · 只传正证据：相机没看到 ≠ 声呐条带没东西
                                 · 窗按 M₀ 标定残差外扩
```

训练分布：每个样本的相机输入按 s~p(s) 从 {D0,…,D4,D∞} 采样
（`AOSampleDegLevel`）。D∞ 训练期物化为黑帧（batch 保持稠密，
`dinf_mode="zero"`），测试期真正丢相机（batch 1，`dinf_mode="drop"`）。
声呐永不退化——声呐配置轴属于兄弟课题 F1，不混入。

### 2.3 方法空间与当前进度

| 家族 | 选题文档编号 | 做法 | 状态（2026-10-31） |
|---|---|---|---|
| 纯声呐基线 | B0 | 声呐单流单头 YOLOX-S | 配置 `configs/f2/s0_b0_sonar_only_yoloxs.py`，E06 重训中 |
| 推理期融合 | ≈B2/B3 | 双流三头 + 交叉注意力，训练期按 p(s) 退化采样 | 配置 `configs/f2/s0_cross_attn_yoloxs.py`，E06 重训中 |
| 特权蒸馏 | B1、M1–M5 | 冻结清晰教师 + branch_kd +（M4）跨平面 MIL | 模块与测试就绪（`teachers/`、`losses/`），训练配置待 S0 闸门后展开 |
| 晚融合参照 | B5 | 两个单模态检测器结果级匹配融合 | 规划中（结构上无耦合，作支路相关性参照） |

完整方法空间（B0–B6、M1–M5）定义见选题文档 §2.2；逐实验进度见
`doc/experiments.md`。

## 3. 整体架构与数据流

任何源数据集先落入同一套配对格式，之后的数据集、训练、评测代码
完全复用：

```mermaid
flowchart TB
    subgraph RAW["① 原始数据"]
        UXO["UXO 2024：ARIS 声呐 + GoPro + 位姿"]
        RSF["RSFusion：官方仓库为空，监视中"]
        UMOD["UMOD：可选"]
    end
    subgraph CONV["② 转换器 aodet/datasets/converters"]
        C1["uxo_to_paired：×3 框规则、recording 级 split"]
        C2["uxo_projection + tools/project_uxo_sonar：稳健标定 + 投影 → 声呐框与跨模态对应 π"]
        C3["rsfusion_to_paired / umod_to_paired"]
    end
    subgraph FMT["③ 统一配对格式（如 data/paired/uxo/）"]
        F1["camera.json · sonar.json · pairs.csv · frame_meta.csv · images/"]
    end
    subgraph TRAIN["④ 数据集与训练"]
        D1["PairedSonarCameraDataset + transforms：同步几何 → 退化采样 D0–D∞ → fork&pack"]
        T1["FrozenTeacher：清晰相机、no_grad"]
        L1["KD 损失：branch_kd · cross_plane_mil（M₀ 方位窗）"]
        M1["TwoStreamYOLOX 三头学生"]
        TR["tools/train.py + configs/f2/*.py"]
    end
    subgraph EVAL["⑤ 评测与记录"]
        E1["tools/test.py：逐支路 × 逐退化档 AP → ap_summary.json"]
        E2["tools/collect_results.py → results/results.csv"]
        E3["tools/correlation_table.py：φ · ΔP_miss · Spearman · ECE → correlation.csv"]
        E4["doc/experiments.md 五段式实验台账"]
    end
    UXO --> C1 --> F1
    UXO --> C2 --> F1
    RSF --> C3
    UMOD --> C3
    C3 --> F1
    F1 --> D1 --> M1
    T1 --> L1 --> M1
    TR --> M1
    M1 -->|"checkpoint"| E1
    E1 --> E2 --> E4
    E1 --> E3 --> E4
```

各阶段要点：

1. **转换**：源数据集各写一个转换器（UXO 已全流程跑通，2573 对 +
   投影报告；RSFusion/UMOD 转换器就绪待数据）。声呐框没有人工标注，
   由 `uxo_projection` 用官方位姿做逐 recording 稳健标定后投影生成，
   GT 语义 = 可见回波区域（框钳制到回波尺度）。
2. **配对格式**：四个 JSON/CSV + 图片目录就是全部状态；`frame_meta.csv`
   把每个 sample 键到两模态 COCO image id、split 与退化种子。
3. **训练**：管线顺序固定「同步几何 → 退化采样 → fork&pack」，
   无 Mosaic/MixUp（师生增强必须同步）；多种子用 `tools/run_seeds.sh`。
4. **评测**：`tools/test.py` 逐退化档覆盖数据集 `deg_level` 字段后
   推理，逐支路落盘预测与 AP；相关性工具只消费这些落盘文件——
   所有方法共享同一条评测路径。
5. **记录**：每个实验按五段式（设置/结果/解读/结论）进
   `doc/experiments.md`，当日踩坑与排查链在 `doc/logs/`。

## 4. 目录结构

| 路径 | 内容 |
|---|---|
| `aodet/datasets/` | 配对格式读写（`paired_format.py`）、`PairedSonarCameraDataset`、转换器（`uxo_to_paired`、`uxo_projection`、`rsfusion_to_paired`、`umod_to_paired`）、变换（同步几何、相机退化 D0–D∞、fork & pack） |
| `aodet/models/` | `TwoStreamYOLOX`（三头，`detectors/`）、融合模块（`fusion/`：cross-attn / concat / none）、KD 损失（`losses/`：`branch_kd`、`cross_plane_mil`）、冻结教师（`teachers/`）、`PairedDataPreprocessor` |
| `aodet/evaluation/` | 逐支路 COCO AP（`per_branch_metric.py`）、支路误差相关性（`branch_correlation.py`：φ、ΔP_miss、Spearman、朴素贝叶斯融合、ECE） |
| `aodet/geometry/` | M₀ 相机→方位映射 + 翻转镜像（`m0.py`）、ARIS 波束表 |
| `configs/` | `_base_`（模型/数据/调度/运行时）+ `f2/` S0-lite 配置（B0、cross-attn） |
| `tools/` | `train.py`、`test.py`（逐支路 D0–D∞ 扫描）、`project_uxo_sonar.py`（标定+投影）、`check_gt_validity.py`、`make_degraded_testsets.py`、`run_seeds.sh`、`collect_results.py`、`correlation_table.py`、第三方 pin、RSFusionDet 监视 |
| `tests/` | 87 项测试：退化物理、格式往返、几何、指标、模型、UXO 转换器与投影、端到端管线与工具 |
| `env/` | 安装脚本 + `ENVIRONMENT.md` 版本记录 |
| `doc/` | 选题方案、实验台账 `experiments.md`、过程日志 `doc/logs/`、证据存档 `doc/evidence/` |
| `data/paired/` | 转换后的配对数据（UXO：2573 对 + 投影报告与拼图） |

## 5. 快速开始

```bash
source .venv/bin/activate
python -m pytest tests/ -q          # 87 passed

# S0-lite（UXO sample 已转换至 data/paired/uxo/）
python tools/train.py configs/f2/s0_b0_sonar_only_yoloxs.py --seed 0
# mmengine 只写 epoch_N.pth + last_checkpoint（没有 latest.pth 软链）
ckpt=$(cat work_dirs/s0_b0_sonar_only_yoloxs_seed0/last_checkpoint)
python tools/test.py  configs/f2/s0_b0_sonar_only_yoloxs.py "$ckpt" \
    --out-dir work_dirs/s0_b0_sonar_only_yoloxs_seed0
python tools/collect_results.py work_dirs --out results/results.csv
```

## 6. 配对格式（所有源数据集共用一套布局）

```
<root>/camera.json  sonar.json  pairs.csv  frame_meta.csv
<root>/images/camera/...  images/sonar/...
```

`frame_meta.csv` 把每个 `sample_id` 键接到两个 COCO image id、split
与退化种子；`read_paired` 负责 CSV 类型归一。同步翻转仅当声呐方位轴
0° 居中时合法（`sonar_zero_centred=True`）。

## 7. 退化协议

`aodet/datasets/transforms/camera_degradation.py`：物理有序算子
（衰减 → 波长吸收 → 散射雾幕 → 模糊），级别 D0–D4、D∞（全黑/传感器
缺失），按 (sample_id, level, seed) 确定性生成。训练按 p(s) 抽级别
（F2 §5.2），D∞ 物化为黑帧（`dinf_mode="zero"`，batch 保持稠密）；
测试期 D∞ 丢弃相机（`dinf_mode="drop"`，batch 1）。离线固定测试集：
`tools/make_degraded_testsets.py`（逐级别 sha256 manifest）。

## 8. 评测

基于保存的预测文件按退化级别出逐支路 AP（`tools/test.py` →
`pred.<branch>.<level>.json` + `ap_summary.json`）；声呐支路同时报
全量 GT 与 blob 子集两个口径（`gt=all` / `gt=blob`）——投影框带
标定距离维误差，blob 子集是干净参照。支路误差相关性
（φ、带 bootstrap 置信区间的条件漏检升幅 ΔP_miss、Spearman 置信
相关、朴素贝叶斯融合存在性 ECE）实现在
`aodet/evaluation/branch_correlation.py`，经 `tools/correlation_table.py`
对 run 目录输出 `correlation.csv`。

## 9. 状态（2026-10-31）

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
