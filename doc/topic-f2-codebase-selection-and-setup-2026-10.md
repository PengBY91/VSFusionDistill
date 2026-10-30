---
title: 选题 F2 代码库选型与搭建方案（2026-10-29）
aliases: ["选题 F2 代码库选型与搭建方案（2026-10-29）", "F2 代码库方案"]
created: 2026-10-29
updated: 2026-10-29
type: plan
status: draft
status_note: "选型建议，未评审；S0 数据路线待决（第 6 节）"
hub: [research-portfolio, f1-sonar-low-label]
related: [F2, F1]
tags: [acoustic-optic-fusion, detection, deep-learning, multi-modal]
keywords: [codebase, mmdetection, yolox, knowledge-distillation, RSFusionDet, UXO, reproducibility]
---

# 选题 F2 代码库选型与搭建方案

本页写于 2026-10-29，服务于 [[plans/2026-10/topic-f2-degradation-robust-fusion-detection-distillation-2026-10|F2 草案 v0.1]] 进入实施前的代码库准备。所有仓库信息在当天通过 GitHub API、Zenodo API、PyPI 和 Hugging Face API 核实。WebSearch 在本环境不可用，因此检索只覆盖 GitHub 仓库搜索和库内已知链接。未克隆、未运行任何候选代码。

## 0 结论先行

1. **RSFusionDet 的代码和数据都没有发布。** 官方仓库是空仓库（0 个提交）。F2 的 S0 写的是"跑通 RSFusionDet 官方代码"，现在无法执行。F1 把 RSFusion 当主数据，同样受影响。库内笔记"代码与数据开源"的说法不成立（第 1.1 节）。
2. **UMOD 公开部分只有约 259 对，不是 4,000 对。** 用户在 2026-08 至 10 月多次报告，作者没有回复。UMOD 不足以承担外部测试（第 1.2 节）。
3. **当前唯一能下载、又适合 F2 的真实声光配对数据是 Dahn UXO。** 它在 Zenodo 公开，BSD-3 许可，96.4 GB，有相机标注，还有声呐和目标的位姿，可以投影出声呐框和跨模态对应（第 1.4 节）。
4. **底座建议用 MMDetection 3.3，默认检测家族用 YOLOX。** 依据有四条。第一，RSFusionDet 作者的整套工具链是 OpenMMLab，代码日后发布时大概率能直接接入。第二，YOLOX 和 DINO 两个家族都在同一框架内，F2 §5.1 "二选一"的决定可以推迟而不用换框架。第三，F1 W2 本来就在 ultralytics 和 mmdetection 之间选，两边共用一套管线最省事。第四，许可是 Apache-2.0。版本必须锁定（第 4.1 节）。
5. **自写代码集中在四处：双输入数据模型、相机退化算子、蒸馏损失、逐支路评测与相关性指标。** 这四处都要先写单元测试，再接入训练（第 4.5 节）。其余部分只写 glue code，沿用[[comparisons/turbidity-aware-code-audit-and-reuse-principles|代码复用原则]]。

## 1 核验发现

### 1.1 RSFusionDet：仓库为空

- `LEFTeyex/RSFusionDet` 创建于 2025-03-31。API 返回 "This repository is empty"，size 为 0，没有提交。
- 三个 issue 都在问什么时候开源：2025-08-05 一个，2026-08-27 一个（Hugging Face 团队邀请上传），2026-08-31 一个。作者没有回复。
- Hugging Face 上没有作者账号下的数据集或模型，也搜不到 RSFusion。
- **作者的工具链**：`LEFTeyex` fork 过 mmdetection、mmengine、mmcv、mmyolo。已发布的 [U-DECN](https://github.com/LEFTeyex/U-DECN)（TGRS 2025，水下检测）基于 mmdet 3.2.0、mmengine 0.10.1、mmcv 2.0.1 和 PyTorch 1.13.1。RSFusionDet 用 DINO 双分支，mmdet 3.x 自带 DINO，因此日后发布时大概率也是这套组合。
- **影响**：F2 的 S0、B2 和 B3 都依赖官方代码。F1 §6 写的"RSFusion 已开源"需要改。[[2026-rsfusiondet-rgb-sonar-detection-heu|RSFusionDet 笔记]]里的"可复现性 4.5/5"也需要下调。

### 1.2 UMOD：公开数据不完整

- `UM-Research-Ng/UMOD-dataset` 仓库只有 demo 子集，全量数据走百度网盘。
- 用户在 issue #1、#3、#4 里反复报告：网盘里只有 259 对；早些时候有人下载到 700 对；第 6 类缺失，第 2 类和第 8 类各只有个位数样本。作者没有回复，有用户发邮件也没有回音。
- **影响**：F2 §6.1 把 UMOD 列为可选的外部测试集，目前只能当作 objectness 的小样本 sanity check。

### 1.3 SCFusion：代码可用，参考价值低

- `NWPU-ASVLab/SCFusion` 是纯 PyTorch 实现，带 UMOD 预处理脚本。
- 模型结构是两个轻量骨干，逐层融合后接一个检测头，只输出声呐框，没有逐模态的检测头。
- 一个 issue 指出仓库里"只有一个小模型，数据集只有 200 对"。
- 可以借用的只有 `tools/prepare_umod.py`。

### 1.4 Dahn UXO：当前唯一可用的真实配对数据

| 项目 | 内容 |
| --- | --- |
| 来源 | Dahn 等，OCEANS 2024，DOI:10.1109/OCEANS55160.2024.10754316；数据在 [Zenodo 13778485](https://zenodo.org/records/13778485)，导出脚本在 [dfki-ric/uxo-dataset2024](https://github.com/dfki-ric/uxo-dataset2024) |
| 许可与规模 | BSD-3-Clause，公开访问。共 96.4 GB：`data_processed.7z` 61.5 GB、`data_export_recordings.7z` 16.7 GB、`data_export_polar.7z` 16.5 GB、`sample.7z` 1.7 GB，另有标定与 3D 模型 |
| 内容 | ARIS 3000 声呐加 GoPro Hero 8，近 100 条轨迹，74,000 多帧，3 类 UXO；有原始声呐帧和极坐标变换后的声呐帧；GoPro 帧有标注；有声呐和目标的位姿、标定矩阵和带纹理的 3D 模型 |
| 对 F2 的价值 | 相机侧有框。声呐侧没有框，但目标位姿已知，可以投影出声呐框，同时得到跨模态对应 π（CMHD 用的 UXO† 就是这种投影标注）。相机是清水图像，适合施加 D0–D4 合成退化。CMOLD 用的也是这份数据（约 4,700 对），结果可以直接对照 |
| 已知问题 | README 注明标签坐标要乘 3。issue 报告 JSON 框与原始 GoPro 帧对不上，另有外参和相机模型的疑问。GoPro 没有同步时间戳，靠运动起点人工对齐。场景是吊车受控环境，每帧单目标，只有 3 类，检测任务偏容易，AP 可能饱和 |

### 1.5 其他

- [[2026-degradation-aware-foundation-fusion-umod|Degradation-aware Foundation Fusion]] 没有代码，D0–D4 退化算子需要自己写。
- [[2026-cross-modality-object-level-kd-sonar-detection|CMOLD]] 和 [[2026-cmhd-lightweight-sonar-rgb-heterogeneous-distillation|CMHD]] 都没有代码。
- [KD-YOLOX-ViT](https://github.com/remaro-network/KD-YOLOX-ViT)（Apache-2.0，98 星）是 CMHD 对比过的同模态蒸馏基线，基于 Megvii YOLOX。读源码后确认，它的在线蒸馏每个 batch 都把增强后的图像写盘，再用子进程调用 `tools/demo.py` 跑教师；离线蒸馏则关闭数据增强。所以它只能用作 YOLOX 蒸馏损失的参考，不能作为底座。

## 2 F2 对代码库的硬要求

| 编号 | 要求 | 来源 |
| --- | --- | --- |
| R1 | 双输入、不配准：相机和声呐各一个骨干，各自的框各自监督 | 草案 §5.1 |
| R2 | 三个检测头：相机头、融合声呐头、不和相机耦合的声呐头；可选跨模态匹配头 | 草案 §5.1、§6.4 |
| R3 | 融合模块可替换：拼接、CAFusion 式交叉注意力、不融合（B5） | 草案 §2.2 |
| R4 | 相机退化 D0–D4、D∞ 和"只有色偏"档只作用于相机；测试集每档固定一份 | 草案 §2.1、§6.1 |
| R5 | 教师看清晰视角、学生看退化视角，两者的几何增强必须一致 | 草案 §5.2 |
| R6 | 同一检测家族承载 B0–B6 和 M1–M5，统一输入尺寸、训练轮数和调参预算 | 草案 §6.2 |
| R7 | 评测要能按支路和退化档出 AP，并在匹配对上计算 ΔP_miss、φ 系数、Spearman 相关和独立融合 ECE | 草案 §2.3、§6.4 |
| R8 | 3 个随机种子，配置、git 哈希和结果 CSV 可追溯；与 F1 共用划分、M₀ 和评测 | 草案 §2.3、F1 §7 |
| R9 | RSFusionDet 日后发布时，B2/B3 能以最小改动接入 | 第 1.1 节 |

## 3 候选评估

### 3.1 底座框架

| 候选 | 许可与维护 | R1–R3 双输入多头 | R6 家族 | R9 接 RSFusionDet | 判断 |
| --- | --- | --- | --- | --- | --- |
| **MMDetection 3.3.0**（open-mmlab，33k 星） | Apache-2.0；最后推送 2024-08，事实上停更 | 需自写双图 transform 和 data preprocessor；多头和多损失在 `BaseDetector` 子类里写，较顺 | YOLOX 有 nano 到 x 全系 COCO 权重；DINO 有 r50 4scale 等配置 | 作者工具链，几乎可以直接接入 | **推荐** |
| Megvii YOLOX（10.7k 星）＋ KD-YOLOX-ViT | Apache-2.0；最后推送 2025-06 | 代码小，改起来直接；评测和配置基础设施简陋 | 只有 YOLOX | DINO 基线要放在另一个环境，"同一框架"做不到 | **备选**（第 5 节） |
| Ultralytics（63k 星）及 RGBT 分支（如 YOLOv11-RGBT） | AGPL-3.0，产品部署有传染风险；版本更新很快 | RGBT 分支默认两路像素配准，单头共享标签，与声光不配准、逐模态标签的前提相反 | YOLO 系列 | 不兼容 | 不采用 |
| detrex / IDEA DINO（detectron2 系） | Apache-2.0 | 可做，但更重 | 只有 DETR 系 | 不同栈 | 只有选定 DETR 家族、且 mmdet 不够用时才考虑 |
| 从零写（如 SCFusion 式） | — | 自由 | — | — | 违反复用原则，不采用 |

### 3.2 只读参考仓库（固定 commit，放在 `third_party/`，不 import）

| 仓库 | 借用什么 |
| --- | --- |
| [KD-YOLOX-ViT](https://github.com/remaro-network/KD-YOLOX-ViT) | YOLOX 的 objectness 和类别软损失写法；同模态蒸馏基线的超参 |
| [Megvii YOLOX](https://github.com/Megvii-BaseDetection/YOLOX) | 核对 mmdet 版 YOLOX 的 SimOTA 和损失权重与原版是否一致 |
| [dfki-ric/uxo-dataset2024](https://github.com/dfki-ric/uxo-dataset2024) | 官方导出脚本、标定和位姿读取，用来投影声呐框（不自己重写投影） |
| [ProbEn](https://github.com/Jamie725/Multimodal-Object-Detection-via-Probabilistic-Ensembling)（Apache-2.0） | B5 晚融合与条件独立融合规则，即 §6.4 第 4 项下游代理的公式 |
| [CRKD](https://github.com/Song-Jingyu/CRKD)（Apache-2.0，mmdet3d 系） | 跨模态 objectness 对齐损失的写法（CSRD） |
| [SCFusion](https://github.com/NWPU-ASVLab/SCFusion) | UMOD 预处理脚本 |
| RSFusionDet | 发布后作为 B2/B3 的官方实现；在此之前每两周检查一次仓库 |

不建议依赖 mmrazor：它最后推送于 2024-06，只适配 mmdet 3.0 时代。F2 需要的蒸馏损失只有几十行，直接写在自定义检测器里更容易审。

## 4 推荐方案

### 4.1 环境：锁定版本

| 组件 | 版本 | 理由 |
| --- | --- | --- |
| Python | 3.10 | mmcv 2.1.0 预编译轮子覆盖 cp38–cp311；本机 WSL 系统 Python 是 3.12，必须用独立的 conda 或 uv 环境，不与 ROS 环境混用 |
| PyTorch | 2.1.2 + cu121 | mmcv 2.1.0 有 cu121/torch2.1.0 的 Linux 和 Windows 轮子（2026-10-29 核实）；RTX 4070（Ada 架构）需要 CUDA 11.8 以上 |
| mmcv | 2.1.0 | mmdet 3.3.0 在 `__init__.py` 里断言 mmcv 低于 2.2.0，所以不能装 2.2.0（这是 mmdet 用户最常见的安装失败） |
| mmengine | 0.10.7 | 满足 mmdet 3.3 要求的低于 1.0.0 |
| mmdet | 3.3.0 | 最后一个正式版 |
| numpy | 1.26.x | mmcv 2.1 针对 numpy 1.x 编译 |
| 评测 | pycocotools 2.0.x；faster-coco-eval 1.8 作为加速选项，需先和 pycocotools 对拍 | — |

环境建好后保存 `pip freeze`、`pip check`、`torch.version.cuda` 和 `nvidia-smi` 输出，与[[plans/2026-10/acoustic-optic-month1-environment-runbook-2026-10|仓库环境手册]]的做法一致。框架停更的风险用锁版本来控制。F2 的研究周期不超过一年，这个风险可以接受。

### 4.2 检测家族：默认 YOLOX

- 教师用 YOLOX-L，学生用 YOLOX-S，可以下探到 Tiny 和 Nano；都从 mmdet 的 COCO 权重初始化，声呐骨干也一样。这与 CMOLD 和 CMHD 的 YOLOX 谱系一致，objectness 分支可以直接对应草案 §5.3 的 L_KD。
- DINO 留在同一框架。只有 RSFusionDet 发布后要复用它的匹配头时，才启用 DETR 家族。这正是草案 §9 的默认设置，现在多了一条理由：没有官方代码，DETR 家族的主要好处暂时不存在。

### 4.3 仓库结构（F1 和 F2 共用）

```text
ao-det/
  env/                      # environment.yml、pip-freeze.txt、安装说明
  aodet/                    # 通过 mmengine custom_imports 注册，不改 site-packages 里的 mmdet
    datasets/
      paired_dataset.py     # PairedSonarCameraDataset：两张图、两套 instances、对应关系 π、元数据
      converters/           # rsfusion_to_paired.py、uxo_to_paired.py、umod_to_paired.py
      transforms/
        paired_geometric.py # 两路各自缩放；翻转要么两路同步，要么都不做（见 4.4 第 2 条）
        camera_degradation.py  # D0–D4、D∞、只有色偏；由 (sample_id, level, seed) 确定
        pack_paired.py
    models/
      data_preprocessors/paired_preprocessor.py
      detectors/two_stream_yolox.py  # 两个骨干、可选融合、三个头
      fusion/               # concat.py、cross_attn.py（CAFusion 式）、none.py
      losses/               # branch_kd.py（objectness 和带温度的类别 KL）、cross_plane_mil.py（L_xp）
      teachers/             # frozen_teacher.py（在线）、teacher_cache.py（离线，需通过等价测试才能用）
    evaluation/
      per_branch_metric.py  # 按 支路 × 退化档 输出 COCO AP
      branch_correlation.py # ΔP_miss、φ 系数及自助法置信区间、Spearman、朴素贝叶斯融合 ECE
    geometry/m0.py          # 只读取 F1 产出的 M₀，不重新拟合
  configs/
    _base_/                 # 数据集与调度
    f2/                     # s0_*、b0–b6、m1–m5，每个方法一个配置
  tools/                    # train.py、test.py（包 mmengine Runner）、make_degraded_testsets.py、
                            # cache_teacher.py、run_seeds.sh、collect_results.py → results.csv
  tests/                    # 见 4.5
  third_party/              # git submodule，固定 commit，只读
```

### 4.4 八个关键设计决定

1. **统一的配对数据格式。** 每个数据集只写一个转换器，转成"两份 COCO 风格标注加一张对应表"：`camera.json`、`sonar.json`、`pairs.csv`（camera_ann_id、sonar_ann_id、sample_id），另外写一份逐帧元数据（光照、量程、增益、退化档）。训练和评测代码只认这个格式。RSFusion 一旦可得，只需加一个转换器。
2. **几何增强。** 两路不配准，所以缩放、裁剪可以各自独立。相机水平翻转对应声呐方位镜像，只有满足两个条件时才能同步做：声呐图以 0° 方位居中，并且 M₀ 的偏置项也随之取反。否则一律不做翻转。**F2 的所有方法都关闭 Mosaic 和 MixUp**。原因是拼接增强会破坏配对，也会让教师缓存失效。这样做的代价是 YOLOX 的绝对 AP 偏低，但各方法之间的比较仍然公平。
3. **退化算子是风险最高的自写组件。** 测试集按档位离线生成一次，落盘并记录哈希，所有方法共用这一份。训练时在线生成，种子由样本 ID 决定。教师与学生的分叉点放在几何增强之后：先做完增强，复制一份清晰图给教师，再对另一份施加退化给学生。
4. **教师先在线运行，再考虑离线缓存。** 先用冻结教师在线前向（fp16、no_grad）作为参考实现，这样增强一致性自动成立。离线缓存只是加速手段，必须先在一个子集上和在线版本对比，逐元素误差在容差内才能启用。草案 §5.2 的"离线缓存"作为第二步做，不作为第一步。
5. **蒸馏损失写在自定义检测器里。** objectness 和类别 KL 参照 KD-YOLOX-ViT 的写法。L_xp 需要声呐特征处于"距离 × 方位"的矩形网格上，这样沿距离取最大才简单。声呐输入用极坐标矩形图还是扇形图，要在 W2 结合团队声呐只能输出图像的情况决定（见 log 2026-10-29 的声呐数据源判定）。
6. **评测和训练分开。** 评测脚本只读预测结果文件（每个支路、每个退化档各一个 json），先输出 AP，再输出相关性指标。这样同一批预测可以反复分析，不用重新推理。
7. **实验管理。** 每次运行的目录里保存完整配置、git 哈希、种子、环境摘要和 `results.csv` 的一行。3 个种子由 `run_seeds.sh` 统一下发。暂不引入 wandb，需要时再加 TensorBoard。
8. **与 F1 共用。** 建议 F1 在 W2 选定 mmdetection。F1 只用单输入路径，也就是 `paired_dataset` 的声呐半边。划分脚本、M₀ 和评测在两个课题之间共用一份代码。

### 4.5 先写测试的组件

| 测试 | 断言 |
| --- | --- |
| `test_degradation.py` | 档位越高，对比度和直接项透射率单调下降；"只有色偏"档的亮度和结构不变；同一 (sample_id, level, seed) 两次生成逐像素一致；D∞ 输出为全零或被丢弃 |
| `test_paired_geometry.py` | 同步翻转后，经 M₀ 映射的方位窗与声呐框仍然对应；关闭翻转时管线不碰方位 |
| `test_converter_roundtrip.py` | 转换后的框数、类别分布和对应表与源数据一致；UXO 的"坐标乘 3"在这里显式处理并做断言 |
| `test_teacher_cache.py` | 离线缓存的教师输出与在线前向一致（只有离线模式启用前需要） |
| `test_branch_correlation.py` | 在构造的独立漏检数据上 φ≈0、ΔP_miss≈0；在构造的完全相关数据上 φ=1；ECE 与手算一致 |
| `test_per_branch_metric.py` | 用 pycocotools 官方样例对拍 AP |

这吸取了 [[comparisons/turbidity-aware-code-audit-and-reuse-principles|代码审计]]的教训：自写的退化流水线曾经把非单调结果打印成"通过"。所以测试失败时脚本必须返回非零退出码。

## 5 备选方案：纯 YOLOX

如果 mmcv 环境在 3 个工作日内搭不起来，例如目标机器的驱动或 CUDA 不兼容，就改用 Megvii YOLOX 原版加自写的双输入数据集。

- 好处：代码量小，实习生容易完全掌握；纯 PyTorch，没有编译算子。
- 代价：B2/B3 无法和学生网络放在同一框架里；评测、配置和日志基础设施要自己补；YOLOX 的 `Exp` 文件不如 mmengine 配置容易复用。
- 不管选哪种，第 4.4 节的八个决定和第 4.5 节的测试都不变。

## 6 S0 的数据路线（需要决策）

S0 的问题是"相机退化时，交叉注意力融合是否会拖累声呐支路"。在没有 RSFusionDet 的情况下有三条路：

| 路线 | 做法 | 代价 | 能宣称的结论 |
| --- | --- | --- | --- |
| a 联系作者 | 由团队向 RSFusionDet 作者发邮件索取代码和数据；每两周检查一次仓库 | 不确定，可能一直等不到 | 原定的"RSFusionDet 在相机退化下"的结论 |
| b UXO 上自建 | 在 UXO 上用 mmdet 搭双流 YOLOX 加交叉注意力融合，投影出声呐框，做推理期退化扫描 | 约 2–3 周；需先下载 sample（1.7 GB）核验标注，再决定是否下载全量 96 GB | 降级为"交叉注意力融合检测器在合成相机退化下"；单目标、3 类、受控环境 |
| c 等 H1 | 用团队自采的浊度阶梯数据 | 最早约 2027-03 | 真实浊度，但没有清晰与退化的逐像素配对 |

建议 a 和 b 并行。b 的工作同时就是第 4 节的代码库搭建和验收，即使 a 成功，搭好的管线也能直接复用。

## 7 前三周的搭建顺序

| 周 | 内容 | 验收 |
| --- | --- | --- |
| W1 | 建环境并锁定版本；mmdet YOLOX-S 在 COCO val 子集上复现官方精度（确认安装正确）；下载 UXO `sample.7z`，用官方脚本读取并核验标注和"乘 3"问题 | 环境记录；复现数字与官方差距在 1 AP 以内；UXO 标注核验报告 |
| W2 | 写配对格式和 UXO 转换器、声呐框投影、退化算子；第 4.5 节测试全部通过；离线生成退化测试集 | 测试全绿；退化档位样张交给非作者成员目检 |
| W3 | 双流 YOLOX-S 的三个头；B0（只用声呐）和交叉注意力融合各训 1 个种子；跑 D0–D4 推理期扫描；输出逐支路 AP 和相关性表 | S0-lite 的第一张表；决定是否下载全量和扩到 3 个种子 |

## 8 算力

- 本机是 RTX 4070 Laptop，8 GB 显存，只够做冒烟测试和小批量调试。
- YOLOX-L 教师在线前向加双流 YOLOX-S 学生，640 输入、batch 8，估计需要 24 GB 级显卡。草案 §6.5 假设的单卡 4090 还没有落实，和 F1 W1 的"确认可用 GPU"是同一件事。
- UXO 全量 96 GB，解压后更大，需要提前规划存储位置。

## 9 待决问题

1. S0 走哪条数据路线（第 6 节）。路线 a 要由团队出面联系作者。
2. 是否同意 F1 W2 也选定 mmdetection，两个课题共用 `ao-det` 仓库。
3. 训练用哪台机器：本机只适合调试，需要确认 24 GB 级显卡和存储空间。
4. 库内 RSFusionDet 笔记、UMOD 笔记和 F1 v1.4 中"已开源"的表述，是否按第 1 节更正。本页没有改动这些文件。

## 10 关联页面

- [[plans/2026-10/topic-f2-degradation-robust-fusion-detection-distillation-2026-10|F2 草案]] · [[plans/2026-10/intern-project-2026-f1-sonar-low-label-adaptation-2026-10|F1 v1.4]] · [[projects/f1-sonar-low-label|F1 项目主页]]
- [[comparisons/turbidity-aware-code-audit-and-reuse-principles|代码审计与复用原则]] · [[plans/2026-10/acoustic-optic-month1-environment-runbook-2026-10|仓库环境手册]]
- 数据与方法笔记：[[2026-rsfusiondet-rgb-sonar-detection-heu|RSFusionDet]] · [[2026-vsafdet-umod-visual-sonar-detection|VSAFDet/UMOD]] · [[2026-scfusion-sonar-camera-detection|SCFusion]] · [[2026-cross-modality-object-level-kd-sonar-detection|CMOLD]] · [[2026-cmhd-lightweight-sonar-rgb-heterogeneous-distillation|CMHD]] · [[2022-chen-proben-rgbt-probabilistic-ensembling|ProbEn]] · [[2024-zhao-crkd-camera-radar-distillation|CRKD]]
