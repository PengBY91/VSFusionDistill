---
title: 选题 F2 相机退化下的声光融合检测：清晰视角特权蒸馏与支路独立性（草案）
aliases: ["选题 F2 相机退化下的声光融合检测：清晰视角特权蒸馏与支路独立性（草案）", "F2 草案"]
created: 2026-10-19
updated: 2026-11-01
type: plan
status: draft
status_note: "idea-draft：讨论形成，未评审；启动条件 G0–G2 均未满足，不占用 A/B 与 F1 资源"
hub: [research-portfolio, f1-sonar-low-label]
version: "0.1"
parent_topic: "F（推理期声光双输入方向的延伸；F 详案 §3 步骤 5 的扩展）"
related: [F1, C, H1]
tags: [acoustic-optic-fusion, detection, deep-learning, multi-modal, turbidity]
keywords: [knowledge-distillation, heterogeneous-distillation, privileged-information, modality-dropout, degradation-aware, error-correlation, RSFusion, UMOD]
---

# 选题 F2：相机退化下的声光融合检测——清晰视角特权蒸馏与支路独立性（草案）

> 水下机器人上线时同时有声呐和相机，但相机的可靠性会随浊度和光照连续变化。本草案研究一个轻量的声光融合检测学生网络。它向一个大容量、看清晰相机的融合教师学习，能否在相机从清晰（D0）到完全失效（D∞）的整个范围内，保住两条支路的检测能力。同时，草案还要测量这种"互增强"让两条支路的误差相关了多少。原因是下游的 C 和 A 把逐模态输出当作独立证据。
>
> 来源：2026-10-19 围绕 [[projects/f1-sonar-low-label|F1]] 的讨论。起点是用户提出的想法：把异构蒸馏（[[2026-cross-modality-object-level-kd-sonar-detection|CMOLD]]）与跨模态视觉基础模型（[[2026-degradation-aware-foundation-fusion-umod|Degradation-aware Foundation Fusion]]）结合起来。讨论中被修正的判断记在第 12 节。

## 0 定位、边界与当前状态

### 0.1 为什么不并入 F1

[[plans/2026-10/intern-project-2026-f1-sonar-low-label-adaptation-2026-10|F1 v1.4]] 研究的是：声呐换配置后，人工预算花在哪里更划算。可选的有三处：直接标注、声呐自训练、核验相机伪标签。学生网络只用声呐输入，主结果是性能—人工小时曲线和决策边界 RQ-d（F1 的研究问题之一）。F2 研究的是另一件事：推理时有两种输入、相机还会退化，应该怎么融合、怎么蒸馏。

两者的研究问题不同，自变量也不同。F1 的自变量是声呐配置，F2 的自变量是相机可靠性。两者的推理输入同样不同。如果把 F2 并入 F1，会打断 F1 §3 "每步只加一个变量"的归因顺序。它还会挤占 F1 已经很紧的 170–330 GPU 小时预算。因此 F1 只吸收两项低成本借用（第 8.4 节），改不改由指导人决定。

### 0.2 为什么不单独立为第七个选题

[[plans/2026-10/acoustic-optic-six-topics-internal-review-2026-10|六选题评审稿]] §10 规定"新增方法的选择必须挤出或结束一个在役方法课题，而非叠加"。它还规定"E、F 不同时扩展为第三、第四条方法主线"。F2 放在 F 名下，是 F 详案 §3 步骤 5（"推理分别测试声呐单独输入及声光共同输入"）的扩展。它对应"B 加 F"组合，或者 C 的配套检测研究，所以不违反组合规则。

### 0.3 与其他课题的边界

| 课题 | 关系 | 边界 |
| --- | --- | --- |
| [[plans/2026-10/topic-f-low-label-sonar-adaptation-detailed-2026-10\|F]] / F1 | 父课题与兄弟子课题 | F1 管声呐配置轴与标注预算，F2 管相机可靠性轴与推理期融合；共用 RSFusion 划分、评测管线和名义映射 M₀，不重复宣称同一检测收益 |
| [[plans/2026-10/topic-c-persistent-target-mapping-detailed-2026-10\|C]] | 主要使用方 | F2 交付检测器版本和逐模态检出曲线；C 的跨帧身份维护和物理检出模型不属于 F2 |
| [[plans/2026-10/topic-a-direction-aware-localization-detailed-2026-10\|A]] | 原则借鉴 | A 不消费检测输出；F2 只向 A 提供退化扫描协议和"耦合会让检验能力消失"的感知侧证据（第 8.1 节） |
| [[plans/2026-10/topic-b-ambiguity-aware-spatial-mapping-detailed-2026-10\|B]] | 无实质交集 | B v2.3 已在仿真中否定双目关联对仰角候选的收缩，F2 的目标级匹配改变不了这一几何结论（第 8.2 节） |
| [[plans/2026-10/intern-project-2026-h1-rov-platform-testbed-2026-10\|H1]] | 数据供给方 | H1 的受控浊度声光配对数据是 F2 唯一的真实退化来源；F2 不宣称 H1 的作业包络结论，数据使用需与 H1 约定 |
| [[2026-degradation-aware-foundation-fusion-umod\|Degradation-aware Foundation Fusion]] | 最近邻外部工作 | 该文做识别（真值框裁剪），冻结编码器加 softmax gate；F2 做检测，主张落在蒸馏与独立性，不提出新的 gate |
| G3 浊度感知模态调度线 | 历史同题 | 吸取其教训：不以"提出一个 gate"为主张，不用推理期特征缩放（见 [[comparisons/sonarsweep-wrapper-variant-failure-analysis\|SonarSweep 失败分析]]、[[comparisons/research-topics-genealogy-2026-10\|课题谱系]] §4） |

### 0.4 当前状态

以下状态截至 2026-11-01，细节见 `doc/f2-rethink-2026-11-01.md` 与 `doc/experiments.md`。

- RSFusionDet 官方代码与数据不可得，S0 数据路线定为 b（UXO 自建）。第 3 节、第 6.1 节中以 RSFusion 为主数据的写法保留为原始设计，实际执行以复盘页为准。
- 在 UXO sample（5 条录音）上完成了 E01–E10：管线、GT 与训练配方已修好；E10 首次跑了 M3 对 B3，但声呐支路一直在地板上，G0（RQ-0）状态为不可测，不是通过。
- 2026-11-01 复盘后：研究问题不变；主假设收窄为 H-cls（光学的特权信息主要是类别，需跨模态路由才能进声呐头）；测量平台迁到全量 UXO 类别均衡划分（E11 已预登记）。
- 启动条件 G1（H1 浊度数据）与 G2（C 的下游需求）仍未满足，E11 计入 G0 范围，E12 起超出 G0、需明确授权（复盘 §0、§7）。
- 2026-11-02 二次评估（复盘 §9）：主线写定为"先证明相机信息能进声呐支路（融合活性、融合余量、互毒害），再做特权/容量归因，再测支路相关性"，新颖性排序沿用本页 §11.1（RQ-4 > 2×2 > 跨平面）。三条新风险：现有融合惰性，G0 可能是架构性阴性；E11 按 CMHD 同协议读数大概率落天花板档；跨模态 logit 蒸馏在单目标数据上近似重复监督。G0 为否时按本页 F2-M0 收缩。

## 1 问题陈述：六环链条

和 F1 一样，每一环是一个可以单独检验的命题，标明已有证据还是待检验假设。

**Q1（部署条件）**：上线时有两路输入，相机可靠性是连续退化，不是开关。

证据有三条。第一，在 [[2026-degradation-aware-foundation-fusion-umod|Degradation-aware Foundation Fusion]] 的 D0–D4 合成退化上，目标检测器 YOLO11n（YOLO 系列的轻量版本）在 D4 的 mAP@0.5 只剩 0.0281（约 4%）。第二，H1 v3.0 的预测模型给出了相机可用距离随浊度和灯档变化的曲线。曲线依据的[[concepts/underwater-image-formation-model|成像模型]]认为，直接项按 exp(−c·r) 衰减，后向散射随灯照亮的水体增加。第三，团队平台有已集成的双目、云台相机和前视声呐，所以推理期双输入是实际条件。

**Q2（两条现有路线各占一端）**：现有方法只覆盖可靠性谱的两个端点。
- 纯声呐蒸馏路线（[[2026-cross-modality-object-level-kd-sonar-detection|CMOLD]]、[[2026-cmhd-lightweight-sonar-rgb-heterogeneous-distillation|CMHD]]）对应 D∞。部署时不用相机，所以相机清晰时能提供的信息也浪费了。
- 推理期融合路线（[[2026-rsfusiondet-rgb-sonar-detection-heu|RSFusionDet]]、[[2025-uamfdet-acoustic-optical-detection-hrbeu|UAMFDet]]、[[2026-aomnet-acoustic-optical-joint-detection|AOMNet]]、[[2026-scfusion-sonar-camera-detection|SCFusion]]）在名义条件下训练和评测，没有系统测过相机退化时的行为。
- 识别层面的退化感知 gate（基础模型那篇）已证明"训练期见过退化"是权重重分配的必要条件，但只做了真值框裁剪的识别，没做检测。

**Q3（干净条件下互增强空间小）**：清水条件下，特征融合对单模态检测的增益只有约 1 AP。
证据：RSFusionDet 的 RGB / 声呐 AP 为 76.4 / 48.6，比单模态 DINO 基线只高 0.7 / 1.4。对比之下，CMOLD 在 UXO（Unexploded Ordnance，水下未爆弹药）数据上用相机教师做纯声呐蒸馏，AP@[0.50:0.95] 从 0.661 提到 0.797（+13.6）。两者数据集不同，不能直接比较，但这提示了一个待检验的假设：对声呐支路来说，训练期蒸馏可能比推理期特征融合更有效。提升空间应该在退化条件下，而不是清水条件下。

**Q4（待检验：互毒害）**：相机退化时，交叉注意力融合是否会拖累声呐支路？
RSFusionDet 的 CAFusion 让声呐特征去查询 RGB 特征（Sonar2RGB / RGB2Sonar 双向可变形交叉注意力）。相机退化后，被查询的 RGB 特征变成噪声，声呐支路的 AP 可能下降。这一点没有人测过。[[2026-cvprw-cross-modality-transfer-sonar-detection|CVPRW 2026]] 证明了跨模态直接迁移存在负迁移，但那是数据和权重层面的迁移，不是推理期融合。这是 F2 的第 0 步问题核验（S0）：结果为否，F2 的主动机就不成立。

**Q5（蒸馏收益的来源）**：学生也能看到相机时，蒸馏收益来自两部分，需要分开归因。
- **容量收益**来自大教师带小学生。输入相同时这部分收益也成立，这就是 Hinton 提出的暗知识。纯声呐时这部分很小：CMOLD 的同模态蒸馏只到 0.671 / 0.688，CMHD 的同模态 KD-YOLOX-ViT（KD 即知识蒸馏）为 71.6，低于不蒸馏的 77.0。学生有相机输入时，这部分收益多半主要被相机支路吸收。
- **特权收益**来自教师和学生看到的输入不同。教师看清晰相机，学生看退化相机。学生没法从自己的相机输入复现教师的判断，只能从声呐里找线索，这就是声呐支路被增强的机制。
- 如果教师和学生都看清晰相机，蒸馏做的就是模型压缩，不是模态互增强。

**Q6（下游代价，未被测过）**：互增强会让两条支路的误差相关，而下游把逐模态输出当作独立证据。
- C 把检出概率写成 P(可见 | B 的地图) × pD_m(状态, 配置)，按模态分别建模。多假设关联时，C 把声呐和光学的检出当作条件独立。
- A v2.4 的 H-A1 已在定位侧证明："同一传感器一旦作为因子进入估计器，它对该估计器的检验能力就消失。"
- 感知侧有对应的情况。推理期交叉注意力让声呐支路的输出依赖相机特征。训练期蒸馏让声呐支路继承相机教师的盲区（[[2026-cross-modality-object-level-kd-sonar-detection|CMOLD 笔记]] §2.3）。两者都会让"两模态同时漏检"的概率高于独立假设下的乘积。
- 库内没有任何声光检测工作报告过支路之间的误差相关性。

**链条总结**：Q1 说明双输入是实际条件。Q2 指出现有方法只覆盖两端。Q3 把提升空间定位到退化区。Q4 是需要先确认的问题前提。Q5 给出蒸馏的机制，并说明为什么要做 2×2 归因。Q6 是这种机制的代价，也是 F2 区别于"又一个融合检测器"的地方。

## 2 问题的形式化定义

### 2.1 给定

- 配对声光数据 D = {(x_c, x_s, y_c, y_s, π)}：x_c 是相机图，x_s 是声呐图，y_c / y_s 是逐模态框，π 是跨模态对应关系。RSFusion 提供 π（训练 / 验证匹配对 4,084 / 1,032）。
- 相机退化算子 𝒯_s，s ∈ {D0, D1, D2, D3, D4, D∞}。D0–D4 沿用基础模型那篇的四效应叠加（光照衰减、波长吸收 β_R < β_G < β_B、散射 I = J·t_s + A_s(1 − t_s)、高斯模糊）；D∞ 表示相机输入置零或丢弃。声呐图不做退化。
- 名义映射 M₀：相机水平角 → 声呐方位，由 F1 §7.6 在 RSFusion 训练集上拟合（W2 判可行性，W9 定稿）。F2 直接复用，不重新拟合。
- 固定的检测框架家族（第 5.1 节），以及学生的部署算力上限。机载算力尚未核实，见评审稿 §2。

### 2.2 方法空间

| 编号 | 名称 | 教师 | 学生输入 | 说明 |
| --- | --- | --- | --- | --- |
| B0 | 纯声呐 | — | x_s | 声呐单模态基线 |
| B1 | CMOLD 式纯声呐蒸馏 | 相机检测器（清晰） | x_s | D∞ 端点的专家模型 |
| B2 | RSFusionDet 官方 | — | x_c, x_s | 只在名义条件下训练 |
| B3 | 退化增强融合 | — | 𝒯_s(x_c), x_s | 在 D0–D4 上训练，把基础模型那篇的结论用到检测上；**F2 的主要强基线** |
| B4 | B3 + 模态丢弃 | — | 𝒯_s(x_c)（含 D∞）, x_s | ModDrop 式随机丢弃相机 |
| B5 | 结果级晚融合 | — | 两个单模态检测器加匹配 | 支路之间结构上独立，作为独立性上界参照 |
| B6 | 输入端退化条件化（可选） | — | 拼接 5 维退化向量 | [[2026-aomfdm-uuv-acoustic-optical-fusion-detection-nue\|AOMFDM]] 式；按负面预期作对照 |
| M1 | 只有容量收益 | 大融合教师，输入与学生相同的退化相机 | 𝒯_s(x_c), x_s | 2×2 中的一格 |
| M2 | 只有特权收益 | 同容量融合教师，输入清晰相机 | 𝒯_s(x_c), x_s | 2×2 中的一格 |
| M3 | 容量加特权 | 大融合教师，输入清晰相机 | 𝒯_s(x_c), x_s | 用户提出的主设计 |
| M4 | M3 + 跨平面蒸馏 | 同 M3 | 同 M3 | 第 5.3 节，新颖性落点 |
| M5 | M4 + 模态丢弃 | 同 M3 | 含 D∞ | 一个模型覆盖 D0 到 D∞ |

### 2.3 评价

- **逐支路、逐退化档的 AP**：AP_m(s)，m ∈ {相机, 声呐}。所有方法用同一检测框架、同一输入尺寸、同一训练轮数。
- **互增强量**：Δ_m(s) = AP_m(融合学生, s) − AP_m(同容量单模态, s)。Δ_s(s) < 0 就是互毒害。
- **退化保留率**：R_m(s) = AP_m(s) / AP_m(D0)，以及 MRR（D1–D4 的平均），沿用基础模型那篇的定义，方便对照。
- **D∞ 性能**：模态丢弃后的声呐支路 AP，必须和 B1（CMOLD 式）对比，验证"一个模型覆盖全谱"不损失端点性能。
- **支路误差相关性**（第 6.4 节定义），按退化档报告。
- **部署代价**：参数量、FLOPs（浮点运算量）和目标硬件延迟。学生的上限在立项评审时冻结。
- **统计**：每个配置 3 个随机种子，报告均值 ± 标准差；统计单位是独立训练运行。
- **最差档单列**：D3、D4、dark 子集和 D∞ 的结果单独列出，不只报告平均值。

### 2.4 研究问题

- **RQ-0（问题存在）**：RSFusionDet 在相机 D0–D4 下，声呐支路 AP 是否显著下降（Δ_s(D4) 低于种子间差异的两倍）？
- **RQ-1（蒸馏还是融合）**：在 RSFusion 上，对声呐支路而言，B1（纯声呐蒸馏）、B2 / B3（推理期融合）、M3（融合加蒸馏）哪个更有效？
- **RQ-2（归因）**：M1 / M2 / M3 与同容量不蒸馏对照构成 2×2，容量收益和特权收益各占多少？是否随退化档变化？预期特权收益在 D3–D4 占主导，在 D0 接近零。
- **RQ-3（几何）**：沿方位窗做跨平面蒸馏（M4），是否优于 CMHD 式的逐网格位置蒸馏？
- **RQ-4（独立性代价）**：在支路 AP 增益相同的条件下，各方法的支路误差相关性差多少？增益和相关性之间是否存在权衡关系？
- **RQ-5（从合成到真实）**：用合成 D0–D4 训练的模型，在 H1 的真实高岭土浊度阶梯上，退化保留率和排序是否保持？

### 2.5 结论的适用边界

- **RSFusion**：消声水池、清水、静态刚体、Oculus MD750d（1.2 MHz，130° 声呐视场对 30.75°×23.31° 相机视场）。退化是合成的，只能宣称"在合成相机退化下"。
- **真实浊度结论只来自 H1 数据**：同一平台、导轨单目标、0–200 NTU 高岭土，不外推到天然浊水和多目标场景。
- **声呐保持不变的前提**：H1 的 Q2 会检验 1.2 MHz 在高浊度下是否明显变差。若变差，F2 在高浊度档的"声呐不变"前提失效，需改为按频率分别报告。
- **UMOD**：没有像素配准，也没有跨模态对应标注，不能用来算支路误差相关性，只可选作相机支路退化保留率的外部测试。

## 3 子问题分解与回答顺序

顺序沿用 F 详案和 F1 的策略：先确认问题存在，再试最便宜的方案，每步只加一个变量。

| 步骤 | 回答 | 内容 | 若结果为否 |
| --- | --- | --- | --- |
| S0（约 2 周） | RQ-0 | 跑通 RSFusionDet 官方代码；在验证集相机侧施加 D0–D4，只做推理不重新训练，测 AP_c(s)、AP_s(s)、匹配 F1(s)；再训练 B3 看能恢复多少 | Δ_s(D4) 在噪声范围内 → 互毒害不存在，F2 收缩为 RQ-1 的短篇实证，或并入 F1 的 S5 |
| S1（约 3 周） | RQ-1 | B0 / B1 / B2 / B3 在同一框架下 3 个种子 | 若 B3 的声呐支路已经不低于 B1，说明融合本身已够用，蒸馏线重心转到 RQ-4 |
| S2（约 4 周） | RQ-2 | 2×2 加 M3，再加模态丢弃（B4 / M5 的前半） | 特权单元在 D3–D4 不优于只有容量的单元 → 主张收缩为"融合检测的模型压缩"，RQ-3 降为可选 |
| S3（约 3 周） | RQ-3 | 跨平面方位窗蒸馏（M4）对比逐位置蒸馏 | M4 不优于逐位置 → 撤回新颖性主张，作为负结果报告 |
| S4（与 S1–S3 并行，只做评测） | RQ-4 | 在所有已训练模型上算支路误差相关性和下游代理指标 | 相关性在各方法间无差别 → 如实报告"该数据上蒸馏不引入额外相关"，对 C 是正面结论 |
| S5（H1 数据可用后，约 3 周） | RQ-5 | 在 H1 清水帧上施加合成退化训练，在真实 NTU 阶梯上测试；导轨配对帧上做真实特权蒸馏 | 合成到真实不迁移 → 报告差距，全部主张限定为合成退化 |

S0 是整个选题的闸门。它只需要官方代码加推理期退化，代价约一周 GPU 时间，应在任何方法实现之前完成。

## 4 核心张力

张力 1：互增强与独立性。让声呐支路"借"相机信息，可以通过推理期交叉注意力，也可以通过训练期蒸馏。这样提升的是单支路 AP，代价是两条支路的错误变得相关。下游把两路当独立证据时，相关性会转化为过度自信。下游包括 C 的多假设关联，以及将来可能的 A 冲突检查。F2 不预设哪一端更好，而是把这条权衡曲线测出来（RQ-4）。对应设计：保留一个推理时不和相机交叉注意力的声呐输出头（第 5.1 节），把它和融合后的输出分开记账。

张力 2：特权差距与可达性。大教师 → 轻量学生本来就有容量差距；清晰 → 退化又加了一层信息差距。到了 D4，再碰上目标在声呐里本身回波弱的情况，教师给的目标学生根本达不到，硬逼的结果是学生学会虚报。KD 文献已知师生差距过大会伤害蒸馏（[[2020-mirzadeh-teacher-assistant-knowledge-distillation|Teacher Assistant KD]]，Mirzadeh 等，AAAI 2020；该文只处理容量差距，见第 11.1 节）。对应设计：第 5.4 节的可达性调权、退化课程和助教模型。

张力 3：合成退化可控与真实退化可信。合成 D0–D4 可以精确生成"同一帧的清晰版和退化版"，这是特权蒸馏需要的配对；真实退化没有逐像素的清晰配对。RSFusion 的 dark 档是真实退化，但没有同场景的清晰版本。H1 的导轨数据提供"近配对"：ROV 锁在导轨小车上，同一刻度位置在清水和不同 NTU 下各采一次，几何近似一致但不是逐像素一致。所以 F2 在两个层面分开报告：合成层（完全配对）和 H1 层（近配对加真实退化）。

张力 4：跨平面对应的粒度。CMHD 的逐网格位置蒸馏默认两模态的同一网格位置指向同一场景位置，[[2026-cmhd-lightweight-sonar-rgb-heterogeneous-distillation|CMHD 笔记]] §2.2 指出这一点从未验证。RSFusionDet 的笔记说明两路成像根本不在同一投影平面（前视图对距离—方位图）。单目相机不给距离，所以几何上一致的对应只能是"相机列 → 声呐方位窗、距离未知"。F2 的跨平面蒸馏只在这个粒度上做（第 5.3 节）。

张力 5：gate 输出的语义。基础模型那篇明确声明 gate 权重"不是校准的可靠性概率"。G3 线的教训是"提出一个 gate"不足以支撑论文。F2 把 gate 当作融合结构的一部分，不当作主张，也不把它的输出当可靠性概率用于调权或下游。需要可靠性信号的地方，用可以对照误差检验的量（例如教师在声呐头上的 objectness），并先检验信号与误差的关系，再用于干预（评审稿 §3 末段原则）。

## 5 方法思路（后置：S0 为正之后才进入）

### 5.1 架构约束

- 教师和学生用同一个检测家族，这样逐预测的蒸馏才对得上。可选的有两种：
  - YOLOX 家族（一类单阶段检测器）：用融合版 YOLOX-L 做教师，用融合版 YOLOX-S 或 Ghost 化的网络做学生。它沿用 CMOLD / CMHD 的解耦头和分支选择性蒸馏，实现最省事。
  - DETR 家族（基于 transformer 的检测器）：沿用 RSFusionDet 的 DINO 双分支。查询级蒸馏需要先做匈牙利匹配，实现更重，好处是能直接复用官方的匹配头。
  - 在 S1 之前从两者中二选一，选定后不再更换。
- **双检测头**：每个模态各一个检测头，沿用 RSFusionDet 的"逐模态检测 + 跨模态匹配"输出表示；声呐和相机的框在不同坐标系、不同尺度分布（RSFusion 里 RGB 以大目标为主，声呐以中小目标为主），共用一个头会互相拖累。
- **一个不和相机耦合的声呐输出头**：由融合前的声呐特征单独接一个轻量检测头，推理时同时输出。它服务于张力 1 的分开记账和 D∞ 回退，训练时同样接受蒸馏，评测时单独报告。
- **融合模块不作为贡献**：用 RSFusionDet 的 CAFusion 或简单拼接，不另行设计 gate。

### 5.2 训练分布

- **退化采样**：每个训练样本的相机输入按 s ~ p(s) 从 {D0, …, D4, D∞} 中采样。初值为 D0–D4 各占 0.16、D∞ 占 0.2，在验证集上冻结。退化参数沿用基础模型那篇的四效应，加一档"只有色偏"作为负对照（该文显示纯色衰减下融合增益为 +0.0%，模型不应在这一档转移权重）。
- **教师输入固定为清晰视角**：所以教师输出可以按图像离线缓存一次，学生每步只需前向自己。这要求几何增强对教师和学生同步施加：翻转和缩放可以对缓存的输出图同步变换；Mosaic 这类拼接增强对蒸馏样本关闭。
- **声呐输入不做退化**：声呐侧的配置变化是 F1 的研究轴，F2 不混入。

### 5.3 蒸馏损失

总损失：

L = L_det^c + L_det^s + L_det^{s0} + λ_s·L_KD^s + λ_c·L_KD^c + λ_x·L_xp

各项含义：
- **L_det^c、L_det^s、L_det^{s0}**：相机头、融合声呐头、未耦合声呐头各自的检测损失，监督来自各模态自己的真值框。
- **L_KD^s、L_KD^c（同模态支路蒸馏）**：教师声呐头 → 学生声呐头，教师相机头 → 学生相机头。只蒸 objectness（MSE 或 BCE）和类别分布（带温度 τ 的 KL，乘 τ²），不蒸框回归，沿用 CMOLD / CMHD 的分支选择。CMOLD 的消融显示单开 objectness 就到 0.794（完整版 0.797），所以 objectness 是首选，类别项是第二步消融。
- **L_xp（跨平面蒸馏）**：教师相机头在清晰视角下的检测框 b_c，经 M₀ 映射为声呐方位窗 W(b_c) = [θ₁, θ₂]。单目不给距离，所以约束取多示例学习形式：

  L_xp = Σ_{b_c} BCE( max_{r, θ∈W(b_c)} p_s^{obj}(r, θ),  p_c^{T,obj}(b_c) )

  也就是说，教师确信相机里有目标时，学生声呐头在对应方位条带内沿距离方向的最大 objectness 应当与之相符。
  - **只传正证据，不传负证据**：相机垂直视场 23.31° 与声呐垂直孔径不同，相机看不到的远距目标也可能在声呐里。相机没看到，不代表声呐条带里没有东西。这和 [[2026-prism-prior-reliability-visual-scheduling-underwater-slam|PRISM]] 的单边信息上限是同一类原则：证据只能单向使用。
  - **适用范围**：只在方位窗落入声呐视场时生效。130° 声呐对 30.75° 相机，条带只覆盖声呐方位的约四分之一，条带之外 L_xp 为零。
  - **对 M₀ 误差的容忍**：方位窗按 M₀ 拟合残差外扩。F1 L1 的标定容差曲线可以直接用来设定外扩量。

### 5.4 差距控制

- **可达性调权**：对每个目标，用教师在声呐头上的 objectness 作为"声呐里可见"的代理，权重 w = σ(k·(p_s^{T,obj} − θ_w))。声呐里回波弱的目标降低特权蒸馏权重，避免逼学生虚报。这个代理量先在验证集上检验与学生实际可达性（单声呐训练后的召回）的相关性，再启用（评审稿 §3 原则：先验证信号与误差，再设计干预）。
- **退化课程**：训练前若干轮 p(s) 偏向 D0–D2，逐步移向均匀分布。
- **助教模型**：若 S2 中 M3 不如 M1 或 M2 的较优者，插入一个中等容量、输入清晰相机的助教模型，分两级蒸馏。

### 5.5 明确不做的

- 推理期对已训练特征做缩放或调制（G3 教训，见 [[comparisons/sonarsweep-wrapper-variant-failure-analysis|SonarSweep 失败分析]]）。
- 把 gate 权重当可靠性概率交给下游。
- 输入端退化条件化作为方法主张。AOMFDM 跨域后 5 维向量系统性偏移，会把错误的退化状态注入主干；F2 只把它作为可选对照 B6。
- 视觉基础模型的声呐侧微调。基础模型那篇的声呐侧本来就是 ImageNet ResNet-18，没有证据支持 VFM 对声呐图有优势。

## 6 实验设计

### 6.1 数据与划分

| 数据 | 用途 | 关键属性 | 限制 |
| --- | --- | --- | --- |
| [[2026-rsfusiondet-rgb-sonar-detection-heu\|RSFusion]] | 主数据，S0–S4 | 7,073 对（训练 5,658 / 验证 1,415）；7 类；跨模态对应标注；light / dark 与 sr5–20 分层；声呐元数据 | 清水消声水池；没有官方测试集，验证集需拆分为调参验证和最终测试，拆分方式与 F1 共用并按实例与日期留出 |
| H1 受控浊度数据（[[plans/2026-10/intern-project-2026-h1-rov-platform-testbed-2026-10\|H1]]） | S5 真实退化 | 0–200 NTU 高岭土；相机侧 6 档浊度 × 3 档灯；导轨刻度位置；按帧配置元数据；备赛 5 类目标替代物与标准靶标 | 导轨单目标，检测本身容易，主要评 objectness 召回和类别；声呐型号与 RSFusion 不同（团队为双频 750 kHz / 1.2 MHz，型号待核实）；数据量与时间由 H1 决定 |
| [[2026-vsafdet-umod-visual-sonar-detection\|UMOD]]（可选） | 相机支路退化保留率的外部测试 | 4,000 对、9 类；grouped split（40 个连续 ID 为一块） | 时间同步但不配准、没有对应标注，不算支路相关性 |

- RSFusion 上的相机退化在加载时按样本种子确定性生成，测试集每档固定一份，所有方法共用。
- 退化参数只在训练集和验证集上调，测试集只评测一次。

### 6.2 基线与消融

第 2.2 节的 B0–B6 与 M1–M5 全部使用同一检测家族、同一输入尺寸、同一训练轮数、同一调参预算。自建强基线（B3、B4）的实现由非作者成员评审，参数在验证集冻结，遵守评审稿 §3 第 7 条。主要消融项：
- 去掉未耦合声呐头；
- 去掉 L_KD^c、L_KD^s、L_xp 中的单项；
- 可达性调权的开与关；
- 模态丢弃比例 p(D∞) ∈ {0, 0.1, 0.2, 0.4}；
- 只有色偏的负对照档。

### 6.3 2×2 归因设计

| | 教师输入：同样退化的相机 | 教师输入：清晰相机 |
| --- | --- | --- |
| 教师与学生同容量 | 自蒸馏对照（D-self） | M2：只有特权收益 |
| 大教师 → 轻量学生 | M1：只有容量收益 | M3：容量加特权 |

- **容量效应**：(M1 − D-self) 与 (M3 − M2) 的均值。
- **特权效应**：(M2 − D-self) 与 (M3 − M1) 的均值。
- **交互项**：M3 − M1 − M2 + D-self。

三者都按退化档 s 和支路 m 分别报告。预期：特权效应随 s 增大而增大、在 D0 接近零；容量效应主要出现在相机支路。

### 6.4 支路误差相关性的定义

在 RSFusion 的跨模态匹配对（同一物体在两模态都有真值框）上，对每个物体 o 和退化档 s，记 m_c(o,s)、m_s(o,s) 为该方法在相机支路、声呐支路是否漏检（IoU 0.5，置信阈值在验证集固定为每图 1 个虚警的工作点）。报告四个量：

1. **条件漏检升幅**：ΔP_miss(s) = P(m_s = 1 | m_c = 1) − P(m_s = 1)。独立时为 0；正值表示相机漏的时候声呐也更容易漏。
2. **φ 系数**：两个二值漏检变量的相关系数，附自助法 95% 置信区间。
3. **置信度相关**：两支路在匹配对上置信度分数的 Spearman 相关，按 s 分档。
4. **下游代理**：用"条件独立"假设把两支路的检出合成物体存在概率（朴素贝叶斯：P(存在 | 检出) 按两路似然相乘），在测试集上计算存在概率的 ECE。相关性越强，独立假设下的过度自信越明显，ECE 越高。这个代理量直接对应 C 的使用方式（第 8.3 节）。

B5（晚融合）在结构上没有推理期耦合，作为相关性的参照下限。但两个单模态检测器在同一数据上训练，仍可能因为同一批困难样本而相关，所以 B5 的 ΔP_miss 不一定为 0，需要实测。

### 6.5 算力估计（立项评审时复核）

以下按单卡 4090、RSFusion 训练集 5,658 对、输入 640 估算，未经实测：

| 项目 | 次数 | 单次 | 小计 |
| --- | --- | --- | --- |
| S0 推理期退化评测（官方 RSFusionDet） | 5 档 × 1 | < 0.5 h | < 3 h |
| S0 B3 重训 | 3 种子 | 约 4–6 h（DINO 双分支） | 12–18 h |
| 教师训练（大、同容量 × 清晰 / 退化输入） | 4 | 约 6–10 h | 24–40 h |
| S1 B0 / B1 / B2 / B3 | 4 × 3 种子 | 约 2–4 h | 24–48 h |
| S2 2×2 + 模态丢弃 | 6 × 3 种子 | 约 2–4 h | 36–72 h |
| S3 M4 及消融 | 4 × 3 种子 | 约 2–4 h | 24–48 h |
| S5 H1 合成训练 + 真实测试 | 约 6 | 约 1–2 h | 6–12 h |
| 合计 | | | 约 150–240 GPU 小时 |

教师输出离线缓存（第 5.2 节），学生训练不需要同时驻留教师，显存按单个双分支学生估计。

## 7 启动条件、关键节点与停止条件

### 7.1 启动条件（同时满足才立项；任一不满足时只允许做 G0）

- **G0 问题核验为正**：S0 中 RSFusionDet 的 Δ_s(D4) 下降幅度超过种子间标准差的 2 倍，且 B3 不能把声呐支路恢复到 B0 水平以上。G0 本身只需约 2 周、20 GPU 小时以内，可以先做。
- **G1 真实退化数据可得**：H1 的受控浊度阶梯数据按其协议采完，至少相机侧 6 档 × 1 档灯、带导轨刻度位置；或者有其他同等的真实浊度配对数据。不满足时 F2 只能停在合成退化，竞争力明显下降，需在立项评审中明确接受这一点。
- **G2 下游需求**：C 的 M3 错误归因显示检测误差占清单错误的主要部分（C 详案 §3 步骤 3：关联研究空间不足时"算法投入转 F 或 A/B"；评审稿 §6："若改进主要来自检测器，则将算法投入转向 F"），或者产品验收提出了相机退化下的检测需求。

### 7.2 关键节点

| 节点 | 内容 | 判定 |
| --- | --- | --- |
| F2-M0（G0 结束） | S0 结果 | 为否 → 收缩为 RQ-1 短篇实证，或并入 F1 S5 |
| F2-M1（S1 结束） | 蒸馏 vs 融合的声呐支路对比 | 决定主线偏向"蒸馏增强"还是"独立性权衡" |
| F2-M2（S2 结束） | 2×2 归因 | 特权效应不显著 → 主张收缩为模型压缩 |
| F2-M3（S3、S4 结束） | 跨平面蒸馏与相关性曲线 | 冻结图表，进入写作 |
| F2-M4（S5 结束） | 合成到真实 | 决定论文的结论层级（只有合成，或合成加真实） |

### 7.3 阈值说明

"种子间标准差的 2 倍"和"恢复到 B0 以上"都是起始值，没有团队历史数据支撑，在 F2 立项时与 F1 的门槛一并审定。

## 8 对其他课题的借鉴

### 8.1 对 A（逐轴完好性监测）：一套协议，一条原则；不引入方法

- **方法层不引入**：A v2.4 的检测器由开环参考通道组成（冻结零偏陀螺积分、只用声呐几何拟合的竖直参考面、深度计），不消费学习型检测输出。F2 的网络、蒸馏和融合在 A 中没有使用位置。
- **借用协议（视觉退化档位扫描）**：A 的五类片段（清晰、视觉退化、声学困难、双侧困难、恢复）中，"视觉退化"目前是二值分类。在回放时对双目图像按 D0–D4 施加合成退化，可以得到"退化档位 × 失效类型"表：哪一档开始估计器跟踪丢失（内部量可检出），哪一档出现"一致但错误"（H-A1 关心的类型，内部量检不出）。这能帮 A 判断合成视觉退化能不能作为 H-A1 的测试来源。预期视觉退化更多导致跟踪丢失而不是一致但错误；若属实，视觉退化片段对 H-A1 的检验价值有限，这本身就是有用的结论。限制：合成退化对特征匹配的影响和真实浊水不同，只作压力测试；真实视觉退化片段仍以 H1 的浊度阶梯为准。
- **借用原则（耦合即失去检验能力）**：H-A1 的推论"同一传感器一旦作为因子进入估计器，它对该估计器的检验能力就消失"在感知侧有直接对应：声呐支路经交叉注意力或蒸馏吸收了相机信息后，它的检出就不再能独立检验相机的检出。F2 第 6.4 节的相关性指标正是这种损失的度量。如果 A 将来加入检测级的跨模态冲突检查（例如对象级的声光位置冲突），应使用未耦合的声呐头或晚融合检测器，并先用 F2 的指标测出相关性。
- 本条只是建议，A 详案不随本草案修改，由 A 的负责人决定是否采纳。

### 8.2 对 B（有界不确定占据建图）：无实质借鉴（更正上一轮判断）

- 讨论中曾建议"目标级视声匹配可帮 B 收缩仰角候选"。这一判断依据的是评审稿 v2.7 的旧表述，已作废。B 详案 v2.3 §3.3 的仿真结果是：双目关联在梁上下为开阔水体的结构上不能收缩候选（存活 0.89–1.00，余量收紧到能多排除候选时就开始误删真实仰角），这是几何限制，不是实现问题。H-B2 只剩"弧所在高度有背景面"的条件主张。F2 的目标级匹配改变不了这一几何结论。
- B 第一年明确不训练新的深度网络（评审稿 §5）。B 需要的视觉退化梯度由 H1 提供，不需要 F2。
- 唯一的间接关系：如果将来 C 用 F2 的检测器把某段弧归属到对象 o，这属于 C 的对象级关联，不进入 B 的占据表示。

### 8.3 对 C（持续目标地图）：主要使用方

- **检测器供给**：F2 交付带版本号的检测器，同时输出相机头、融合声呐头、未耦合声呐头三路结果和跨模态匹配。C 详案 §3 步骤 2 要求"检测器换代不与关联算法实验混在一起"，F2 的版本按 C 的冻结节点交付。
- **物理检出模型的输入**：C 步骤 4 的光学检出模型以距离、能见度等级（浊度协议的 `level_id`／`ntu`／`jerlov_type` 字段）和光照为输入。F2 在 H1 数据上按 NTU 档测出的逐支路召回曲线，可以作为 C 所用检测器的 pD_m(能见度) 标定数据。前提是 C 部署的检测器就是被测的那个版本。
- **独立性**：C 的多假设关联按模态分别计算检出似然。F2 第 6.4 节的 ΔP_miss 和下游 ECE 直接说明 C 能否假设条件独立。若相关性显著，C 有两种选择：改用晚融合或未耦合声呐头的输出，或在关联中显式建模相关性。
- **启动耦合**：F2 的 G2 条件由 C 的 M3 错误归因给出，两边的决策节点要对齐。

### 8.4 对 F1：两项低成本借用（需指导人决定，F1 v1.4 文本未改）

1. 给相机教师加一条可控的可靠性轴：F1 的主要边界是 RSFusion 为清水水池：相机的难层（dark）和声呐的难层（sr20）不重合，P6 的负相关假设在这份数据上几乎测不了（F1 §1 P6、§2.5、§8.6）。对相机图施加 D0–D4 后再生成伪标签，教师精度就变成可控自变量，对应 RQ-d 里"教师精度高于某个阈值时 A4 占优"那一轴。
   - 只改伪标签生成这一步，学生训练管线不变，不违反单变量原则。也可以同时比较 YOLO 教师和更强的教师，给这条轴多两个点。
   - 规模：只在 T2 预算档，3 档退化 × A3/A4 × 2 种子，约 12 次训练、12–24 GPU 小时，需要从 L1 的 66 次中匀出。
   - 主张必须写明"合成相机退化，不等于真实浊度"。
2. S5 按"D∞ 端点"的框架做，并加上 F2 的 S0 核验：F1 的 S5（纯声呐 vs 声光双输入）若有余量，把 F1 的 A4 纯声呐学生作为 D∞ 端点，与 RSFusionDet 在 D0–D4 下的声呐支路对照。代价是一次官方代码复现加推理期退化评测，结果直接充当 F2 的 G0。

### 8.5 对 F 父课题

F 详案 §3 步骤 5 "推理分别测试声呐单独输入及声光共同输入"是 F2 的入口。F 进入工作范围时，可以把 F2 的 S0 作为步骤 5 的最小实现。

## 9 风险

| 风险 | 概率 | 处理 |
| --- | --- | --- |
| S0 显示互毒害不存在 | 中 | 按 F2-M0 收缩，不硬做；RQ-1 与 RQ-4 仍可作短篇实证 |
| 缺失模态融合与特权蒸馏在 RGB-T、RGB-D、相机-激光雷达上已有大量工作，新颖性被质疑 | 高 | 库外检索已完成（2026-10-19，第 11.1 节）：设定本身（融合教师多一份更好输入 → 融合学生）已有 CRKD 等先例；跨平面方位窗蒸馏只剩部分新颖性，需 RQ-3 实验支撑；支路独立性的新颖点收窄为"同一检测器内模态支路的相关性 + 独立融合概率的校准，按退化档与蒸馏方式比较"。不押在"退化感知融合"或"特权蒸馏"本身 |
| 国内声光检测赛道密集（G4 方向卡 D3 的判断） | 高 | 不比单一数据集上的 AP；以 2×2 归因、相关性曲线和 H1 真实浊度为差异 |
| H1 数据延迟或规模不足 | 中 | G1 不满足时按合成退化层收窄主张，S5 推迟 |
| H1 Q2 发现 1.2 MHz 在高浊度下明显变差 | 中 | 高浊度档按频率分开报告，"声呐不变"前提改为"750 kHz 不变" |
| M₀ 在 F1 中拟合失败（F1 §8.6） | 中 | 跨平面蒸馏改用 F1 的图像域方位近似；方位窗按残差外扩；仍失败则 RQ-3 降为可选 |
| 师生差距导致 M3 劣于 M1 / M2 | 中 | 第 5.4 节的可达性调权、退化课程、助教模型 |
| DETR 家族蒸馏实现成本高 | 中 | S1 前选定 YOLOX 家族为默认，只有需要复用 RSFusionDet 匹配头时才选 DETR |
| 合成退化分布与真实差距大 | 高 | 本身就是 RQ-5 的测量对象；主张分合成层和 H1 层报告 |
| 机载算力未知，学生上限无法冻结 | 中 | 先以 CMHD 量级（约 3 M 参数、8 GFLOPs 级）为占位，平台算力核实后冻结 |

## 10 人员、时间与资源

- **执行人**：F1 结束后的算法实习生或研究生，要求跑过检测训练与蒸馏。F1 的管线、M₀、划分脚本可以直接复用，所以 F1 实习生续做的启动成本最低。
- **周期**：G0 约 2 周可单独先做；完整 S1–S5 约 16 周（S1 3 周、S2 4 周、S3 3 周、S5 3 周、写作 3 周，S4 并行），S5 的起点由 H1 数据决定。
- **最早时间线**：F1 于 2027-02 中旬结束；H1 自 ROV 到货（2026-10 W43）起约 5 个月，浊度阶梯数据最早约 2027-03。所以 F2 的 S1–S3 可以在 2027-02 至 2027-05 进行，S5 衔接其后。
- **算力**：约 150–240 GPU 小时（第 6.5 节），单卡 16 周可容纳。
- **评审**：立项评审时确认学生的部署上限、检测家族、阈值初值和 H1 数据使用约定。

## 11 文献

**库内**
- 蒸馏路线：[[2026-cross-modality-object-level-kd-sonar-detection|YOLOX-CMOLD]] · [[2026-cmhd-lightweight-sonar-rgb-heterogeneous-distillation|CMHD]] · [[2026-cvprw-cross-modality-transfer-sonar-detection|CVPRW 2026 跨模态迁移]]
- 融合检测：[[2026-rsfusiondet-rgb-sonar-detection-heu|RSFusionDet]] · [[2025-uamfdet-acoustic-optical-detection-hrbeu|UAMFDet]] · [[2026-aomnet-acoustic-optical-joint-detection|AOMNet]] · [[2026-scfusion-sonar-camera-detection|SCFusion]] · [[2026-vsafdet-umod-visual-sonar-detection|VSAFDet/UMOD]] · [[2025-ao-uod-acousto-optic-underwater-detection|AO-UOD]] · [[2026-aopq-net-acoustic-optical-proposal-query-detection|AOPQ-Net]]
- 退化感知：[[2026-degradation-aware-foundation-fusion-umod|Degradation-aware Foundation Fusion]] · [[2026-aomfdm-uuv-acoustic-optical-fusion-detection-nue|AOMFDM]] · [[2026-sonarllm-sonar-optical-mllm|SonarLLM]]（AGFM 退化驱动门控） · [[2026-prism-prior-reliability-visual-scheduling-underwater-slam|PRISM]]（单边信息上限） · [[2025-russo-robust-underwater-slam-sonar|RUSSO]]
- 下游与检出模型：[[2026-active-mapping-litter-camera-sonar-aqtr|主动建图（aqtr）检出表]] · [[2025-multimodal-fusion-small-scale-underwater-target-tracking|小目标声光跟踪]]
- 成像：[[concepts/underwater-image-formation-model|水下成像模型]] · [[2018-akkaynak-revised-underwater-ifm-cvpr|Akkaynak 2018]] · [[2025-sonar-mast3r-real-time-optic-acoustic|Sonar-MASt3R]]（真实 NTU 分级）

**库外（2026-10-19 核实并建页，放在 `entities/papers/misc-slam-perception/`）**
- 特权信息学习：[[2009-vapnik-vashist-lupi-privileged-information|LUPI]]（Vapnik & Vashist，Neural Networks 22(5–6)，2009；全文未读，笔记 confidence low） · [[2016-lopez-paz-generalized-distillation-privileged|Generalized Distillation]]（Lopez-Paz 等，ICLR 2016）
- 模态丢弃：[[2016-neverova-moddrop-multimodal-dropout|ModDrop]]（Neverova 等，TPAMI 38(8)，2016）
- 师生差距：[[2020-mirzadeh-teacher-assistant-knowledge-distillation|Teacher Assistant KD]]（Mirzadeh 等，AAAI 2020）
- 检索得到的检测工作：[[2016-hoffman-modality-hallucination-rgbd-detection|Modality Hallucination]]（RGB-D，CVPR 2016；只读到摘要） · [[2017-xu-cross-modal-rgbt-pedestrian-detection|CMT-CNN]]（RGB-T，CVPR 2017） · [[2021-wang-rodnet-radar-cross-modal-supervision|RODNet]]（相机 → 雷达距离—方位监督，WACV 2021） · [[2022-hwang-cramnet-camera-radar-ray-cross-attention|CramNet]]（相机-雷达射线约束注意力与传感器丢弃，ECCV 2022） · [[2022-chen-proben-rgbt-probabilistic-ensembling|ProbEn]]（RGB-T 条件独立晚融合，ECCV 2022） · [[2024-zhao-crkd-camera-radar-distillation|CRKD]]（LiDAR-相机教师 → 相机-雷达学生，CVPR 2024）

### 11.1 库外检索与新颖性判断

**检索范围与方法**：检索日期为 2026-10-19。WebSearch 在本环境不可用，用内置浏览器查 arXiv API、OpenAlex 和 arXiv HTML 全文。关键词组合为 missing modality / modality dropout / privileged information / knowledge distillation × object detection × RGB-T / RGB-D / camera-LiDAR / camera-radar，另查 conditional independence、error correlation × multimodal fusion。四篇指定文献的 venue、年份、作者全部核实；LUPI 全文在付费墙后，出版商页面要求人机验证，只核实了书目信息。检索只覆盖 arXiv 与 OpenAlex 的索引，未查 IEEE Xplore、CNKI 与 2026 年下半年的会议录，不能排除未上 arXiv 的近期工作。

**已核实存在、未建页的工作**（供立项评审追查）：Kruthiventi 等 ICIP 2017（RGB+热红外教师 → RGB 学生，低照度行人，DOI:10.1109/ICIP.2017.8297075）；HalluciDet（WACV 2024，红外 → RGB 幻觉）；MonoTAKD（CVPR 2025，arXiv:2404.04910，在 LiDAR 教师与单目学生之间加相机助教）；BEVDistill（arXiv:2211.09386，LiDAR → 多视角相机 BEV 蒸馏，venue 未核实）；Scarf-DETR（arXiv:2511.06406，红外-可见光缺失模态检测，伪模态丢弃）；MMANet（CVPR 2023，缺失模态的边距感知蒸馏，分类与分割）；RoboDistill（arXiv:2609.23541，2026-09，LiDAR-相机 3D 检测在 27 种 OOD 退化下的 VFM 蒸馏）；Kuncheva & Whitaker 2003（Machine Learning，DOI:10.1023/A:1022859003006，集成多样性度量）。

**逐篇对照 F2 的两项新颖性主张**

| 工作 | 设定（教师 → 学生；测试输入） | 主张 A：方位窗跨平面 MIL 蒸馏（§5.3） | 主张 B：支路误差相关性测量（§6.4） |
| --- | --- | --- | --- |
| [[2016-hoffman-modality-hallucination-rgbd-detection\|Modality Hallucination]] | RGB+深度 → RGB（幻觉深度特征） | 无几何投影 | 无 |
| [[2017-xu-cross-modal-rgbt-pedestrian-detection\|CMT-CNN]] | RGB-T 重建 → RGB | 无几何投影；夜间增益不大于白天 | 无 |
| [[2021-wang-rodnet-radar-cross-modal-supervision\|RODNet]] | 相机定位 + 雷达峰值 → 雷达；测试只用雷达 | **部分重叠**：相机目标投到距离—方位平面作监督；但先用单目深度定成一个点（距离方差随深度增长），出硬 ConfMap，学生不看相机 | 无 |
| [[2022-hwang-cramnet-camera-radar-ray-cross-attention\|CramNet]] | 推理期相机-雷达融合 | **部分重叠**：相机像素沿射线采样、用雷达特征的注意力选深度，属于"方向已知、距离未知时沿一维选最佳匹配"的同类几何；用在推理期，不是蒸馏损失 | 无；雪天融合 8.89 低于单雷达 15.84（BEV AP），是互毒害的外部个例 |
| [[2024-zhao-crkd-camera-radar-distillation\|CRKD]] | LiDAR+相机融合 → 相机+雷达融合；测试两路 | 不重叠：师生都在 BEV 中，可逐格对齐（CSRD 让雷达特征对齐教师 objectness 热图） | 无；按天气、光照分组只报 mAP，夜间蒸馏增益 +0.20 |
| [[2022-chen-proben-rgbt-probabilistic-ensembling\|ProbEn]] | 两个单模态检测器晚融合 | 无 | **部分重叠**：把条件独立写成融合规则，并在排序指标上显示非独立检测器融合仍有效；没有测相关性，也没有测融合概率的校准 |
| [[2016-neverova-moddrop-multimodal-dropout\|ModDrop]] | 训练期整路丢弃 | 无 | 间接相关：其正则项是跨模态权重耦合，预测模态丢弃可能**增加**支路相关 |
| [[2016-lopez-paz-generalized-distillation-privileged\|Generalized Distillation]] | 理论框架 | 无 | 无；其实验 2（x* 是 x 的干净版本）蒸馏无显著收益，约束 F2 特权收益的来源 |
| [[2020-mirzadeh-teacher-assistant-knowledge-distillation\|TAKD]] | 同输入的容量差距 | 无 | 无；助教只缓解容量差距，不缓解 F2 的信息差距 |

**结论**

1. "特权蒸馏 + 缺失或退化模态鲁棒融合"作为设定不新：教师看完整或更好的模态、学生看缺失或较弱的模态，在 RGB-D（2016）、RGB-T（2017）、相机-雷达（2021、2024）检测中都已发表；融合教师 → 融合学生的结构有 CRKD。第 9 节把这条风险标为"高"是对的，F2 不能把设定本身写成贡献。F2 仍可主张的设定差异只有一条：特权差距是**同一传感器的清晰版与退化版**，可以合成逐像素配对并沿退化档连续扫描，从而做 2×2 容量×特权归因；检索中没有找到在检测任务上做这种拆分的工作。
2. 主张 A（方位窗跨平面 MIL 蒸馏）部分存活：几何思路已见于 CramNet（推理期射线注意力）和 RODNet（带距离方差的相机 → 距离—方位监督）。检索中未见的是这一组合：训练期蒸馏损失 + 沿距离取最大的多示例形式 + 只传正证据 + 前视声呐。这是组合层面的新颖，单凭描述不成立，必须由 RQ-3 证明它优于两个更直接的替代：CMHD 式逐网格蒸馏（已在计划中）和 RODNet 式"单目深度定点"投影（建议加入 RQ-3）。M4 不优于后者时应撤回主张 A。
3. 主张 B（支路误差相关性）存活，但要收窄：φ 系数、条件漏检升幅是集成多样性文献的标准度量（Kuncheva & Whitaker 2003），不是新度量；条件独立融合规则来自 ProbEn。检索中未见的是：在**同一个融合检测器内部**测模态支路的误差相关，比较推理期交叉注意力、训练期蒸馏、模态丢弃三种"互增强"方式对相关性的影响，再用独立融合概率的校准误差（ECE）把相关性换算成下游代价。ProbEn 只报排序指标（AP、LAMR），对单调变换不敏感，测不出过度自信，这正是 §6.4 第 4 项要测的量。主张 B 是两项中更稳的一项。
4. 综合判断：两项主张都没有被已发表工作完全覆盖，可以进入立项评审；但贡献排序应改为 B（独立性代价）> 2×2 归因 > A（跨平面蒸馏），而不是 §2.2 M4 行现在写的"A 为新颖性落点"。

**建议的正文修改（本次未改，留待立项评审决定）**

- §1 Q4："这一点没有人测过"改为"车载相机-雷达融合中已有融合低于单传感器的个例（CramNet 雪天 8.89 vs 15.84 BEV AP），声光检测中未测"。
- §1 Q5：补一句"特权收益要求声呐支路携带关于目标的额外信息；只有相机一路时，按 Generalized Distillation 实验 2，清晰版作特权几乎没有收益"。
- §2.2 M4 行"新颖性落点"改为"候选新颖性，需 RQ-3 证明"；RQ-3 加 RODNet 式单目深度定点对照。
- §5.4 助教分两种：容量助教（TAKD 式，中等容量、清晰输入）与信息助教（MonoTAKD 式，同容量、中等退化输入）。
- §6.4 注明下游代理来自 ProbEn、φ 系数来自集成多样性文献；把"模态丢弃可能增加支路相关"（ModDrop 式 (15)）写成 RQ-4 的待检验假设。

## 12 讨论记录：形成本草案时修正过的判断

| 初始说法 | 修正 | 依据 |
| --- | --- | --- |
| 双输入融合加互蒸馏会"侵入课题 A" | 不成立。A 是 SLAM 的逐轴定位完好性监测，不涉及检测融合；实际交叠的是 G3 浊度感知调度线和基础模型那篇 | 评审稿 §4、A 详案 v2.4 |
| 基础模型那篇提供了"同时融合两种模态的方法" | 视觉基础模型只用在相机侧（冻结 DINOv2-S），声呐侧是冻结 ResNet-18，任务是真值框裁剪识别；可借用的只有 D0–D4 协议和"训练期必须见过退化"这条结论 | [[2026-degradation-aware-foundation-fusion-umod\|笔记]] §2 |
| 学生也能看到相机时，相机教师的蒸馏会失效 | 说得太绝对，已撤回。大教师 → 轻量学生在输入相同时仍有容量收益（用户指出）；准确的说法是收益来源改变，主要被相机支路吸收，所以需要 2×2 归因 | 第 1 节 Q5、第 6.3 节 |
| 只有合成退化能提供清晰 / 退化配对 | 部分修正。H1 的导轨数据在同一刻度位置提供清水与浊水的近配对，可做真实层的特权蒸馏，但不是逐像素配对 | H1 v3.0 §2.3、§5（导轨锁定、刻度读位置） |
| 目标级视声匹配可帮 B 收缩仰角候选 | 作废。B v2.3 仿真已否定双目关联的收缩作用 | B 详案 v2.3 §3.3 |
| 用 gate 输出当教师可靠性权重 | 不采用。原文声明 gate 权重不是校准概率；改用可对照误差检验的量（教师声呐头 objectness），并先验证再使用 | 第 4 节张力 5、第 5.4 节 |
| 双模型驻留会让显存与时长增加 3–5 倍 | 高估。教师输入固定为清晰视角，输出可以离线缓存 | 第 5.2 节 |

## 13 关联页面

- 实施准备：[[plans/2026-10/topic-f2-codebase-selection-and-setup-2026-10|F2 代码库选型与搭建方案]]（2026-10-29；本仓库副本已于 2026-11-01 删除，仍有效的事实并入 `doc/f2-rethink-2026-11-01.md` §6）
- 思路复盘与路线：`doc/f2-rethink-2026-11-01.md`（2026-11-01）
- 父课题与兄弟课题：[[plans/2026-10/topic-f-low-label-sonar-adaptation-detailed-2026-10|F 详案]] · [[plans/2026-10/intern-project-2026-f1-sonar-low-label-adaptation-2026-10|F1 v1.4]] · [[projects/f1-sonar-low-label|F1 项目主页]]
- 使用方与数据供给：[[plans/2026-10/topic-c-persistent-target-mapping-detailed-2026-10|C 详案]] · [[plans/2026-10/intern-project-2026-h1-rov-platform-testbed-2026-10|H1]] · [[plans/2026-07/turbidity-aware-week1-2-experiment-protocol|浊度协议]]
- 原则来源：[[plans/2026-10/topic-a-direction-aware-localization-detailed-2026-10|A 详案 v2.4]]（H-A1） · [[plans/2026-10/topic-b-ambiguity-aware-spatial-mapping-detailed-2026-10|B 详案 v2.3]]（§3.3 否定结果）
- 组合与教训：[[plans/2026-10/acoustic-optic-six-topics-internal-review-2026-10|六选题评审稿]] · [[comparisons/research-topics-genealogy-2026-10|课题谱系]] · [[plans/2026-07/turbidity-aware-modality-scheduling-paper-spec|G3 浊度感知调度规格]] · [[plans/2026-09/research-directions-2026-09-dataset-first|D1–D5 方向卡]]
- 概念：[[concepts/acoustic-optic-fusion|声光融合]] · [[concepts/datasets-and-simulators|数据集与仿真器]]

## 14 TODO

本节建于 2026-10-29，来源有两处：[[plans/2026-10/topic-f2-codebase-selection-and-setup-2026-10|F2 代码库方案]]的核验结果，以及本草案第 7、10、11.1 节里还没落实的事项。完成一项就勾选一项，并在行尾写上日期和结果页链接。

**核验背景**：RSFusionDet 官方仓库是空的，UMOD 公开部分只有约 259 对。所以 S0 写的"跑通 RSFusionDet 官方代码"目前无法执行。下面各项都按这个前提排序。

### 14.1 待决（团队或指导人决定，阻塞后续）

- [x] **S0 数据路线**：定为 b（UXO 上自建），a 继续监视。（2026-10-30 起执行）
- [ ] **F1 W2 框架是否也定为 mmdetection**：如果定了，F1 和 F2 共用 `ao-det` 仓库、划分脚本、M₀ 和评测（方案页 4.4 第 8 条）。
- [ ] **训练机器**：需要 24 GB 级显卡，并且能放下 UXO 全量 96 GB（解压后更大）。（硬件阻塞已解除：2026-10-31 发现本机已挂 RTX 5090 D v2 24 GB，磁盘 944 GB 可用；`.venv-gpu` 环境已建并验证：pinned cu121 栈在 Blackwell sm_120 上实测不可用（无内核映像），改用 torch 2.7.1+cu128 + mmcv 2.1.0 CPU-only 源码构建 + 推理期 NMS CPU 回退，全套测试通过，S0-lite 可在本机跑，见 `env/ENVIRONMENT.md`；是否用本机承担全量训练仍与 F1 W1 合并待团队确认）
- [ ] **库内"已开源"表述是否更正**：涉及三处，见 14.4。

### 14.2 数据

- [ ] 路线 a：由团队向 RSFusionDet 作者发邮件，索取代码和 RSFusion 数据，并说明用途和引用方式。邮件由团队成员发送。
- [ ] 每两周检查一次 `LEFTeyex/RSFusionDet`。看到提交时，记录 commit、框架版本和数据下载方式。（机制已建立：`tools/check_rsfusiondet.sh` + `doc/logs/rsfusiondet-watch.log`；2026-10-29/30 两次核验均为空仓库）
- [x] 路线 b：下载 UXO `sample.7z`（1.7 GB）。用官方 `dfki-ric/uxo-dataset2024` 的脚本读取，核验三件事：标签坐标"乘 3"的规则；issue 报告的框与原始 GoPro 帧对不上的问题；外参。（2026-10-30 完成：md5 精确匹配；①×3 规则定量+目检成立（labels 在 640×360 空间、随包 jpg 为 1920×1080，×3 框在 2 个 recording 样帧上套住目标）；②"对不上"成因=把 ×3 框套到 5.3K 原始帧（缩放比 2.77≠3），随包 jpg 上不复现，转换器以随包 jpg 为准；③外参齐备（transforms.yaml TF 树+4 份 GoPro 内参+逐帧 gantry/ARIS 元数据）。类别 3 种 2616 标注帧。详见[[logs/2026-10-30-f2-codebase-w1-w3|日志]]）
- [ ] 根据 sample 的核验结果，决定是否下载全量，以及下载哪几个包（`data_export_polar` 16.5 GB、`data_processed` 61.5 GB 等）。（2026-11-01 复盘建议下载，作为 E11 前置项，待用户决定）
- [x] UXO 声呐框投影：用官方位姿和标定投影，同时生成跨模态对应 π；抽样做人工目检，记录投影误差。（2026-10-31 完成：`aodet/datasets/converters/uxo_projection.py` 逐 recording 三阶段稳健标定（相机方位→[R_son,b_rng]→10 参联合 Huber）+ `tools/project_uxo_sonar.py`；合成水槽 9 测试覆盖极坐标/瞄准×弧线/直线四几何，P↔b 退化用先验+|b|≤0.8 m 硬界处理。实数据 2573 帧全部落框：blob 门控 1185 / 几何投影 1388 / 丢弃 0，pairs.csv π 就绪。误差如实记录：直线录音最好（dcol p50 0.75–1.23 beams）；弧线录音存在 ~2° rms 的未建模时变瞄准残差（行程端部达 10–30 beams）；100927 的 b 顶在 −0.8 界上、P_z 偏深（该录音退化活跃）。已知局限与非作者人工目检待办见[[logs/2026-10-30-f2-codebase-w1-w3|日志]]投影一节。）
- [ ] `project_uxo_sonar.py` 加 `--calib-cache`：逐 recording 标定结果（含逐帧拟合诊断）落盘复用，键=输入指纹（源标注+位姿+参数的 hash）。动因：GT 语义修正后 `--write` 重写时，与 dry-run 完全相同的标定被从零重算：~50 分钟里 ~49 分钟是重复计算；全量 UXO（96 GB、数十 recording）会按比例放大这笔浪费。须在**全量转换前**完成，并加缓存命中/未命中结果一致的等价测试。
- [ ] UMOD：只做 objectness 小样本 sanity check，不再列为外部测试集；以后补全了再评估。
- [ ] 与 H1 约定数据使用方式（第 10 节"评审"一项；G1 的前提）。

### 14.3 代码库搭建（方案页第 7 节，路线 b 成立时执行）

- [x] W1：建立独立 conda 环境，Python 3.10、torch 2.1.2+cu121、mmcv 2.1.0、mmengine 0.10.7、mmdet 3.3.0、numpy 1.26；保存 `pip freeze`、`pip check` 和 CUDA 信息。（2026-10-30，调试机以 venv+CPU wheel 完成，cu121 环境文件就绪待训练机；[[logs/2026-10-30-f2-codebase-w1-w3|日志]]）
- [x] W1：mmdet YOLOX-S 在 COCO val 子集上复现官方精度，差距在 1 AP 以内，以此确认安装正确。（2026-10-30，val2017 全集 5000 张 AP 40.30 vs 官方 40.5，差 0.2，通过；过程中发现并修复 YOLOX 0–255 BGR 预处理坑——ImageNet 归一化会把 AP 打到 0.004；另证实按 id 排序子集偏易（first-1000 AP 42.2），gate 必须用随机采样；[[logs/2026-10-30-f2-codebase-w1-w3|日志]]）
- [x] W1：建 `ao-det` 仓库骨架；`third_party/` 下固定 commit 引入 KD-YOLOX-ViT、Megvii YOLOX、uxo-dataset2024、ProbEn、CRKD，只读，不 import。（2026-10-30，另含 SCFusion 共 6 仓库；SHA 见 `third_party/PINNED.md`）
- [x] W2：写配对格式（`camera.json`、`sonar.json`、`pairs.csv`、逐帧元数据）和 UXO 转换器。（2026-10-30 完成：转换器含路径白名单/×3 框/recording 级 split/frame_meta 投影字段，8 项新测试；实数据转换 2573 对，跳过原因逐项计数且与清点吻合（43 孤儿 label）；16 格拼图目检 15/16 套住、1 格为源标注部分框噪声（`105856/0282`，非转换器 bug）；sonar 标注与 pairs 留待投影工具；[[logs/2026-10-30-f2-codebase-w1-w3|日志]]）
- [ ] W2：写相机退化算子，包括 D0–D4、D∞ 和只有色偏一档，按 (sample_id, level, seed) 确定性生成；离线生成退化测试集，记录哈希；样张交给非作者成员目检。（算子与离线集工具+哈希 manifest+确定性测试均完成；"非作者成员目检"待人工执行，2026-10-30）
- [x] W2：方案页 4.5 的测试全部通过。测试包括退化单调性与确定性、几何一致性、转换器往返、相关性指标、逐支路 AP 对拍；失败时返回非零退出码。（2026-10-30，55 项测试全绿；[[logs/2026-10-30-f2-codebase-w1-w3|日志]]）
- [x] W2：确定声呐输入用极坐标矩形图还是扇形图。要结合团队声呐只能输出图像的判定，并考虑 L_xp 实现时的便利。（2026-10-30，定为极坐标矩形图、x=方位角升序、0° 居中；依据与对 L_xp 的影响见[[logs/2026-10-30-f2-codebase-w1-w3|日志]]）
- [x] W3：双流 YOLOX-S 加三个头（相机头、融合声呐头、未耦合声呐头），关闭 Mosaic 和 MixUp。（2026-10-30，`TwoStreamYOLOX`+`configs/f2/` 两份 S0-lite 配置；合成数据端到端冒烟通过）
- [x] W3：S0-lite，训练 B0 和交叉注意力融合各 1 个种子，做 D0–D4 推理期扫描，输出逐支路 AP 表和相关性表。（2026-10-31 至 11-01 经 E04–E09 完成；声呐读数在地板，相关性表无信息量，见 `doc/experiments.md`）
- [x] W3 结束：根据 S0-lite 的结果决定是否扩到 3 个种子和全量 UXO。（2026-11-01：S0-lite 平台测不出声呐效应，扩到全量 UXO 均衡划分 + 3 种子，见 E11；F2-M0 状态记为不可测）
- [x] 后续：先写在线冻结教师；离线教师缓存要通过等价测试后才能启用（方案页 4.4 第 4 条）。（2026-10-30，在线 `FrozenTeacher`+branch_kd 已实现并有测试；缓存按约定延后）

### 14.4 wiki 更正（待 14.1 最后一项决定）

- [ ] [[2026-rsfusiondet-rgb-sonar-detection-heu|RSFusionDet 笔记]]："代码与数据开源"改为"仓库为空（2026-10-29 核验）"，可复现性评分下调。
- [ ] [[2026-vsafdet-umod-visual-sonar-detection|UMOD 笔记]]：补上"网盘实际约 259 对，类别缺失"。
- [ ] [[plans/2026-10/intern-project-2026-f1-sonar-low-label-adaptation-2026-10|F1 v1.4]]：§6 主数据和"守恒约束"段里 RSFusion"已开源"的说法；同时评估 F1 主数据是否改用 UXO。
- [ ] 本草案第 0.4、3、6.1、9 节：S0 改写为依赖数据路线的形式，第 9 节加"官方代码不可得"一行风险。

### 14.5 立项评审前（草案原有事项）

- [ ] G0：S0 结果为正（第 7.1 节）。路线 b 下阈值和结论层级要重新表述。
- [ ] G1：H1 浊度阶梯数据采完，最早约 2027-03。
- [ ] G2：C 的 M3 错误归因显示检测误差占主要部分，或者产品验收提出了这一需求。
- [ ] 阈值初值（"种子间标准差的 2 倍""恢复到 B0 以上"）与 F1 门槛一并审定（第 7.3 节）。
- [ ] 冻结学生的部署上限（参数量、FLOPs、延迟），前提是机载算力已经核实（第 2.1、9 节）。
- [ ] 检测家族在 S1 前最终确认，默认 YOLOX（第 5.1 节）。
- [ ] 第 11.1 节"建议的正文修改"五条，由立项评审决定是否并入正文。
- [ ] RQ-3 加入 RODNet 式"单目深度定点"对照（第 11.1 节结论 2）。

### 14.6 E11-2 收尾与 E11-4 发车（2026-11-04 建，训练机上执行）

约定：仓库根目录执行，GPU 步骤先 `source .venv-gpu/bin/activate`；E11 各 run 落在 `work_dirs/<config 名>_seed<N>/`，评测输出在其下 `eval_<split>/`（E11-1 惯例）。B0 只有非融合声呐头，预测文件是 `pred.sonar_uncoupled.<level>.json`；B2 系的融合声呐头是 `pred.sonar.<level>.json`。所有评测显式 `--split`，不碰 test。

- [ ] **B2 链停在 2 种子**（台账 E11 节"单种子协议"）。先看链进度：`tail -n 20 work_dirs/e11_b2_chain.log; ls work_dirs/e11_b2_cross_attn_nominal_yoloxs_seed*/`。不要改正在运行的 `/tmp/e11_b2_chain.sh`（bash 边读边执行，改文件会让在跑的链读到错位内容）。等 seed 1 的 val 评测落盘后、seed 2 训练开始前停链：`pkill -f e11_b2_chain.sh`，再用 `pgrep -af "train.py.*e11_b2_cross_attn"` 确认没有 seed 2 训练进程（有则 kill），在链日志里补一行停链原因。
- [ ] **B2a 冒烟**（1 epoch，含 val 评测与 γ 读数）：
  ```
  python tools/train.py configs/f2/e11_b2a_gamma_fix_yoloxs.py --work-dir work_dirs/e11_b2a_smoke --seed 0 --cfg-options train_cfg.max_epochs=1
  ckpt=$(cat work_dirs/e11_b2a_smoke/last_checkpoint); [ -f "$ckpt" ] || ckpt=work_dirs/e11_b2a_smoke/$ckpt
  python tools/test.py configs/f2/e11_b2a_gamma_fix_yoloxs.py "$ckpt" --out-dir work_dirs/e11_b2a_smoke/eval_val --levels d0 dinf --split val
  python tools/fusion_gamma.py work_dirs/e11_b2a_smoke
  ```
  通过标准：训练日志里 camera / fused_sonar / uncoupled_sonar 三组 loss 共 9 项、全部有限；`eval_val/` 下 camera、sonar、sonar_uncoupled 三支路的 d0 与 dinf 预测文件都在；三层 mean |γ| 仍在 0.1 附近（没有被衰减拉回 0）。
- [ ] **B2b 冒烟**：同上，配置换 `configs/f2/e11_b2b_sonar_qdrop_yoloxs.py`，work-dir 换 `work_dirs/e11_b2b_smoke`。通过标准同 B2a。另记 s/iter，与 B2a 比较，查声呐查询丢弃有没有带来额外耗时。
- [ ] **B2 融合活性读数**（E11-4 触发判定；B2 seed 0/1 评测落盘后）：
  ```
  for s in 0 1; do python -c "import json,sys; r={(x['branch'],x['level'],x['gt']):x['AP50'] for x in json.load(open(sys.argv[1]))}; d0=r[('sonar','d0','blob')]; di=r[('sonar','dinf','blob')]; print(sys.argv[1], 'd0', round(d0,4), 'dinf', round(di,4), 'd0-dinf', round(d0-di,4))" work_dirs/e11_b2_cross_attn_nominal_yoloxs_seed$s/eval_val/ap_summary.json; python tools/fusion_gamma.py work_dirs/e11_b2_cross_attn_nominal_yoloxs_seed$s; done
  ```
  两种子 d0−dinf 均值 ≤ 0.032：融合惰性，触发 E11-4（下一条）。> 0.032：融合活跃，E11-4 不跑，判据 5 直接按 B2 判。
- [ ] **类别移植上界**（Bcam seed 0 落盘即可跑，不等 B2）：
  ```
  python tools/class_transplant.py --sonar-pred work_dirs/e11_b0_sonar_only_yoloxs_seed0/eval_val/pred.sonar_uncoupled.d0.json --camera-pred work_dirs/e11_bcam_camera_only_yoloxs_seed0/eval_val/pred.camera.d0.json --paired data/paired/uxo_full --split val --tag b0s0+bcams0 --out work_dirs/e11_b0_sonar_only_yoloxs_seed0/transplant.bcam_s0.json
  python tools/class_transplant.py --sonar-pred work_dirs/e11_b0_sonar_only_yoloxs_seed0/eval_val/pred.sonar_uncoupled.d0.json --camera-pred work_dirs/e11_b2_cross_attn_nominal_yoloxs_seed0/eval_val/pred.camera.d0.json --paired data/paired/uxo_full --split val --tag b0s0+b2s0 --out work_dirs/e11_b0_sonar_only_yoloxs_seed0/transplant.b2_s0.json
  ```
  上界 = camera_hard 与 camera_soft 中较高者减去 original。> 0.063 才满足 E12 启动前提。
- [ ] **Δ_fuse(d0) 与 Δ_s(s)**：B2 融合声呐头（`branch=sonar`）减去 B0 非融合声呐头（`branch=sonar_uncoupled`），同为 gt=blob，逐档 d0–d4、dinf，都从各自的 `eval_val/ap_summary.json` 读取。逐类与逐录音分解读 `ap_breakdown.json`，bomb 单列。
- [ ] **E11-4 发车**（仅在融合惰性时，且两臂冒烟都已通过）。两臂同卡并跑，种子 0；链脚本以 `/tmp/e11_b2_chain.sh` 为模板（`.ok`、心跳、重试两次停链），每臂的步骤：
  ```
  cfg=configs/f2/e11_b2a_gamma_fix_yoloxs.py; work=work_dirs/e11_b2a_gamma_fix_yoloxs_seed0   # B2b: e11_b2b_sonar_qdrop_yoloxs
  python tools/train.py $cfg --work-dir $work --seed 0
  ckpt=$(cat $work/last_checkpoint); [ -f "$ckpt" ] || ckpt=$work/$ckpt
  python tools/test.py $cfg "$ckpt" --out-dir $work/eval_val --levels d0 d1 d2 d3 d4 dinf --split val
  python tools/test.py $cfg "$ckpt" --out-dir $work/eval_val_id --levels d0 --split val_id
  python tools/test.py $cfg "$ckpt" --out-dir $work/eval_train --levels d0 --split train
  python tools/fusion_gamma.py $work
  ```
  发车后十分钟内人工确认一次（s/iter、9 项 loss、显存），把实测 ETA 写进台账。
- [ ] **出数 24 h 内入账**：按台账 E11-4 节判据 1–7 逐条过账（阈值 0.063，读数落在阈值 ±0.02 带内的报用户），同步台账 §0、问答板 Q6 和 CLAUDE.md 主线卡点；融合活跃且 Δ_fuse(d0) > 0.063 的那条臂，作为 E12/E13 学生侧的融合基线写进 E12 预登记。
