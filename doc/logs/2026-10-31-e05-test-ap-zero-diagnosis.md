# E05 test AP≈0 二次排查 → b_rng 锚定修正（2026-10-31）

> 承接 [E04 首跑失败](./2026-10-30-f2-codebase-w1-w3.md 末节)：GT 钳制
> （`_PROJ_BOX_MAX=(40,60)`）重写后的重训进行中，用户要求仔细排查
> COCO AP/AP50 仍近乎为 0 的根因。结论先行：**钳制只修了框的尺度，
> 没修框的位置**——test 两段录音的 projected GT 在距离维（row）带着
> ~0.28–0.30 m 的标定系统偏移，66% 的 test GT 与任何真实回波检测的
> IoU 恒 < 0.5。骨架记录在 `doc/experiments.md` E05/E06；本页是排查
> 链与修复细节。

## 症状与快速复现

B0 重训（GT 钳制后）进行到 epoch 10/100 时用 `tools/test.py --levels
d0` 探针评测：test AP=0.0000 / AP50=0.0001——与 E04 同样的全灭数字。
但此时**训练侧信号全部正常**（loss 收敛、grad_norm 稳定）。

## 排查链（每步排除一个假设）

1. **评测实现**：oracle 测试（GT 自身当预测）双侧 AP 精确 = 1.0
   → id 空间（frame_meta ↔ sonar/camera.json）、类别映射
   （categories 位置枚举）、xywh 转换、split 过滤全对，
   **评测代码无 bug**。
2. **预测解剖**：保存的 pred 文件坐标在 128×780/441 原图空间、
   尺度与 blob 同量级、类别 1/2/3、image_id 在 test 范围
   → 坐标回映射（YOLOXHead scale_factor 除法）无 bug。
3. **图像尺寸**：全量 2573 张声呐 pgm 与 json 声明一致
   （128×780 一段、128×441 四段）→ 无转置/尺寸错位。
4. **train split 自评**（同一 epoch_10 检查点）：AP50 = **0.80**。
   → 训练管线（数据、GT、损失、预处理）完全健康；问题被隔离到
   **训练域 → 测试域**之间。
5. **GT 光度检验**：blob GT 框内对比背景 +54.6（真回波）；
   projected GT **+0.3 ~ +2.4（与背景无异）**——66% 的 test GT
   框内没有任何可见回波。
6. **错位方向分解**（GT 中心 vs 图像亮斑）：projected GT
   |dx| p50 = **0.4–0.9 px**（方位完美）、|dy| p50 = **92–101 px**
   （距离维系统偏移）。blob GT 的 |dx|/|dy| 都小。
7. **闭环对账**：dy × 3.0 mm/row = 0.28–0.30 m，与 E03
   `projection_report.json` 里 105856/124735 的 `rms_rng_m`
   **0.261 / 0.295**（五段中最大，其余 0.022–0.067）吻合——
   信号一直都在表里，只是当时被「dcol 只看列向 + 过门自洽」的
   口径掩盖。

证据存档：`work_dirs/s0_b0_sonar_only_yoloxs_seed0/e05_probe/`。

## 根因机制（自洽链）

- 标定的 `(P_radial, b_rng)` 退化（目标近弧心，距离剖面平坦）让
  系统性距离偏移漏进 P，`_SIG_B_PRIOR=0.25` 先验把 b_rng 拉在 0 附近。
- `sonar_bbox` 的 blob 门（`_GATE_RNG=0.20 m`）把偏移 >0.20 m 的
  真回波帧全部降级为 `projected` → 过门率低（105856 40%、124735 28%），
  **门内统计（rng rms）循环自洽**，门外的帧从未被距离维验证。
- 投影框中心 = 投影点 → 全部 projected 框 row 同向偏 ~100 px（框高
  仅 60 px）→ 评测 IoU 恒小；训练侧同源框也在毒化（100927 偏 24 px）。
- 次要因素：弧线录音训练 → 直线录音测试的域移位 + 上述毒化，使
  epoch_10 模型在 test 上的预测 row 也偏（105856：预测 119 vs
  blob GT 176）。即便模型完美，test AP50 上限也被 66% 坏 GT 压到
  ~0.34。

方法论教训（接 E04）：**投影式 GT 的"语义"要连同"位置可验证性"
一起审——方位对齐 ≠ 距离对齐；门控统计（过门率、分源残差）必须
连同门外的帧一起看，否则门会把自己的失效模式筛成不可见。**

## 修复（三层，用户拍板）

1. **主修：`anchor_b_rng`**（`uxo_projection.py`）——标定后冻结
   其余参数，对**仅过方位门**的 blob 帧取 median((blob_row −
   proj_row)·dr) 加回 b_rng。predict 的 row 对 b_rng 仿射、col 无关
   → 一步中位数修正精确；中位数对锁在池底亮带的误检 blob 稳健。
   方位-only 门是刻意的：距离门会拒掉携带信号的帧（循环）。
2. **评测口径**：`tools/test.py` 声呐侧输出 `gt=all` 与 `gt=blob`
   双口径（`filter_gt_sources`）；`collect_results.py` 加 `gt` 列。
3. **训练口径**：`PairedSonarCameraDataset(sonar_gt_sources=...)`
   可只训 blob 框（帧保留 → 无框帧变负样本，权衡假负例）。

配套：`tools/project_uxo_sonar.py --use-cache` + 每次落
`projection_calib.json`（TODO ad4f178 的 calib-cache；重写 GT 从
~50 min 降到秒级）。新增 5 测试：锚定恢复合成偏移、无 blob 透传、
cache 往返 predict 逐位一致、filter_gt_sources、dataset 源过滤。

## 处置

- E05 双 run（B0 epoch~25 / cross-attn epoch~10 时）停止，证据留档，
  E05 关闭为「❌ 失败（保留为反例）」，重开 E06。
- b_rng 锚定 dry-run 验证 → `--use-cache --write` 重写 GT → 复检
  （projected |dy|、光度对比度、过门率）→ E06 重训。

---

## 勘误与终判（2026-10-31 晚，接 E06 gate 与判别实验）

> 本页上半部分的结论「projected GT 带 ~0.28–0.30 m 距离维系统
> **标定**偏移」**不成立**，保留原文于上仅作反例。证据链：

1. **锚定 dry-run 反证**：anchor_b_rng 对 blob 帧测得 median
   (blob−proj)·dr ≈ 0（100927 −0.792 m 为基线 b_rng 本身，锚定
   修正量微小）——标定没有可修的系统偏移。
2. **上页 dy 证据的方法缺陷**：① 自写亮斑检测器在暗帧锁到池底
   亮带（peak<40 帧未剔除，E05 §5/§6 的 92–101 px 是「框到带」
   的距离）；② 40×60 钳制框的框均值对比度物理上测不出回波
   （回波 ~15×10 仅占框内 ~15% 面积）。
3. **判别实验（区分恒定偏移 vs 无关结构）**：直线两段录音上，
   blob 帧条带峰行与框行 corr = 0.94/0.87（回波跟随投影，dy
   p50 = 5/2 px）；projected 帧 corr = 0.20/−0.01（**恒定偏移
   假设下应为 1.0**）——projected 帧的条带峰是池底带等无关结构，
   dy≈80 px 不携带标定信息。
4. **projected 帧的语义**：有回波且过门的帧按构造都成了 blob 帧，
   projected 帧 = 无过门回波的帧（对比度 +0.1~+7 ≈ 空框；弧线段
   dy 小是距离近恒定 + 目标贴底使带与目标行重合的巧合）。它们
   不是「可见回波区域」GT，是无回波帧上的几何猜测。

**终判（取代上文根因机制节）**：E05 test AP≈0 是两件事叠加——

- **主因：E02 划分的几何混淆**（3 弧全 train、2 线全 test）→
  弧→线域移位。逐 recording 评测：弧线（train 段）AP50
  0.743–0.853，直线（test 段）0.000–0.002，与 GT 口径无关
  （blob-only 同样全灭）。
- **次因（GT 语义）**：test 66% 的 GT 框落在无回波帧上，即便
  域移位修复，全 GT 口径的 AP 上限也被压低。

**处置更新**：训练 GT 改 blob-only（`sonar_gt_sources=("blob",)`，
无回波帧保留为诚实负样本）；E02 划分重划为 E07 split（test =
100927 弧 + 124735 线，每段恰一个新颖维）；重训开 E07（判定标准
经 E06 结论修订后预登记，见 `doc/experiments.md` E07 节）。
