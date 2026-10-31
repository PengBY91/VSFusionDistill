# F2 代码审查：蒸馏 / 损失 / 架构 / 训练逻辑（2026-10-31）

> **范围**：对照 `doc/topic-f2-degradation-robust-fusion-detection-distillation-2026-10.md`（草案）、
> `doc/topic-f2-codebase-selection-and-setup-2026-10.md`（方案页）、`doc/experiments.md`（台账），
> 审查 HEAD `1a69ba5` 的实现。重点是**蒸馏方法、损失函数、模型架构、训练逻辑（迭代、损失计算、反传）**。
>
> **证据等级**：本次审查在一份全新的 clone 上进行，没有 `.venv`、`data/` 和 `work_dirs/`，因此**没有在真实数据上复跑**。
> 结论来自读代码，外加最小数值验证（E08 抖动抵消见 §1.2）。§1–§4 中每条都注明了"已验证 / 读码确认 / 需本机核实"。
>
> **补丁状态**：P0–P2 以及 §3.6(a–c)、§4.1–§4.4 已在分支 `fix/review-2026-10-31-kd-arch-training` 实现，并在 CPU `.venv` 下跑过全量测试（结果见 §6）；`.venv-gpu` 尚未复跑。
> 实际实现与下文示例代码在细节上可能不同，以代码为准。未实现的项：§2.4 的方位重采样拼接（只实现了 `cam_pool="global"`）、§3.6(d) 的射线曲线 MIL 与 M₀ 针孔化 / 按录音加载、§3.7 的 2×2 改动（需评审决定）、§4.1 的 ECE 列、§4.2 训练端"跳过 projected 帧"选项。

---

## 0. 问题总表

| # | 级别 | 模块 | 问题 | 影响 | 修复节 |
|---|---|---|---|---|---|
| 1 | 🔴 | 训练 | KD 损失键名不含 `loss`，不参与反传 | 所有蒸馏实验（M1–M5）等于没有蒸馏 | §1.1 |
| 2 | 🔴 | 训练 | E08 尺度抖动被后续固定尺寸 resize 完全抵消 | E08 的结论必然是假阴性 | §1.2 |
| 3 | 🔴 | 架构 | CrossAttn 的 D∞ 训练路径（黑图过注意力和 LN）与测试路径（drop、跳过 LN）不一致 | D∞ 档"互毒害"会被人为制造出来 | §2.1 |
| 4 | 🔴 | 架构 | 交叉注意力没有位置编码，也没有几何先验 | 学不到"相机列 ↔ 声呐方位"的对应；S0 测的不是 CAFusion 类机制 | §2.2 |
| 5 | 🔴 | 损失 | objectness KD 用全图 MSE，前景信号被背景稀释，量级比检测损失低几个数量级 | 即使修好 #1，KD 也近似无效 | §3.1 |
| 6 | 🔴 | 损失 | 类别 KD 用全图 softmax-KL，与 YOLOX 的逐类 sigmoid、只训正样本不匹配 | 蒸馏的是背景位置上的无语义 logit | §3.2 |
| 7 | 🔴 | 评测 | `correlation_table.py` 不按 split 过滤，train 帧被记成"双支路同时漏检" | φ、ΔP_miss 被系统性虚高 | §4.1 |
| 8 | 🟠 | 训练 | cross-attn 模型的声呐骨干承受两份声呐检测损失，B0 只有一份 | Δ_s 里混进了损失权重差 | §1.3 |
| 9 | 🟠 | 训练 | 没有预训练初始化；lr 是 YOLOX 标准的 8 倍；BN/bias 做了 decay；没有 EMA；warmup 与 cosine 重叠 | 过拟合和跨录音泛化差无法单独归因 | §1.4 |
| 10 | 🟠 | 训练 | D∞ 黑图上仍按相机 GT 计算相机检测损失 | 相机头学会靠位置先验幻觉出框 | §1.5 |
| 11 | 🟠 | 训练 | 一次迭代里融合模块和各头前向两次 | 算力约翻倍，且 KD 与检测用的是两次独立前向 | §1.6 |
| 12 | 🟡 | 训练 | `AOSampleDegLevel` 的 rng 种子写死为 0，`deg_seed` 恒为 0 | worker 间、训练种子间档位序列相同；3 种子方差里不含退化抽样方差 | §1.7 |
| 13 | 🟠 | 架构 | 未耦合声呐头只在推理时解耦，训练时与融合头共用骨干 | RQ-4 的"独立性"读数需要重新定义 | §2.3 |
| 14 | 🟠 | 架构 | `ConcatFusion` 把相机特征插值到声呐网格后逐像素拼接 | 默认两平面已配准，是草案批评 CMHD 的同一错误 | §2.4 |
| 15 | 🟠 | 蒸馏 | 有融合头时，未耦合声呐头不接受 KD（`elif`） | 测不到"特权信息进入声呐本身"（Q5 机制） | §3.3 |
| 16 | 🟠 | 蒸馏 | 可达性代理量若取教师融合头，会形成循环论证 | §5.4 落地时容易做错 | §3.4 |
| 17 | 🟠 | 蒸馏 | `FrozenTeacher` 以 `strict=False` 加载 | 键名不匹配时教师为随机权重，且无报错 | §3.5 |
| 18 | 🟠 | 损失 | L_xp 不是单边约束；硬 max；方位→列为线性映射；M₀ 忽略视差和畸变 | 违背"只传正证据"；几何误差大 | §3.6 |
| 19 | 💡 | 设计 | 2×2 需要训练 4 个教师，格子之间不是配对比较 | 成本高、方差大 | §3.7 |
| 20 | 🔴 | 设计 | 没有验证集，test 录音被用于诊断和定参（E08 的抖动范围来自 100927） | 违反草案 §6.1"测试集只评测一次" | §4.3 |
| 21 | 🟠 | 设计 | S0-lite 的融合模型带退化训练（接近 B4），缺少名义训练的 B2 | RQ-0 问的问题没有被直接测量 | §4.4 |
| 22 | 🟠 | 评测 | blob 口径删掉了 projected 帧的 GT，但这些帧仍在 `img_ids` 中 | 在这些帧上的正确检测全部算作虚警 | §4.2 |
| 23 | 🟡 | 文档 | E01 称 NMS 0.65 / score 0.01 已固化进模型配置，实际仍是 0.55 / 0.05；TODO 14.2 calib-cache 已完成但未勾选 | 台账与代码不一致 | §4.5 |
| 24 | 🟡 | 数据 | test 录音 100927 正是标定最可疑的一段（b 顶界，锚定 −0.79 m） | GT 质量与泛化性能混在一起 | §4.3 |

建议修复顺序见 §5。

---

## 1. 训练逻辑（迭代、损失计算、反传）

### 1.1 🔴 KD 损失不参与反传（读码确认）

**位置**：`aodet/models/detectors/two_stream_yolox.py:223`

```python
kd = branch_kd_losses(s_out, t_out, cfg=self.distillation_cfg)
return {f"kd/{k}": v for k, v in kd.items()}      # 键名如 "kd/sonar/obj"
```

**机制**：mmengine `BaseModel.parse_losses` 只把键名含 `"loss"` 的项加进总损失：
`loss = sum(value for key, value in log_vars if 'loss' in key)`。
`kd/sonar/obj` 和 `kd/camera/obj` 都不含 `loss`，所以只会写进日志，梯度恒为 0。

**测试为什么没发现**：`tests/test_two_stream_model.py:201` 写的是

```python
sum(v for k, v in losses.items() if "loss" in k).backward()
```

它复刻了同一个过滤规则，KD 项同样被排除在外。测试只断言 KD 键存在，没有断言 KD 产生了梯度。

**修改**：

```python
# two_stream_yolox.py  _kd_losses 末尾
kd = branch_kd_losses(s_out, t_out, cfg=self.distillation_cfg)
# 键名必须含 "loss"，否则 mmengine parse_losses 不会把它加进总损失
return {f"kd/loss_{k.replace('/', '_')}": v for k, v in kd.items()}
```

**新增测试**（`tests/test_two_stream_model.py`）：

```python
def test_kd_loss_reaches_student_gradients():
    """KD 项必须进入 mmengine 的总损失并产生梯度。"""
    sbb, snc, shd = _stream_cfg()
    teacher_cfg = dict(
        type="TwoStreamYOLOX",
        sonar_backbone=sbb, sonar_neck=snc,
        camera_backbone=_stream_cfg()[0], camera_neck=_stream_cfg()[1],
        camera_head=_stream_cfg()[2], uncoupled_sonar_head=shd)
    student = _model(
        fusion=dict(type="NoneFusion"),
        distillation=dict(teacher=teacher_cfg, teacher_input="clear",
                          obj_weight=1.0, cls_weight=0.0))
    data = _batch(b=2, with_clear=True)
    inputs, samples = _prep(student, data, training=True)
    student.train()
    losses = student.loss(inputs, samples)
    # 用 mmengine 自己的解析逻辑，而不是在测试里复刻过滤规则
    total, log_vars = student.parse_losses(losses)
    kd_keys = [k for k in log_vars if k.startswith("kd/")]
    assert kd_keys and all("loss" in k for k in kd_keys), kd_keys
    # 只对 KD 项反传，学生参数必须拿到非零梯度
    student.zero_grad()
    sum(v for k, v in losses.items() if k.startswith("kd/")).backward()
    g = [p.grad.abs().sum() for n, p in student.named_parameters()
         if p.requires_grad and p.grad is not None
         and not n.startswith("teacher.")]
    assert g and float(sum(g)) > 0
```

### 1.2 🔴 E08 尺度抖动被 resize 抵消（已数值验证）

**位置**：`configs/f2/e08_s0_b0_sonar_only_yoloxs.py:14–17`，`configs/f2/e08_s0_cross_attn_yoloxs.py`

**机制**：`AOSonarScaleJitter` 先把行数缩放为 `h·f`，紧接着 `AOPairedResize(sonar_size=(640,640), keep_ratio=False)` 又把图拉回 640 行，所以 `fy = 640/(h·f)`。最终框高等于 `h_box·f·640/(h·f) = h_box·640/h`，与 f 无关。

**数值验证**（441 行原图，17 行回波，cv2 实算）：

| f | 管线后 GT 框高 | 管线后图像中回波行数 |
|---|---|---|
| 0.6 | 24.63 px | 25 |
| 1.0 | 24.67 px | 25 |
| 1.8 | 24.66 px | 25 |

**后果**：E08 若按现在的配置启动，预登记标准 1 会判"仍 < 0.01 → 外观域主导成立，尺度抖动弃用"。这是一个**假阴性**，会把决策引向错误的分支。

**修改**：把抖动放到 resize **之后**，在固定画布上缩放行向，再裁剪或补边，保持画布尺寸不变。

```python
# aodet/datasets/transforms/paired_geometric.py —— 替换 AOSonarScaleJitter
@TRANSFORMS.register_module()
class AOSonarScaleJitter(BaseTransform):
    """Rescale the sonar range axis (rows) on a FIXED canvas — train-side.

    Must run AFTER AOPairedResize: a fixed-size resize placed after this
    transform undoes the jitter exactly (E08 review 2026-10-31).  The
    canvas keeps its (H, W); f > 1 crops an H-row window (preferring one
    that keeps every box), f < 1 pads with ``pad_val``.  Boxes are
    shifted and clipped; boxes thinner than ``min_box_h`` after clipping
    are dropped together with their labels.
    """

    def __init__(self, scale_range: tuple = (0.6, 1.8), prob: float = 1.0,
                 pad_val: int = 0, min_box_h: float = 2.0):
        super().__init__()
        if not 0 < scale_range[0] <= scale_range[1]:
            raise ValueError(f"invalid scale_range {scale_range}")
        self.scale_range = scale_range
        self.prob = prob
        self.pad_val = pad_val
        self.min_box_h = min_box_h

    def transform(self, results: dict) -> dict:
        img = results.get("img_sonar")
        results["sonar_scale_jitter"] = 1.0
        if (img is None or self.prob <= 0
                or tuple(self.scale_range) == (1, 1)
                or np.random.rand() > self.prob):
            return results
        f = float(np.random.uniform(*self.scale_range))
        h, w = img.shape[:2]
        new_h = max(1, int(round(h * f)))
        if new_h == h:
            return results
        scaled = cv2.resize(img, (w, new_h), interpolation=cv2.INTER_LINEAR)
        fy = new_h / h
        boxes = [[b[0], b[1] * fy, b[2], b[3] * fy]
                 for b in (results.get("boxes_sonar") or [])]

        if new_h > h:
            # crop: choose y0 so the window keeps every box when possible
            lo, hi = 0, new_h - h
            if boxes:
                top = min(b[1] for b in boxes)
                bot = max(b[1] + b[3] for b in boxes)
                k_lo = max(0, int(np.ceil(bot - h)))
                k_hi = min(new_h - h, int(np.floor(top)))
                if k_lo <= k_hi:
                    lo, hi = k_lo, k_hi
            y0 = int(np.random.randint(lo, hi + 1))
            out = scaled[y0:y0 + h]
            shift = -y0
        else:
            off = int(np.random.randint(0, h - new_h + 1))
            out = np.full_like(img, self.pad_val)
            out[off:off + new_h] = scaled
            shift = off

        labels = list(results.get("labels_sonar") or [])
        kept_boxes, kept_labels = [], []
        for i, b in enumerate(boxes):
            y1 = max(b[1] + shift, 0.0)
            y2 = min(b[1] + b[3] + shift, float(h))
            if y2 - y1 >= self.min_box_h:
                kept_boxes.append([b[0], y1, b[2], y2 - y1])
                if i < len(labels):
                    kept_labels.append(labels[i])
        results["img_sonar"] = out
        results["boxes_sonar"] = kept_boxes
        results["labels_sonar"] = kept_labels
        results["sonar_scale_jitter"] = f
        return results
```

**配置**（两份 E08 配置都改）：

```python
train_pipeline = [
    dict(type="AOLoadPaired"),
    dict(type="AOPairedResize",
         camera_size=(640, 640), sonar_size=(640, 640), keep_ratio=False),
    # 必须在 resize 之后：resize 到固定尺寸会抵消先前的行向缩放
    dict(type="AOSonarScaleJitter", scale_range=(0.6, 1.8), prob=1.0),
    dict(type="AOPairedSyncFlip", prob=0.0, allow=True),
    dict(type="AOSampleDegLevel"),
    dict(type="AOStudentCameraView", dinf_mode="zero", jitter=0.08),
    dict(type="AOPackPairedInputs"),
]
```

**新增测试**：必须在**管线层面**断言抖动生效，单测变换本身不够。

```python
def test_scale_jitter_survives_full_train_pipeline():
    """E08 回归测试：经过完整训练管线后，框高必须随 f 变化。"""
    from mmengine.dataset import Compose
    img = np.zeros((441, 128, 3), np.uint8)
    img[200:217, 60:70] = 255
    heights = []
    for f in (0.6, 1.8):
        pipe = Compose([
            dict(type="AOPairedResize", camera_size=None,
                 sonar_size=(640, 640), keep_ratio=False),
            dict(type="AOSonarScaleJitter", scale_range=(f, f)),
        ])
        out = pipe(dict(img_sonar=img.copy(), img_camera=None,
                        boxes_sonar=[[60.0, 200.0, 10.0, 17.0]],
                        labels_sonar=[0], boxes_camera=[]))
        assert out["img_sonar"].shape[:2] == (640, 640)
        heights.append(out["boxes_sonar"][0][3])
    assert heights[1] / heights[0] == pytest.approx(1.8 / 0.6, rel=0.02)
```

**台账**：E08 节"设置"中的"`AOLoadPaired` 之后、`AOPairedResize` 之前"要改为"之后"，并注明本次勘误。预登记标准不变（这次改的是实现 bug，不是标准）。

### 1.3 🟠 B0 与 cross-attn 的声呐监督强度不对等（读码确认）

**位置**：`configs/_base_/models/two_stream_yolox_s.py`，`loss_weights=dict(camera=1.0, fused_sonar=1.0, uncoupled_sonar=1.0)`

**机制**：cross-attn 模型里，融合声呐头和未耦合声呐头的梯度都流回同一个声呐骨干，骨干收到约两倍的声呐检测梯度。B0 只有一个声呐头。Δ_s = AP(cross-attn) − AP(B0) 因此混进了"有效学习率 / 损失权重翻倍"的差异。

**修改**（二选一，推荐 A）：

```python
# A：在 cross-attn 配置中，让两个声呐头的权重之和等于 B0 的单头权重
model = dict(loss_weights=dict(camera=1.0, fused_sonar=0.5,
                               uncoupled_sonar=0.5))
```

```python
# B：B0 也挂两个结构相同的声呐头（都接未耦合特征），总权重对齐
# 代价是 B0 不再是"最简单的单头基线"，需在台账注明
```

无论选哪种，都要在台账 §1 中写明"声呐骨干总监督权重对齐"这一公平性约束，并写进 `tests/test_f2_configs.py`：

```python
def test_sonar_supervision_weight_matched():
    b0 = Config.fromfile("configs/f2/s0_b0_sonar_only_yoloxs.py").model
    ca = Config.fromfile("configs/f2/s0_cross_attn_yoloxs.py").model
    def sonar_w(m):
        w = m.loss_weights
        return (w["uncoupled_sonar"] if m.get("uncoupled_sonar_head") else 0) \
            + (w["fused_sonar"] if m.get("fused_sonar_head") else 0)
    assert sonar_w(b0) == pytest.approx(sonar_w(ca))
```

### 1.4 🟠 训练配方：无预训练、lr 过大、decay/EMA/调度（读码确认）

**现状与问题**：

| 项 | 现状 | 标准（mmdet `yolox_s_8xb8-300e_coco`）/ 方案页 | 问题 |
|---|---|---|---|
| 初始化 | 两条流都从零训练，全库没有 `load_from` 或 `init_cfg` | 方案页 §4.2："都从 mmdet 的 COCO 权重初始化，声呐骨干也一样" | 1680 帧、3 段录音、8.9M/19.9M 参数从零训练；E07 中 train AP50 0.90 而 test 0.005，很难只归因到"尺度锁死" |
| lr | 0.01 @ batch 8 | 0.01 @ batch 64，即每张图 lr 只有现在的 1/8 | 有效 lr 是标准的 8 倍 |
| weight decay | 作用于所有参数 | `paramwise_cfg=dict(norm_decay_mult=0., bias_decay_mult=0.)` | BN 和 bias 被衰减 |
| nesterov | 无 | `nesterov=True` | 次要 |
| EMA | 无 | `ExpMomentumEMA, momentum=0.0001` | 小数据上 EMA 对稳定性影响明显 |
| 调度 | `LinearLR` 0–500 iter（约 2.4 epoch）与从第 1 epoch 开始的 cosine 重叠 | warmup 5 epoch → cosine → 最后 15 epoch 常数 lr | 两个调度器链式相乘 |
| 测试阈值 | `score_thr=0.05`，NMS 0.55 | YOLOX：0.01 / 0.65（E01 已验证） | AP 略低；且与台账 E01 的说法不符 |

**修改**：

```python
# configs/_base_/models/two_stream_yolox_s.py
# 与 E01 验证过的 YOLOX-S COCO 权重保持一致（用本机 E01 实际使用的那份文件；
# 下面的 URL 需在本机核实与 E01 所用权重的哈希一致）
_yolox_s_ckpt = ("https://download.openmmlab.com/mmdetection/v2.0/yolox/"
                 "yolox_s_8x8_300e_coco/"
                 "yolox_s_8x8_300e_coco_20211121_095711-4592a793.pth")

_backbone = dict(
    type="CSPDarknet", deepen_factor=0.33, widen_factor=0.5,
    out_indices=(2, 3, 4),
    init_cfg=dict(type="Pretrained", checkpoint=_yolox_s_ckpt,
                  prefix="backbone."))
_neck = dict(
    type="YOLOXPAFPN", in_channels=[128, 256, 512], out_channels=128,
    num_csp_blocks=1,
    init_cfg=dict(type="Pretrained", checkpoint=_yolox_s_ckpt,
                  prefix="neck."))
# 头保持随机初始化（类别数不同）

model = dict(
    ...,
    test_cfg=dict(nms_pre=1000, min_bbox_size=0, score_thr=0.01,
                  nms=dict(type="nms", iou_threshold=0.65),
                  max_per_img=100),
)
```

注意：
- 声呐流也用 COCO 初始化，这与 CMOLD/CMHD 的做法一致。但应当做一次"声呐从零训练 vs COCO 初始化"的消融，单种子即可，记入台账。
- `init_cfg` 由 `BaseModule.init_weights()` 触发。Runner 会在训练开始时调用；`tools/test.py` 直接 `load_checkpoint` 覆盖，不受影响。
- 修改 `score_thr` 和 NMS 会改变 E07 中 B0 的已有数字。新的 B0 数字要在新编号下重测，不要回改 E07。

```python
# configs/_base_/schedules/schedule_f2.py
max_epochs = 100
num_last_epochs = 15
base_lr = 0.01 * 8 / 64        # YOLOX: 0.01 @ batch 64，按 batch 8 线性缩放

optim_wrapper = dict(
    type="OptimWrapper",
    optimizer=dict(type="SGD", lr=base_lr, momentum=0.9,
                   weight_decay=5e-4, nesterov=True),
    paramwise_cfg=dict(norm_decay_mult=0.0, bias_decay_mult=0.0),
    clip_grad=dict(max_norm=35, norm_type=2),
)

param_scheduler = [
    dict(type="QuadraticWarmupLR", by_epoch=True, begin=0, end=5,
         convert_to_iter_based=True),
    dict(type="CosineAnnealingLR", eta_min=base_lr * 0.05,
         begin=5, T_max=max_epochs - num_last_epochs,
         end=max_epochs - num_last_epochs, by_epoch=True,
         convert_to_iter_based=True),
    dict(type="ConstantLR", by_epoch=True, factor=1,
         begin=max_epochs - num_last_epochs, end=max_epochs),
]

custom_hooks = [
    dict(type="EMAHook", ema_type="ExpMomentumEMA", momentum=0.0001,
         update_buffers=True, priority=49),
]
```

注意：
- **不要**直接加 mmdet 的 `YOLOXModeSwitchHook`。它会访问 `model.bbox_head`，在 `TwoStreamYOLOX` 上会抛 AttributeError。如果需要最后 15 个 epoch 的 L1 损失，写一个自定义 hook，对三个头分别设置 `use_l1=True`。
- 有蒸馏时，EMA 会连教师参数一起复制（教师参数不变，只是浪费显存）。可以在 EMA 的 `ema_model` 构造中排除 `teacher.` 前缀，或者接受这部分开销。
- 修改后，所有方法共用这一份配方（草案 §6.2 的公平性约束不变）。

### 1.5 🟠 D∞ 黑图上仍训练相机检测（读码确认）

**位置**：`configs/_base_/datasets/paired_uxo.py:26`（`dinf_mode="zero"`），`aodet/datasets/transforms/pack_paired.py:AOStudentCameraView`

**机制**：20% 的训练样本被抽到 D∞，相机图被置为全黑，但 `boxes_camera` 原样保留，相机头照常计算检测损失。黑图里没有任何可见信息，相机头只能学到"目标通常在画面中部"这类位置先验，也就是学会幻觉出框。这会污染相机支路的 AP，也会污染 §6.4 的相关性指标：相机的"漏检"和"检出"都不再只取决于图像内容。同理，相机分支的 KD 在 D∞ 样本上也没有意义。

**修改**：D∞ 样本的相机 GT 置空，让它成为一个"诚实负样本"（黑图 → 不出框，这在物理上是对的）。同时记录这个标记，供 KD 跳过。

```python
# pack_paired.py  AOStudentCameraView.transform
def transform(self, results: dict) -> dict:
    img = results.get("img_camera")
    if img is None:
        return results
    level = results.get("deg_level", "d0")
    assert level in ALL_LEVELS
    results["img_camera_clear"] = img.copy()  # teacher view, clean
    if level == "dinf":
        # D∞: nothing reaches the sensor -> no visible camera target.
        # Camera GT is cleared in BOTH modes so the camera head never
        # learns to hallucinate boxes from position priors on black
        # frames; ``camera_valid`` lets the fusion/KD mask this sample.
        results["boxes_camera"] = []
        results["labels_camera"] = []
        results["camera_valid"] = False
        if self.dinf_mode == "drop":
            results["img_camera"] = None
            return results
    else:
        results["camera_valid"] = True
    results["img_camera"] = degrade_camera(
        img, sample_id=results["sample_id"], level=level,
        seed=results.get("deg_seed", 0), jitter=self.jitter)
    return results
```

```python
# pack_paired.py  AOPackPairedInputs.transform —— 把标记带出去
packed["camera_valid"] = bool(results.get("camera_valid", True))
```

```python
# paired_preprocessor.py  forward 中，inputs 构造之后
cv = data.get("camera_valid")
inputs["camera_valid"] = (
    torch.as_tensor([bool(v) for v in cv], device=self.device)
    if cv is not None else None)
```

`camera_valid` 的消费方：融合模块（§2.1）和 KD（§3.3 的 `_kd_losses`，相机分支只在 `camera_valid` 为真的样本上计算）。

**测试**：

```python
def test_dinf_clears_camera_gt():
    t = AOStudentCameraView(dinf_mode="zero")
    out = t.transform(dict(img_camera=np.full((8, 8, 3), 128, np.uint8),
                           boxes_camera=[[1, 1, 3, 3]], labels_camera=[0],
                           deg_level="dinf", sample_id="x"))
    assert out["boxes_camera"] == [] and out["camera_valid"] is False
    assert out["img_camera"].max() == 0
    assert out["img_camera_clear"].max() == 128   # 教师仍看清晰图
```

### 1.6 🟠 单次迭代中融合和头前向两次（读码确认）

**位置**：`two_stream_yolox.py:180`（`loss()` 中的融合）与 `:215`（`_kd_losses()` 中再次融合），以及 `head.loss()` 和 `self.camera_head(cam_feats)` 等各前向一次

**机制**：`head.loss(feats, samples)` 内部会先跑一次 `head(feats)`；`_kd_losses` 里又重新算了一次 `fusion(...)` 和 `head(...)`。交叉注意力在 P3 层是 6400×6400 的规模，这部分开销直接翻倍。数值上两次前向结果一致（没有 dropout），所以这是效率问题，不是正确性问题。但重构后 KD 和检测损失共用同一份 head 输出，逻辑也更清楚。

**修改**：每个头只前向一次，检测损失用 `loss_by_feat`，KD 复用同一份输出。下面的 `loss()` 同时纳入了 §1.5 的 `camera_valid`、§2.1 的融合接口、§2.3 的 stop-grad 选项和 §3.3 的 KD 路由。

```python
# two_stream_yolox.py
from mmdet.models.utils import unpack_gt_instances

def _fuse(self, cam_feats, son_feats, camera_valid=None):
    if self.fusion_detach_sonar:          # §2.3 stop-grad 变体
        son_feats = [s.detach() for s in son_feats]
    return self.fusion(cam_feats, son_feats, camera_valid=camera_valid)

@staticmethod
def _det_loss(head, outs, samples):
    gts, metas, ignores = unpack_gt_instances(samples)
    return head.loss_by_feat(*outs, gts, metas, ignores)

def loss(self, batch_inputs, batch_data_samples):
    losses = {}
    cam_imgs = batch_inputs.get("camera")
    son_imgs = batch_inputs["sonar"]
    cam_valid = batch_inputs.get("camera_valid")

    son_feats = self.extract_sonar_feat(son_imgs)
    cam_feats = (self.extract_camera_feat(cam_imgs)
                 if cam_imgs is not None else None)

    # 每个头只前向一次；检测损失和 KD 共用 outs
    outs = {}
    if self.camera_head is not None and cam_feats is not None:
        outs["camera"] = self.camera_head(cam_feats)
    if self.uncoupled_sonar_head is not None:
        outs["sonar_uncoupled"] = self.uncoupled_sonar_head(son_feats)
    if self.fused_sonar_head is not None:
        outs["sonar"] = self.fused_sonar_head(
            self._fuse(cam_feats, son_feats, cam_valid))

    side = {"camera": "camera", "sonar": "sonar",
            "sonar_uncoupled": "sonar"}
    wkey = {"camera": "camera", "sonar": "fused_sonar",
            "sonar_uncoupled": "uncoupled_sonar"}
    head_of = {"camera": self.camera_head, "sonar": self.fused_sonar_head,
               "sonar_uncoupled": self.uncoupled_sonar_head}
    for name, o in outs.items():
        hl = self._det_loss(head_of[name], o,
                            _unpack(batch_data_samples, side[name]))
        w = self.loss_weights.get(wkey[name], 1.0)
        for k, v in hl.items():
            losses[f"{wkey[name]}/{k}"] = v * w

    if self.teacher is not None and self.training:
        losses.update(self._kd_losses(batch_inputs, outs,
                                      son_feats, cam_valid))
    return losses
```

**测试**：在同一个 batch 上，断言重构前后（固定种子、关闭 dropout）各检测损失项逐项相等，容差 1e-6。

### 1.7 🟡 退化档位抽样的随机性（读码确认）

**位置**：`pack_paired.py:AOSampleDegLevel.__init__`（`np.random.default_rng(seed)`，seed 默认为 0）和 `transform`（`deg_seed` 取自 frame_meta，恒为 0）

**机制**：
- transform 对象在主进程构造，带着同一个 rng 状态被复制进 4 个 worker，4 个 worker 抽出完全相同的档位序列。
- `tools/train.py --seed 1/2` 不会改变这个序列，所以 3 个种子的方差里不包含"退化档位抽样"这一来源。
- `deg_seed` 恒为 0，于是每张图在每个档位上只有一种固定的退化样本，100 个 epoch 反复看同一份。草案 §6.1 只要求**测试集**确定性；训练集应当在线随机。

**修改**：

```python
@TRANSFORMS.register_module()
class AOSampleDegLevel(BaseTransform):
    def __init__(self, probs: Optional[dict] = None):
        super().__init__()
        p = dict(self._DEFAULT_PROBS)
        if probs:
            p.update(probs)
        total = sum(p.values())
        assert abs(total - 1.0) < 1e-6, f"probs must sum to 1, got {total}"
        self.levels = list(p)
        self.probs = np.asarray([p[k] for k in self.levels])
        self._rng = None   # 在 worker 内惰性创建

    def transform(self, results: dict) -> dict:
        if self._rng is None:
            # mmengine 的 worker_init_fn 已按 (run seed, rank, worker_id)
            # 设置了全局 np.random —— 由它派生，使 worker 间、训练种子间
            # 的档位序列都不同，且在给定 run seed 下可复现
            self._rng = np.random.default_rng(np.random.randint(2 ** 31))
        results["deg_level"] = str(self._rng.choice(self.levels, p=self.probs))
        # 训练期每次抽样一个新的退化参数种子（测试集仍走 frame_meta 的固定种子）
        results["deg_seed"] = int(self._rng.integers(2 ** 31))
        return results
```

注意：去掉了 `seed` 参数。如果现有配置显式传了 `seed=`，需要同步删除。台账 §1 中"种子"一条补充说明：训练种子同时控制退化抽样。

---

## 2. 模型架构

### 2.1 🔴 CrossAttn 的 D∞ 训练/测试路径不一致（读码确认）

**位置**：`aodet/models/fusion/cross_attn.py:56`（D∞ 回退）与 `:67`（正常路径）；`configs/_base_/datasets/paired_uxo.py:26`（训练用 zero）与 `:37`（测试用 drop）

**机制**：

| | 训练（`dinf_mode="zero"`） | 测试（`dinf_mode="drop"`, batch=1） |
|---|---|---|
| 相机输入 | 全黑图，照常经过相机流 | `None` |
| 融合输出 | `LayerNorm(q + Attn(q, cam_black))` | `proj(sf)`，**不经过 LayerNorm，也没有注意力项** |

融合头在测试的 D∞ 档看到的是训练中从未出现过的特征分布：没有归一化，尺度完全不同。融合声呐头在 D∞ 的掉分会有一部分是这个 bug 造成的，而它恰好会伪装成 RQ-0 要找的"互毒害"。

**修改**：见 §2.2 的新融合模块。核心是两点：
1. 输出改成 `fused = q + γ·Attn(...)`，不再对整个残差做 LayerNorm，于是"没有相机"严格等价于"注意力项为 0"，即 `fused = q`。
2. 训练时用 `camera_valid` 把 D∞ 样本的注意力项乘 0，不再依赖黑图。训练和测试走的是同一个函数：测试 drop 时 `cam_feats is None` → 返回 `q`；训练 zero 时 `camera_valid=False` → 注意力项乘 0 → 同样返回 `q`。

**测试**（关键回归测试）：

```python
def test_cross_attn_dinf_train_test_consistent():
    """D∞：训练（mask）与测试（drop）两条路径输出必须完全一致。"""
    f = CrossAttnFusion(cam_channels=[16], sonar_channels=[16],
                        embed_dim=16, num_heads=2, layer_scale_init=0.5)
    son = [torch.randn(2, 16, 8, 8)]
    cam = [torch.randn(2, 16, 8, 8)]
    masked = f(cam, son, camera_valid=torch.tensor([False, False]))
    dropped = f(None, son)
    torch.testing.assert_close(masked[0], dropped[0])
```

### 2.2 🔴 交叉注意力没有位置信息，也没有几何先验（读码确认）

**位置**：`aodet/models/fusion/cross_attn.py`

**问题**：
1. **没有位置编码**。q（声呐）和 kv（相机）都是纯内容 token。注意力的输出与相机 token 的排列顺序无关，每个声呐位置只能拿到一个"相机全局内容袋"的加权和，学不到"相机第 j 列 ↔ 声呐方位 θ"这种对应关系。这正是草案张力 4 的核心，而现在的结构从原理上就表达不了。
2. **与要研究的机制不对应**。RSFusionDet 的 CAFusion 是带参考点的可变形交叉注意力。现在的全局无位置注意力测出的"互毒害"不能外推到 CAFusion 类方法，RQ-0 的结论层级需要据此限定。
3. **代价**。P3 层是 80×80 = 6400 个 token，6400×6400 的全局注意力既贵又没有必要：相机里一个目标只对应声呐里一条方位窗。
4. **初始化**。`LayerNorm(q + attended)`，注意力项从训练一开始就以满幅度注入随机的相机相关噪声，没有门控。

**修改方案 A（最小改动，推荐先做）**：在现有模块上加 2D 正弦位置编码、pre-norm 和 LayerScale γ（零初始化），并支持 `camera_valid` 掩码。

```python
# aodet/models/fusion/cross_attn.py —— 替换 CrossAttnFusion
def _sine_pe_2d(h: int, w: int, dim: int, device, dtype) -> Tensor:
    """(h*w, dim) 2D sine positional encoding (DETR-style)."""
    assert dim % 4 == 0
    d = dim // 4
    omega = 1.0 / (10000 ** (torch.arange(d, device=device,
                                          dtype=torch.float32) / d))
    ys = torch.arange(h, device=device, dtype=torch.float32)[:, None] / h
    xs = torch.arange(w, device=device, dtype=torch.float32)[:, None] / w
    py = torch.cat([torch.sin(ys * omega * 2 * math.pi),
                    torch.cos(ys * omega * 2 * math.pi)], 1)   # h, 2d
    px = torch.cat([torch.sin(xs * omega * 2 * math.pi),
                    torch.cos(xs * omega * 2 * math.pi)], 1)   # w, 2d
    pe = torch.cat([py[:, None, :].expand(h, w, 2 * d),
                    px[None, :, :].expand(h, w, 2 * d)], -1)
    return pe.reshape(h * w, dim).to(dtype)


class CrossAttnFusion(nn.Module):
    """fused_l = q_l + γ_l · Attn(LN(q_l)+PE_s, LN(c_l)+PE_c)

    * γ (per-channel LayerScale) is zero-initialised by default: the fused
      head starts identical to a sonar-only head and coupling is learned.
    * No LayerNorm on the residual output, so "no camera" is exactly
      ``fused = q`` — the D∞ drop path (cam_feats None) and the masked
      training path (camera_valid False) are the same function (§2.1).
    """

    def __init__(self, cam_channels, sonar_channels, embed_dim=128,
                 num_heads=4, dropout=0.0, layer_scale_init=0.0,
                 pos_embed=True):
        super().__init__()
        assert len(cam_channels) == len(sonar_channels)
        self.pos_embed = pos_embed
        self.cam_proj = nn.ModuleList(
            [nn.Conv2d(c, embed_dim, 1) for c in cam_channels])
        self.sonar_proj = nn.ModuleList(
            [nn.Conv2d(c, embed_dim, 1) for c in sonar_channels])
        self.norm_q = nn.ModuleList(
            [nn.LayerNorm(embed_dim) for _ in cam_channels])
        self.norm_kv = nn.ModuleList(
            [nn.LayerNorm(embed_dim) for _ in cam_channels])
        self.attn = nn.ModuleList([
            nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout,
                                  batch_first=True)
            for _ in cam_channels])
        self.gamma = nn.ParameterList([
            nn.Parameter(torch.full((embed_dim,), float(layer_scale_init)))
            for _ in cam_channels])

    def forward(self, cam_feats, son_feats, camera_valid=None):
        qs = [proj(sf) for proj, sf in zip(self.sonar_proj, son_feats)]
        if cam_feats is None:
            return qs                                  # 与 mask 路径同一函数
        out = []
        for i, (cf, q) in enumerate(zip(cam_feats, qs)):
            b, e, h, w = q.shape
            kv = self.cam_proj[i](cf)
            hc, wc = kv.shape[-2:]
            q_tok = q.flatten(2).transpose(1, 2)        # B, HsWs, E
            kv_tok = kv.flatten(2).transpose(1, 2)      # B, HcWc, E
            qn, kn = self.norm_q[i](q_tok), self.norm_kv[i](kv_tok)
            if self.pos_embed:
                qn = qn + _sine_pe_2d(h, w, e, q.device, q.dtype)
                kn = kn + _sine_pe_2d(hc, wc, e, q.device, q.dtype)
            att, _ = self.attn[i](qn, kn, kv_tok, need_weights=False)
            att = att * self.gamma[i]
            if camera_valid is not None:
                att = att * camera_valid.to(att.dtype)[:, None, None]
            out.append((q_tok + att).transpose(1, 2).reshape(b, e, h, w))
        return out
```

要点：
- 不再把相机特征插值到声呐网格。两者不配准，插值没有物理意义，注意力本身不要求空间尺寸相同。
- `layer_scale_init=0.0` 是 ReZero 式初始化：γ 初始为 0，但 γ 自身的梯度不为零，训练可以正常启动。如果收敛慢，可以改用 1e-3。
- `NoneFusion` 和 `ConcatFusion` 的 `forward` 也要加上 `camera_valid=None` 参数，以保持接口一致。
- 这个方案仍然是 O((HW)²)。如果显存紧张，只在 P4 和 P5 层做融合，P3 直接透传 q。

**修改方案 B（推荐作为主方法的融合结构）**：方位轴交叉注意力 + 几何偏置。

思路：单目相机不给距离，几何上站得住的对应只有"相机列 → 声呐方位"。所以先把相机特征沿竖直方向做注意力池化，得到每列一个 token 的一维方位序列；声呐每个位置只按方位去查这个序列，再加一个由标定得到的方位距离偏置。复杂度从 (HsWs)·(HcWc) 降到 (HsWs)·Wc。这与 CramNet 的射线约束注意力属于同一类几何，可以写进 RQ-3 的叙事。

```python
# aodet/models/fusion/bearing_attn.py（新文件）
class BearingAxialFusion(nn.Module):
    """Sonar positions attend to camera COLUMNS (range unknown).

    camera (B,C,Hc,Wc) --attention-pool over H--> (B,Wc,E) column tokens
    sonar  (B,C,Hs,Ws) -> queries (B,HsWs,E)
    logits += bias[sonar col, camera col]
      bias = -(θ_s(col) - θ_c(col'))² / (2σ²)   (geometry prior, σ learnable)
      or a learned relative bias table when no calibration is available.
    fused = q + γ·Attn(...)      (γ zero-init; camera_valid masks D∞)
    """

    def __init__(self, cam_channels, sonar_channels, embed_dim=128,
                 num_heads=4, layer_scale_init=0.0,
                 sonar_col_bearing=None, camera_col_bearing=None,
                 sigma_deg_init=3.0):
        super().__init__()
        n = len(cam_channels)
        self.num_heads = num_heads
        self.cam_proj = nn.ModuleList(
            [nn.Conv2d(c, embed_dim, 1) for c in cam_channels])
        self.cam_pool = nn.ModuleList(
            [nn.Conv2d(embed_dim, 1, 1) for _ in cam_channels])
        self.sonar_proj = nn.ModuleList(
            [nn.Conv2d(c, embed_dim, 1) for c in sonar_channels])
        self.norm_q = nn.ModuleList([nn.LayerNorm(embed_dim) for _ in range(n)])
        self.norm_kv = nn.ModuleList([nn.LayerNorm(embed_dim) for _ in range(n)])
        self.attn = nn.ModuleList([
            nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
            for _ in range(n)])
        self.gamma = nn.ParameterList([
            nn.Parameter(torch.full((embed_dim,), float(layer_scale_init)))
            for _ in range(n)])
        # bearing tables in degrees at INPUT resolution (callables or None):
        # sonar_col_bearing(x_norm) / camera_col_bearing(x_norm)
        self.sonar_col_bearing = sonar_col_bearing
        self.camera_col_bearing = camera_col_bearing
        self.log_sigma = nn.Parameter(torch.tensor(math.log(sigma_deg_init)))

    def _bias(self, ws, wc, hs, device):
        if self.sonar_col_bearing is None or self.camera_col_bearing is None:
            return None
        xs = (torch.arange(ws, device=device) + 0.5) / ws
        xc = (torch.arange(wc, device=device) + 0.5) / wc
        ts = torch.as_tensor(self.sonar_col_bearing(xs.cpu().numpy()),
                             device=device, dtype=torch.float32)
        tc = torch.as_tensor(self.camera_col_bearing(xc.cpu().numpy()),
                             device=device, dtype=torch.float32)
        d2 = (ts[:, None] - tc[None, :]) ** 2            # Ws, Wc
        bias = -d2 / (2 * self.log_sigma.exp() ** 2)
        return bias[None].expand(hs, ws, wc).reshape(hs * ws, wc)

    def forward(self, cam_feats, son_feats, camera_valid=None):
        qs = [p(s) for p, s in zip(self.sonar_proj, son_feats)]
        if cam_feats is None:
            return qs
        out = []
        for i, (cf, q) in enumerate(zip(cam_feats, qs)):
            b, e, hs, ws = q.shape
            kv = self.cam_proj[i](cf)                          # B,E,Hc,Wc
            a = self.cam_pool[i](kv).softmax(dim=2)            # over H
            cols = (kv * a).sum(2).transpose(1, 2)             # B,Wc,E
            q_tok = q.flatten(2).transpose(1, 2)               # B,HsWs,E
            bias = self._bias(ws, cols.shape[1], hs, q.device)
            mask = None
            if bias is not None:
                # MultiheadAttention 的 float attn_mask 会加到 logits 上
                mask = bias.repeat(self.num_heads, 1, 1)       # heads,L,S
                mask = mask.repeat(b, 1, 1)                    # B*heads,L,S
            att, _ = self.attn[i](self.norm_q[i](q_tok),
                                  self.norm_kv[i](cols), cols,
                                  attn_mask=mask, need_weights=False)
            att = att * self.gamma[i]
            if camera_valid is not None:
                att = att * camera_valid.to(att.dtype)[:, None, None]
            out.append((q_tok + att).transpose(1, 2).reshape(b, e, hs, ws))
        return out
```

注意：
- 方位表来源：声呐侧用 `aodet/geometry/aris_beam_table.py` 的 `beam_center_deg`（UXO/ARIS），相机侧用标定内参，按针孔模型 `atan((x − cx)/fx)` 计算，不能用线性近似。UXO 每段录音的相机方位是分别标定的，所以偏置表需要**按录音**提供（通过 frame_meta 的 `recording` 字段查表）。这一点要在 `PairedDataPreprocessor` 中把录音 id 带进来，作为第二步实现。
- 第一步可以先不传方位表（`bias=None`），只用"方位轴池化 + 注意力"，与方案 A 对比；再加几何偏置，作为 RQ-3 的一个消融点。
- 近场视差（见 §3.6）会让方位对应随距离变化。σ 可学习，可以部分吸收这种变化；严格做法见 §3.6 的"射线曲线"方案。

**配置**（新增，不覆盖 S0-lite 的原配置，作为新编号实验）：

```python
# configs/f2/s0_bearing_attn_yoloxs.py
_base_ = ["s0_cross_attn_yoloxs.py"]
model = dict(fusion=dict(
    _delete_=True, type="BearingAxialFusion",
    cam_channels=[128, 128, 128], sonar_channels=[128, 128, 128],
    embed_dim=128, num_heads=4, layer_scale_init=0.0))
```

**台账影响**：S0-lite（E07）的 cross-attn 是"无位置编码的全局注意力"。在 §5.3 阶段结论里，RQ-0 的结论应限定为"该结构下"，不能外推到 CAFusion。

### 2.3 🟠 未耦合声呐头"推理解耦、训练耦合"（读码确认）

**机制**：未耦合头和融合头共用声呐骨干与 neck。融合头的梯度经过融合模块回流到声呐骨干，相机信息会在训练期写进声呐特征。所以未耦合头在推理时不看相机，但它的特征是和相机共同训练出来的。

这本身不一定是 bug（草案 §5.1 只要求推理时不耦合），但 RQ-4 的独立性读数必须区分以下三种情况：

| 变体 | 推理是否看相机 | 训练期声呐骨干是否受融合梯度影响 | 用途 |
|---|---|---|---|
| 未耦合头（现状） | 否 | 是 | 草案的"分开记账" |
| 未耦合头 + stop-grad | 否 | 否 | 拆开"训练期耦合"的贡献 |
| B5 晚融合 | 否 | 否（两个独立模型） | 结构独立的参照下限 |

**修改**：在检测器里加一个开关（`_fuse` 的实现见 §1.6）。

```python
# two_stream_yolox.py  __init__ 新增参数
fusion_detach_sonar: bool = False,
...
self.fusion_detach_sonar = fusion_detach_sonar
```

```python
# configs/f2/s0_cross_attn_detach_yoloxs.py
_base_ = ["s0_cross_attn_yoloxs.py"]
model = dict(fusion_detach_sonar=True)
```

`predict()` 中的融合也要改为调用 `self._fuse(...)`，并传 `camera_valid`（测试时 batch=1，drop 后 `cam_feats is None`，传 None 即可）。

### 2.4 🟠 ConcatFusion 默认两平面已配准（读码确认）

**位置**：`aodet/models/fusion/concat.py:45`

**机制**：把相机特征双线性插值到声呐网格，再逐像素拼接，意味着"相机 (i,j) 与声呐 (i,j) 指向同一场景位置"。这正是草案张力 4 指出的 CMHD 未验证假设。用它做"简单拼接"基线时，结果混入了一个已知错误的几何假设。

**修改**：把拼接改为"方位对齐的拼接"。相机特征先沿竖直方向池化成列向量，按方位重采样到声呐的列，再沿距离方向广播；没有标定时退化为全局池化后广播。

```python
# concat.py  forward 中替换插值那段
cf_cols = cf.mean(dim=2, keepdim=True)                    # B,C,1,Wc
if self.bearing_resample is not None:
    # bearing_resample: (B,C,1,Wc) -> (B,C,1,Ws)，按方位表插值
    cf_cols = self.bearing_resample(cf_cols, sf.shape[-1])
else:
    cf_cols = cf_cols.mean(dim=3, keepdim=True)          # 全局，无几何
cf_b = cf_cols.expand(-1, -1, sf.shape[-2], sf.shape[-1])
```

`ConcatFusion` 也要支持 `camera_valid`（D∞ 时相机分量置零），否则它无法参与模态丢弃实验（现在 `cam_feats is None` 直接抛异常）。

---

## 3. 蒸馏方法与损失函数

### 3.1 🔴 objectness KD：全图 MSE 被背景稀释（读码确认）

**位置**：`aodet/models/losses/branch_kd.py:42`

```python
loss = loss + F.mse_loss(s.sigmoid(), t.sigmoid())   # 对所有位置取均值
```

**问题**：
1. **前景被稀释**：640 输入下三层共约 8400 个位置，UXO 每帧只有一个目标，前景位置占比不到 1%。全图 MSE 的均值几乎全部由背景（σ ≈ 0 对 σ ≈ 0）贡献，量级约 1e-3 或更小。
2. **量级不匹配**：mmdet YOLOX 的 obj 损失是"对所有位置求 BCE 之和，再除以前景数 num_fg"，量级在 1 左右。KD 权重设成 1.0 时，KD 项比检测损失小两到三个数量级，相当于没开。
3. **梯度形状**：先 sigmoid 再算 MSE，在饱和区（σ 接近 0 或 1）梯度含 σ(1−σ) 因子，几乎为零；而前景和背景的高置信位置恰好都在饱和区。

这也偏离了 CMOLD "目标级蒸馏"的原意（草案 §5.3 引用的 0.794 就是目标级的结果）。

**修改**：软标签 BCE（在 logit 上计算），按教师前景概率加权，再按权重之和归一化。

```python
# branch_kd.py
def objectness_kd(s_obj, t_obj, bg_floor: float = 0.01,
                  valid: Optional[Tensor] = None) -> Tensor:
    """Foreground-weighted soft-target BCE on objectness logits.

    w = max(σ(t), bg_floor): teacher-confident locations dominate, the
    background keeps a small floor so false positives are still pushed
    down.  Normalised by Σw (≈ number of teacher foreground cells), the
    same scale as YOLOX's own obj loss (sum / num_fg).
    ``valid`` (B,) bool masks whole samples (e.g. camera KD at D∞).
    """
    n = min(len(s_obj), len(t_obj))
    num = s_obj[0].new_zeros(())
    den = s_obj[0].new_zeros(())
    for s, t in zip(s_obj[:n], t_obj[:n]):
        if s.shape[-2:] != t.shape[-2:]:
            s = F.interpolate(s, size=t.shape[-2:], mode="bilinear",
                              align_corners=False)
        p_t = t.sigmoid().detach()
        w = p_t.clamp(min=bg_floor)
        if valid is not None:
            w = w * valid.to(w.dtype)[:, None, None, None]
        l = F.binary_cross_entropy_with_logits(s, p_t, reduction="none")
        num = num + (l * w).sum()
        den = den + w.sum()
    return num / den.clamp(min=1.0)
```

### 3.2 🔴 类别 KD：softmax-KL 与 YOLOX 不匹配（读码确认）

**位置**：`aodet/models/losses/branch_kd.py:59`

**问题**：
1. YOLOX 的类别分支是**逐类 sigmoid**（多标签 BCE），而且只在 SimOTA 正样本上训练，目标是 IoU 感知的软标签。背景位置上的类别 logit 从来没有被监督过，没有语义。现在的做法是在**所有位置**上做**softmax** KL，等于在蒸馏噪声，还强行引入了类间互斥的假设。
2. 文档错误：docstring 写的是 KL(student‖teacher)。`F.kl_div(input=log q_s, target=log p_t, log_target=True)` 实际计算的是 KL(teacher‖student)。后者才是标准蒸馏方向，只需改 docstring。

**修改**：蒸馏**检测分数** σ(cls)·σ(obj)，也就是 YOLOX 推理时真正用来排序的量，用逐类 BCE，只在教师置信的位置上计算。

```python
def score_kd(s_cls, s_obj, t_cls, t_obj, bg_floor: float = 0.01,
             valid: Optional[Tensor] = None, eps: float = 1e-6) -> Tensor:
    """Distil the detection score σ(cls)·σ(obj) (YOLOX's ranking score).

    Per-class BCE against the teacher's score; weighted by the teacher's
    max class score so background logits (never supervised in YOLOX)
    contribute only through the small floor.  fp32 only (BCE on
    probabilities is unsafe under autocast).
    """
    n = min(len(s_cls), len(t_cls))
    num = s_cls[0].new_zeros(())
    den = s_cls[0].new_zeros(())
    for sc, so, tc, to in zip(s_cls[:n], s_obj[:n], t_cls[:n], t_obj[:n]):
        if sc.shape[-2:] != tc.shape[-2:]:
            sz = tc.shape[-2:]
            sc = F.interpolate(sc, size=sz, mode="bilinear",
                               align_corners=False)
            so = F.interpolate(so, size=sz, mode="bilinear",
                               align_corners=False)
        t_score = (tc.sigmoid() * to.sigmoid()).detach()      # B,C,H,W
        s_score = (sc.float().sigmoid() * so.float().sigmoid()).clamp(
            eps, 1 - eps)
        w = t_score.max(1, keepdim=True).values.clamp(min=bg_floor)
        if valid is not None:
            w = w * valid.to(w.dtype)[:, None, None, None]
        l = F.binary_cross_entropy(s_score, t_score, reduction="none")
        num = num + (l * w).sum()
        den = den + w.sum() * sc.shape[1]
    return num / den.clamp(min=1.0)
```

`branch_kd_losses` 相应改为：

```python
def branch_kd_losses(student_out, teacher_out, cfg=None, valid=None):
    """valid: {branch: (B,) bool or None}"""
    cfg = cfg or {}
    obj_w = float(cfg.get("obj_weight", 1.0))
    score_w = float(cfg.get("score_weight", 0.0))
    bg = float(cfg.get("bg_floor", 0.01))
    valid = valid or {}
    losses = {}
    for s_name, t_name in cfg.get("routes", {"camera": "camera",
                                             "sonar": "sonar"}).items():
        if s_name not in student_out or t_name not in teacher_out:
            continue
        s_cls, _, s_obj = student_out[s_name]
        t_cls, _, t_obj = teacher_out[t_name]
        v = valid.get(s_name)
        if obj_w > 0:
            losses[f"{s_name}_obj"] = obj_w * objectness_kd(
                s_obj, t_obj, bg_floor=bg, valid=v)
        if score_w > 0:
            losses[f"{s_name}_score"] = score_w * score_kd(
                s_cls, s_obj, t_cls, t_obj, bg_floor=bg, valid=v)
    if not losses:
        raise RuntimeError(
            f"no KD route matched: student={list(student_out)} "
            f"teacher={list(teacher_out)} routes={cfg.get('routes')}")
    return losses
```

**权重标定**：修复 #1 之后，需要先在验证协议（§4.3）上标定 KD 权重：
- 训练前 1–2 个 epoch，记录 KD 项与对应检测损失项的比值，初始目标是 KD 约占检测损失的 10–30%。
- 再在 {0.5, 1, 2}× 上做一轮扫描。选定后冻结，写入台账，所有 M 系列共用。

**测试**：

```python
def test_objectness_kd_fg_dominates():
    t = [torch.full((1, 1, 20, 20), -8.0)]; t[0][0, 0, 10, 10] = 8.0
    s_bad_fg = [torch.full((1, 1, 20, 20), -8.0)]          # 漏掉前景
    s_bad_bg = [t[0].clone()]; s_bad_bg[0][0, 0, 0, 0] = 8.0  # 多一个虚警
    assert objectness_kd(s_bad_fg, t) > 10 * objectness_kd(s_bad_bg, t)

def test_objectness_kd_scale_comparable_to_det_loss():
    # 单前景时 loss 应在 O(1)，而不是 O(1e-3)
    t = [torch.full((1, 1, 80, 80), -8.0)]; t[0][0, 0, 40, 40] = 8.0
    s = [torch.zeros(1, 1, 80, 80)]
    assert objectness_kd(s, t) > 0.1
```

### 3.3 🟠 未耦合声呐头接收不到 KD；缺少特征级蒸馏（读码确认）

**位置**：`two_stream_yolox.py:214–218`

```python
if self.fused_sonar_head is not None and t_out.get("sonar") is not None:
    ...  s_out["sonar"] = self.fused_sonar_head(fused)
elif self.uncoupled_sonar_head is not None and t_out.get("sonar") is not None:
    s_out["sonar"] = self.uncoupled_sonar_head(son_feats)
```

**问题**：
1. 只要有融合头，未耦合头就收不到 KD。草案 §5.3 明确要求未耦合头"训练时同样接受蒸馏"。
2. 更关键的是，Q5 的机制假设是"特权信息被写进**声呐表示本身**"。作用在融合头上的 KD，学生完全可以通过自己的（退化）相机通路去拟合，不必改进声呐特征。只有未耦合头上的 KD 增益，才能干净地读成"声呐被增强了"，这正是 RQ-2 最需要的读数。
3. 只蒸馏头输出，监督信号太稀疏。头输出层面的蒸馏很容易被融合模块"吸收"掉。

**修改 1：KD 路由显式化**（`routes` 见 §3.2 的 `branch_kd_losses`）。

```python
# 学生分支 -> 教师分支
distillation = dict(
    ...,
    routes={
        "camera": "camera",               # L_KD^c
        "sonar": "sonar",                 # 融合头 <- 教师融合头
        "sonar_uncoupled": "sonar",       # 未耦合头 <- 教师融合头（特权）
    },
)
```

```python
# two_stream_yolox.py  _kd_losses（复用 §1.6 的 outs，不再重复前向）
def _kd_losses(self, batch_inputs, outs, son_feats, cam_valid):
    from aodet.models.losses.branch_kd import (branch_kd_losses,
                                                feature_kd_loss)
    cfg = self.distillation_cfg
    clear = batch_inputs.get("camera_clear")
    t_in = clear if (cfg.get("teacher_input", "clear") == "clear"
                     and clear is not None) else batch_inputs.get("camera")
    with torch.no_grad():
        t_out, t_feats = self.teacher.forward_head_outputs(
            camera=t_in, sonar=batch_inputs["sonar"], return_feats=True)
    valid = {}
    if cam_valid is not None:
        valid["camera"] = cam_valid       # D∞ 样本不做相机 KD
    kd = branch_kd_losses(outs, t_out, cfg=cfg, valid=valid)
    losses = {f"kd/loss_{k}": v for k, v in kd.items()}
    fw = float(cfg.get("feat_weight", 0.0))
    if fw > 0 and t_feats.get("sonar_fused") is not None:
        losses["kd/loss_sonar_feat"] = fw * feature_kd_loss(
            son_feats, t_feats["sonar_fused"], self.kd_adapters,
            fg=[o.sigmoid() for o in t_out["sonar"][2]])
    return losses
```

**修改 2：特征级 KD**（教师融合声呐特征 → 学生**未耦合**声呐 neck，前景加权，FGD 式简化版）。

```python
# branch_kd.py
def feature_kd_loss(s_feats, t_feats, adapters, fg=None,
                    bg_floor: float = 0.05) -> Tensor:
    """Masked MSE between adapter(student sonar neck) and teacher fused
    sonar features, per level.  ``fg``: per-level teacher objectness
    probabilities (B,1,H,W) used as the spatial weight."""
    loss = s_feats[0].new_zeros(())
    for i, (s, t) in enumerate(zip(s_feats, t_feats)):
        s = adapters[i](s)
        if s.shape[-2:] != t.shape[-2:]:
            s = F.interpolate(s, size=t.shape[-2:], mode="bilinear",
                              align_corners=False)
        w = (fg[i].detach().clamp(min=bg_floor) if fg is not None
             else torch.ones_like(t[:, :1]))
        l = ((s - t.detach()) ** 2).mean(1, keepdim=True)
        loss = loss + (l * w).sum() / w.sum().clamp(min=1.0)
    return loss / len(s_feats)
```

```python
# two_stream_yolox.py  _init_teacher 之后
if distillation and distillation.get("feat_weight", 0) > 0:
    s_ch = distillation["student_feat_channels"]     # e.g. [128,128,128]
    t_ch = distillation["teacher_feat_channels"]     # e.g. [256,256,256] (YOLOX-L)
    self.kd_adapters = nn.ModuleList(
        [nn.Conv2d(s, t, 1) for s, t in zip(s_ch, t_ch)])
```

`FrozenTeacher.forward_head_outputs` 需要相应扩展：返回 `(out, feats)`，其中 `feats = {"sonar_fused": fused, "sonar_raw": son_feats}`；`out` 在教师有未耦合头时额外包含 `"sonar_uncoupled"`（供 §3.4 使用）。

注意：教师融合特征的通道数是 `embed_dim`（融合模块输出），学生 neck 是 128，所以 adapter 是必需的。adapter 只在训练时存在，推理不使用，不计入部署参数。

### 3.4 🟠 可达性代理量的循环论证风险（设计问题，§5.4 未实现）

**问题**：草案 §5.4 用"教师在**声呐头**上的 objectness"作为"目标在声呐里可见"的代理。但如果这里取的是教师的**融合**声呐头，它已经看过清晰相机，测到的主要是相机可见性，代理量就失效了。而现在的 `FrozenTeacher` 恰好优先取融合头（`head = model.fused_sonar_head or model.uncoupled_sonar_head`）。

**修改**：
- 教师必须保留未耦合声呐头，可达性权重**只能**来自教师的未耦合头：`w = σ(k·(p_s^{T,uncoupled} − θ_w))`。
- 草案 §5.4 的写法需要相应更正为"教师**未耦合**声呐头的 objectness"。
- 按草案原则，启用前先在验证集上检验这个代理量与"B0 单声呐模型实际召回"的相关性。

```python
# frozen_teacher.py  forward_head_outputs 中
if model.uncoupled_sonar_head is not None:
    out["sonar_uncoupled"] = model.uncoupled_sonar_head(son_feats)
if model.fused_sonar_head is not None:
    fused = model.fusion(cam_feats, son_feats)
    out["sonar"] = model.fused_sonar_head(fused)
    feats["sonar_fused"] = fused
elif model.uncoupled_sonar_head is not None:
    out["sonar"] = out["sonar_uncoupled"]
```

```python
# 可达性权重（§5.4），作用在 sonar 路由的 objectness_kd / score_kd 上
def reachability_weight(t_uncoupled_obj_logits, k=10.0, theta=0.3):
    return [torch.sigmoid(k * (o.sigmoid() - theta)).detach()
            for o in t_uncoupled_obj_logits]
# objectness_kd 中：w = p_t.clamp(min=bg_floor) * reach[i]
```

### 3.5 🟠 `FrozenTeacher` 宽松加载（读码确认）

**位置**：`aodet/models/teachers/frozen_teacher.py:48`

```python
self.model.load_state_dict(state.get("state_dict", state), strict=False)
```

**问题**：教师 checkpoint 可能来自一次本身带教师的训练（键名带 `teacher.` 前缀），或者来自 EMA，或者键名有其他差异。`strict=False` 时，只要键对不上，教师就会**静默地保持随机权重**。蒸馏照样在跑，只是在蒸馏一个随机网络。

**修改**：

```python
if ckpt:
    from mmengine.runner import CheckpointLoader
    state = CheckpointLoader.load_checkpoint(ckpt, map_location="cpu")
    sd = state.get("state_dict", state)
    # 教师自己的 checkpoint 若来自带蒸馏的训练，去掉其嵌套教师与 KD adapter
    sd = {k: v for k, v in sd.items()
          if not k.startswith(("teacher.", "kd_adapters."))}
    missing, unexpected = self.model.load_state_dict(sd, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"teacher checkpoint mismatch: missing={missing[:10]} "
            f"unexpected={unexpected[:10]} (total {len(missing)}/"
            f"{len(unexpected)})")
```

**测试**：用一个随机初始化的 `TwoStreamYOLOX` 保存 checkpoint，再加载到 `FrozenTeacher`，断言参数逐个相等；再构造一个键名带多余前缀的 checkpoint，断言抛出 `RuntimeError`。

另外，方案页 §4.4(4) 写的是教师前向用 fp16 + no_grad，现在是 fp32。显存吃紧时再加 `torch.autocast`，不急。

### 3.6 🟠 L_xp（跨平面 MIL）的问题（读码确认）

**位置**：`aodet/models/losses/cross_plane_mil.py`；`aodet/geometry/m0.py`

**问题与修改**：

**(a) 不是单边约束**（`:124`）

```python
F.binary_cross_entropy(p_max, p_t)
```

BCE 的目标是 p_t。当 p_max > p_t 时，它会把声呐峰值往**下**拉。例如声呐回波很清楚（0.95），相机教师只有 0.7，声呐就会被压到 0.7。这就是在传递负证据，违背草案 §5.3"只传正证据"的原则。

修改：只保留 BCE 的正项 `−p_t·log(p_max)`。它只会把峰值往上推，不会往下压。

**(b) 硬 max 只把梯度传给一个位置**。训练早期 argmax 基本是随机的，收敛慢且不稳定。

修改：用 top-k 均值作为软 max。

**(c) 方位到特征列是线性映射**（`:102`）。ARIS 的 128 个波束在方位上**不均匀**，仓库里已有 `aris_beam_table.col_of_angle`，这里没有用上。

修改：注入一个 `bearing_to_col` 函数。

```python
class CrossPlaneMILLoss(torch.nn.Module):
    def __init__(self, m0, bearing_to_col_norm, k_margin=2.0,
                 base_margin_deg=1.0, level=0, topk=4,
                 bearing_axis="x"):
        """bearing_to_col_norm: θ[deg] -> 归一化列坐标 [0,1]，
        例如 UXO/ARIS: lambda th: col_of_angle(th) / (n_beams - 1)。"""
        ...
        self.bearing_to_col_norm = bearing_to_col_norm
        self.topk = topk

    def forward(self, teacher_boxes, teacher_obj, student_obj_logits,
                camera_img_w, reach=None):
        probs = student_obj_logits[self.level].sigmoid()
        b, _, h, w = probs.shape
        losses, n_used = [], 0
        for i, (boxes, objs) in enumerate(zip(teacher_boxes, teacher_obj)):
            for j, ((x1, _, x2, _), p_t) in enumerate(
                    zip(boxes, objs.tolist())):
                win = window_fov_overlap(
                    azimuth_window(x1, x2, camera_img_w, self.m0,
                                   k_margin=self.k_margin,
                                   base_margin_deg=self.base_margin_deg),
                    self.m0)
                if win is None:
                    continue
                c1 = int(np.floor(self.bearing_to_col_norm(win[0]) * (w - 1)))
                c2 = int(np.ceil(self.bearing_to_col_norm(win[1]) * (w - 1))) + 1
                c1, c2 = max(min(c1, c2), 0), min(max(c1, c2), w)
                if c2 <= c1:
                    continue
                strip = probs[i, 0, :, c1:c2].flatten()
                k = min(self.topk, strip.numel())
                p_soft = strip.topk(k).values.mean().clamp(1e-6, 1 - 1e-6)
                # 单边：只保留 BCE 的正项，永不把声呐峰值往下压
                l = -float(p_t) * torch.log(p_soft)
                if reach is not None:          # §3.4 可达性权重
                    l = l * reach[i][j]
                losses.append(l)
                n_used += 1
        if n_used == 0:
            return probs.sum() * 0.0, probs.new_tensor(0.0)
        return torch.stack(losses).mean(), probs.new_tensor(float(n_used))
```

**(d) M₀ 的几何模型过于简化**（`m0.py`）：
1. 像素到角度用的是仿射 `θ = slope·x + intercept`。针孔模型下应为 `atan((x − cx)/fx)`；GoPro 是广角，还有明显畸变，边缘误差会很大。
2. UXO 的相机和声呐不共心。在约 1 m 的近场，基线视差带来的方位偏差约为 atan(b/r)：b = 0.1 m 时约 5.7°，在 ARIS 30° / 128 波束上大约相当于 25 个波束。所以方位对应**随距离变化**，"方位条带"本身就不准确。
3. UXO 每段录音的相机方位是**分别标定**的（E03），单一 M₀ 不成立。而且 `m0.py` 的说明写的是"读取 F1 在 RSFusion 上的拟合"，这份拟合在 UXO 上并不存在。

**修改（推荐用于 UXO）：射线曲线 MIL**。UXO 有逐帧位姿和标定（`uxo_projection.py`）。对教师相机框的中心列，取它在相机中的射线，在 [r_min, r_max] 范围内采样若干个距离，投影到声呐 (row, col) 平面，得到一条**曲线**而不是竖直条带。MIL 沿这条曲线（按 M₀ 残差外扩成一个带）取软 max。

- 离线预计算：转换阶段对每帧写入一个 sidecar 文件 `ray_curves/<sample_id>.npy`，内容是"相机列（归一化，比如 32 个采样列）→ 声呐 (row, col) 折线"。投影函数直接复用 `uxo_projection.py` 中已有的 predict。
- 训练时按教师框的 x 中心选最近的那条折线，栅格化成特征图上的一个 mask，再在 mask 内取 top-k。
- 对 RQ-3，这给出三个可比的变体：条带（M₀）、射线曲线（标定几何）、单目深度定点（RODNet 式，草案 §11.1 建议加入）。

M₀ 本身的修改：增加 `camera_model="pinhole"` 选项（`θ = atan((x·W − cx)/fx) + yaw`），并支持按录音加载（`load_m0(path)[recording]`）。

### 3.7 💡 2×2 归因：更省、更干净的设计

**现状设计（草案 §6.3）**：四个格子各用一个教师，共需训练 4 个教师（大容量 / 同容量 × 清晰输入 / 退化输入）。

**问题**：
- 教师训练成本高（草案 §6.5 估计 24–40 GPU 小时）。
- M1 与 M3 使用的是**两个不同的教师网络**，特权效应 (M3 − M1) 里混进了两次教师训练之间的随机差异。

**建议**：只训练 2 个教师（大容量、同容量），**训练时都带退化增强**（B3 式，D0–D4 都见过），这样清晰输入和退化输入对教师来说都在分布内。然后每个教师分别以两种方式喂入：

| | 教师输入：学生同款退化相机 | 教师输入：清晰相机 |
|---|---|---|
| 同容量教师 T_s | D-self | M2 |
| 大容量教师 T_L | M1 | M3 |

- 特权效应 (M2 − D-self) 和 (M3 − M1)：同一份教师权重、只改输入，是**配对比较**，教师训练的方差被消掉了。
- 容量效应 (M1 − D-self) 和 (M3 − M2)：仍然是跨教师比较，与原设计相同。
- 教师训练次数从 4 减到 2。

**代价与限制**：教师是在退化数据上训练的，它在清晰输入上的表现可能略低于"只在清晰数据上训练的教师"。所以 M3 的绝对值可能偏保守。建议在 S2 加一个单种子对照（清晰数据训练的大教师 → M3），确认差距在种子噪声以内。

**配置写法**：

```python
# M1：teacher_input="degraded"；M3：teacher_input="clear"；两者 teacher_checkpoint 相同
distillation = dict(
    teacher=_teacher_L_cfg,
    teacher_checkpoint="work_dirs/teacher_L_b3_seed0/epoch_100.pth",
    teacher_input="clear",            # M1 改为 "degraded"
    routes={"camera": "camera", "sonar": "sonar", "sonar_uncoupled": "sonar"},
    obj_weight=1.0, score_weight=0.5, feat_weight=0.0, bg_floor=0.01)
```

建议在立项评审中，把这一点作为草案 §6.3 的修改提案提出。

---

## 4. 实验设计、评测与计划一致性

### 4.1 🔴 相关性表：不按 split 过滤、GT 源与训练口径不一致（读码确认）

**位置**：`tools/correlation_table.py:98`

**问题**：
1. `for fr in frames:` 遍历 `frame_meta.csv` 中**所有**帧。但预测文件只覆盖 test 帧。train 帧（E07 划分下 1680 帧，约占 65%）没有任何预测，两个支路都被判为漏检，构成大量强制的 (1,1) 配对，φ 和 ΔP_miss 被系统性虚高。
2. 声呐 GT 用的是所有源的注释，含 projected 框（E06 已判定为"无回波帧上的几何猜测"），与 blob-only 训练口径不一致。声呐在这些帧上必然漏检，同样抬高相关性。
3. 只报告第一个找到的声呐支路（融合头优先），不报告未耦合头。而草案张力 1 的"分开记账"正是要比较两者。
4. 阈值 τ 固定为 0.25，不是草案 §6.4 要求的"验证集上每图 1 个虚警"的工作点。
5. `gt_cam = {a["image_id"]: a ...}` 每张图只保留一个注释（后写覆盖先写）。UXO 每帧单目标，目前没问题，但要加断言防止换数据集后静默出错。
6. 下游代理（朴素贝叶斯融合 ECE）在 `branch_correlation.py` 中已经实现，但没有接进这个工具。

**修改**：

```python
# tools/correlation_table.py  —— 参数
ap.add_argument("--split", default="test")
ap.add_argument("--sonar-gt-sources", nargs="+", default=["blob"])
ap.add_argument("--tau-json", default=None,
                help="由验证集选出的 {branch: tau}；缺省时回退到 --tau 并在输出中标注")

# —— GT 读取
from aodet.evaluation.per_branch_metric import filter_gt_sources
sonar = filter_gt_sources(_load(os.path.join(args.paired, "sonar.json")),
                          tuple(args.sonar_gt_sources))
def _one_per_image(anns):
    by = {}
    for a in anns:
        if a["image_id"] in by:
            raise ValueError(f"image {a['image_id']} has >1 annotation; "
                             "matched-pair logic assumes one target/frame")
        by[a["image_id"]] = a
    return by
gt_cam = _one_per_image(camera["annotations"])
gt_son = _one_per_image(sonar["annotations"])
frames = [fr for fr in frames if fr.get("split") == args.split]

taus = json.load(open(args.tau_json)) if args.tau_json else {}

# —— 对每个存在的声呐支路分别出一行（融合头、未耦合头都报告）
for level in args.levels:
    ...
    for son_b in [b for b in SONAR_BRANCHES if b in preds]:
        tau_c = taus.get(cam_b, args.tau)
        tau_s = taus.get(son_b, args.tau)
        miss_c, miss_s, conf_c, conf_s = [], [], [], []
        for fr in frames:
            ...   # 与原逻辑相同，阈值分别用 tau_c / tau_s
        rep = summarize_matched_pairs(miss_c, miss_s, conf_c, conf_s)
        rows.append(dict(level=level, split=args.split,
                         camera_branch=cam_b, sonar_branch=son_b,
                         tau_camera=tau_c, tau_sonar=tau_s,
                         tau_source="val" if args.tau_json else "fixed",
                         sonar_gt=",".join(args.sonar_gt_sources),
                         **rep.as_row()))
```

ECE 接入：需要每个支路的 pD 与 pFA。pFA 要在**无目标**的帧或区域上统计，而 UXO 每帧都有目标，所以必须先定义负样本单元。建议用"与 GT 不重叠的检测"按每图计数，换算成每图虚警率。在定义冻结之前，ECE 列先留空，并在台账中写明原因，不要用不合适的定义凑数。

**`tau_json` 的产生**：在验证协议（§4.3）的验证折上，对每个支路找到"每图平均 1 个虚警"时的分数阈值，写入 JSON，随 run 目录保存。

**测试**（`tests/test_correlation_table.py` 中新增）：构造 2 个 test 帧和 3 个 train 帧，train 帧没有预测；断言 `n_pairs == 2`，且 φ 与只用 test 帧手算的结果一致。

### 4.2 🟠 blob 口径评测：projected 帧被当成负样本（读码确认）

**位置**：`tools/test.py`（`variants.append(("blob", filter_gt_sources(gt, ("blob",))))`，`img_ids` 不变）

**机制**：`filter_gt_sources` 删掉了 projected 注释，但这些帧仍在 `img_ids` 里。这些帧里**物理上确实有目标**，只是启发式 blob 检测器没过门。模型如果在这里正确地检测到了弱回波，会被算作虚警，压低 precision，压低 AP。

**修改**：blob 口径只在"有 blob 注释或完全没有注释"的帧上评测，把 projected-only 帧排除出去。

```python
# per_branch_metric.py
def blob_eval_img_ids(gt, img_ids):
    """blob 口径：排除只有 projected 注释的帧（目标在但回波未过门，
    既不能当正样本也不能当负样本）。"""
    has_blob = {a["image_id"] for a in gt["annotations"]
                if a.get("source", "blob") == "blob"}
    has_any = {a["image_id"] for a in gt["annotations"]}
    return [i for i in img_ids if i in has_blob or i not in has_any]
```

```python
# tools/test.py
if side == "sonar":
    variants.append(("blob", filter_gt_sources(gt, ("blob",)),
                     blob_eval_img_ids(gt, img_ids[side])))
for gt_name, gt_v, ids in variants:
    m = evaluate_prediction_file(gt_v, path, img_ids=ids)
```

（`("all", gt)` 一项相应改为 `("all", gt, img_ids[side])`。）

**训练端的同一问题**：blob-only 训练把 projected 帧当作"诚实负样本"（E06 结论）。E06 表中 180722 的 projected 帧对比度 +7、dy p50 16 px，说明其中一部分帧有弱回波。把它们当负样本，会教模型压制弱回波。

- 理想做法是把这些帧的目标区域设为 ignore。但 mmdet 3.3 的 `YOLOXHead.loss_by_feat` 虽然在签名里接受 `batch_gt_instances_ignore`，SimOTA 并不使用它（需在本机源码核实）。
- 可行的替代：在数据集中加一个选项 `sonar_skip_projected_frames=True`，让这些帧不参与**声呐**头的损失（相机头照常训练）。实现上可以在 pack 阶段打一个 `sonar_valid=False` 标记，在 `loss()` 里把该样本从声呐头的 `batch_data_samples` 中剔除（声呐特征同时按 batch 维切片）。
- 两种处理（负样本 vs 跳过）做一个单种子对照，记入台账。

### 4.3 🔴 没有验证集，test 被用于诊断和定参（台账确认）

**事实**：
- `uxo_to_paired.py` 只产生 train/test 两个 split。
- E05 → E06 → E07 → E08 的诊断都基于 test 录音上的预测。E08 的抖动范围 [0.6, 1.8] 是根据 100927（test）的回波尺寸 12–17 px 推算出来的。
- 草案 §6.1 规定："退化参数只在训练集和验证集上调，测试集只评测一次"。§6.4 的工作点也要求"在验证集固定"。

**后果**：E08 即便成功，也是在 test 统计量上调出来的方法在 test 上的结果，已经不是一个干净的泛化测量。另外，sample 只有 5 段录音，任何录音级划分的 test 都只有 2 段，结论只能算个例。

**修改**：
1. **立刻**：在台账中给 E05–E08 补一条"test 已被用于诊断，结果为开发集性质"的注记。今后 sample 上的所有诊断明确称为"开发集"。
2. **短期（仍在 sample 上）**：用录音级留一交叉验证（LOO-CV，5 折）作为开发协议，按折报告均值和极差。数据集按录音过滤（`frame_meta.csv` 已经有 `recording` 列）：

```python
# paired_dataset.py  __init__ 新增参数
recordings: Optional[Sequence[str]] = None,      # 白名单
exclude_recordings: Optional[Sequence[str]] = None,
...
# load_data_list 中，split 过滤之后
rec = sample["meta"].get("recording")
if self.recordings is not None and rec not in self.recordings:
    continue
if self.exclude_recordings is not None and rec in self.exclude_recordings:
    continue
```

```bash
# tools/run_loocv.sh（新增）：每折把一个录音作为 val，其余 train；
# 用 --cfg-options 覆盖两个 dataloader 的录音过滤，split 过滤置空
for rec in 2023-09-20_180722 2023-09-21_100927 2023-09-21_105856 \
           2023-09-21_123500 2023-09-21_124735; do
  python tools/train.py "$CONFIG" --seed 0 \
    --work-dir "work_dirs/$(basename $CONFIG .py)_loo_${rec}" \
    --cfg-options train_dataloader.dataset.split=None \
                  train_dataloader.dataset.exclude_recordings="['$rec']" \
                  test_dataloader.dataset.split=None \
                  test_dataloader.dataset.recordings="['$rec']"
done
```

（需要核实 `--cfg-options` 对 `None` 和列表的解析；必要时为每折生成一个小配置文件。录音名以 `frame_meta.csv` 中的实际值为准。）

3. **中期**：下载全量 UXO（草案 TODO 14.2 第 4 项），按"train / val / test 三份录音级划分"冻结协议；test 录音在冻结后只评测一次。
4. test 录音的选择应当避开标定最可疑的录音（100927：b 顶在边界上、锚定量 −0.79 m），或者把它的 GT 质量问题单独列为一个已知混淆。

### 4.4 🟠 S0-lite 偏离 RQ-0 的设计（读码确认）

**事实**：
- 草案 §3 的 S0：在**名义条件训练**的融合模型（B2，RSFusionDet 官方）上做推理期退化扫描，测 Δ_s；然后再训练 B3（退化增强）看能恢复多少。
- 现在唯一的融合配置 `s0_cross_attn_yoloxs.py` 继承 `paired_uxo.py` 的训练管线，带 `AOSampleDegLevel`（D0–D4 各 0.16，D∞ 0.2）。这是"退化增强 + 模态丢弃"，接近 B4。
- 带 D∞ 和退化训练的模型，本来就被训练成在相机退化时不依赖相机，最可能**掩盖**互毒害。用它回答 RQ-0，方向上偏向得出"不存在"的结论。

**修改**：增加名义训练的 B2 配置，与现有配置（改名为 B4-lite）并列。

```python
# configs/f2/s0_b2_cross_attn_nominal_yoloxs.py
"""S0-lite B2：名义条件（D0）训练的 cross-attn，推理期做 D0–D∞ 扫描。
RQ-0 的直接对象：训练从未见过退化的融合模型，相机退化时声呐支路是否下降。"""
_base_ = ["s0_cross_attn_yoloxs.py"]

train_pipeline = [
    dict(type="AOLoadPaired"),
    dict(type="AOPairedResize",
         camera_size=(640, 640), sonar_size=(640, 640), keep_ratio=False),
    dict(type="AOPairedSyncFlip", prob=0.0, allow=True),
    dict(type="AOSampleDegLevel",
         probs=dict(d0=1.0, d1=0.0, d2=0.0, d3=0.0, d4=0.0, dinf=0.0)),
    dict(type="AOStudentCameraView", dinf_mode="zero", jitter=0.08),
    dict(type="AOPackPairedInputs"),
]
train_dataloader = dict(dataset=dict(pipeline=train_pipeline))
```

```python
# configs/f2/s0_b3_cross_attn_degaug_yoloxs.py（B3：只有 D0–D4，没有 D∞）
_base_ = ["s0_cross_attn_yoloxs.py"]
train_pipeline = [
    ...,   # 同上，仅 AOSampleDegLevel 改为：
    dict(type="AOSampleDegLevel",
         probs=dict(d0=0.2, d1=0.2, d2=0.2, d3=0.2, d4=0.2, dinf=0.0)),
    ...,
]
```

现有的 `s0_cross_attn_yoloxs.py` 在台账中改称"B4-lite（退化 + 模态丢弃）"，避免与 B2 混淆。

S0-lite 的最小矩阵因此是：B0、B2、B3、B4-lite，各 1 个种子，都在验证协议（§4.3）下评测。RQ-0 的核心读数是 B2 的 Δ_s(D4)，其次看 B3 和 B4 恢复了多少。

### 4.5 🟡 台账与代码不一致（读码确认）

| 位置 | 台账 / 草案的说法 | 代码现状 | 处理 |
|---|---|---|---|
| `doc/experiments.md` E01 结论 | "NMS 0.65/0.01 的预处理约定已同步固化进 `two_stream_yolox_s.py`" | `two_stream_yolox_s.py:72` 仍为 `score_thr=0.05`、NMS `0.55` | 按 §1.4 修改配置；台账 E01 加勘误注记，说明 E07 中 B0 的数字是在 0.05/0.55 下得到的 |
| 草案 TODO 14.2 | `--calib-cache` 未勾选 | E06 已实现（`--use-cache` + `projection_calib.json`） | 勾选，注明日期并链接 E06 |
| 草案 TODO 14.3 W3 | "关闭 Mosaic 和 MixUp" 已勾选 | 一致 | — |
| `doc/experiments.md` E08 设置 | 抖动"在 `AOLoadPaired` 之后、`AOPairedResize` 之前" | 这个顺序正是 bug 所在 | 按 §1.2 修改，加勘误注记 |
| `m0.py` 说明 | "读取 F1 在 RSFusion 上的拟合" | UXO 上并不存在这份拟合 | 按 §3.6(d) 修改，支持按录音加载 |
| 草案 §5.4 | "教师在声呐头上的 objectness" | — | 改为"教师**未耦合**声呐头的 objectness"（§3.4） |

---

## 5. 修复顺序与验收

| 优先级 | 内容 | 涉及节 | 验收 |
|---|---|---|---|
| P0（启动 E08 之前） | KD 键名修复 + 梯度测试；E08 抖动顺序修复 + 管线级测试 | §1.1、§1.2 | 两个新测试通过；全量测试在两个环境下全绿 |
| P0 | 相关性表：按 split 过滤、GT 源过滤、两个声呐头都报告 | §4.1 | 新测试 `n_pairs` 正确；对 E07 的 B0 / cross-attn 重算一次并写入台账 |
| P1 | D∞ 训练与测试统一（`camera_valid` + LayerScale + 无输出 LN）；D∞ 帧不计相机损失 | §1.5、§2.1、§2.2A | D∞ 一致性测试通过 |
| P1 | 预训练初始化、YOLOX 优化配方、EMA、测试阈值；声呐监督权重对齐 | §1.3、§1.4 | 配置测试通过；B0 重训，单种子，新编号 |
| P1 | 验证协议（LOO-CV）；补 B2 / B3 配置；台账注记 | §4.3、§4.4、§4.5 | S0-lite 矩阵 B0/B2/B3/B4-lite 在 LOO-CV 下出数 |
| P2（进入 S2 之前） | KD 重写（前景加权 BCE、分数蒸馏、路由、特征 KD）；教师严格加载；单次前向重构；退化抽样随机性 | §1.6、§1.7、§3.1–§3.5 | KD 单元测试；重构前后损失逐项相等；KD/检测损失比值记录在台账 |
| P2 | 方位轴注意力（方案 B）；stop-grad 变体；方位对齐的 Concat | §2.2B、§2.3、§2.4 | 新结构单元测试；与方案 A 的单种子对照 |
| P3（S3 之前） | L_xp 单边化、软 max、波束表、M₀ 针孔化 + 按录音加载；射线曲线 MIL | §3.6 | L_xp 测试：p_max > p_t 时梯度方向不向下 |
| 评审 | 2×2 改为"2 个教师 × 2 种输入"；草案 §5.4、§6.3 措辞修改 | §3.4、§3.7 | 立项评审决定 |

**合入规则**：每一项单独提交，提交信息里写明本文档的节号；改动行为的修复（P0、P1）在 `doc/experiments.md` 中以勘误注记的方式记录，不回改已冻结的数字。

---

## 6. 实施记录（分支 `fix/review-2026-10-31-kd-arch-training`）

**测试**（CPU `.venv`：torch 2.1.2+cpu、mmcv 2.1.0、mmengine 0.10.7、mmdet 3.3.0、numpy 1.26.4、scipy 1.15.3）：
`pytest` 共 130 项，**129 通过，1 失败**。

- 失败项：`tests/test_uxo_projection.py::test_calibrate_recovers_synthetic_tank[aim-line]`（`P error 1.246 m`，断言阈值 0.8 m）。
- 这项失败**与本分支无关**：在未修改的 `main`（`1a69ba5`）上用同一环境复跑，同样失败。本分支没有改动 `uxo_projection.py` 及其测试。排查与修复见 §7。
- `.venv-gpu`（torch 2.7.1+cu128）尚未复跑。

**新增或修改的回归测试**：KD 梯度可达（§1.1）、E08 抖动的管线级测试和配置顺序（§1.2）、声呐监督权重对等（§1.3）、YOLOX 配方（§1.4）、D∞ 清空相机 GT（§1.5）、单次前向与 `head.loss` 逐项一致（§1.6）、退化抽样随机性（§1.7）、三种融合的 D∞ 训练/测试一致性（§2.1）、位置感知与 LayerScale（§2.2）、方位注意力（§2.2B）、KD 前景加权、量级和分数蒸馏（§3.1–§3.2）、KD 路由与特征 KD、可达性权重（§3.3–§3.4）、教师严格加载（§3.5）、L_xp 单边约束与非线性方位映射（§3.6）、相关性表的 split / 录音 / GT 源过滤与双声呐头输出（§4.1）、blob 口径的 img_ids（§4.2）、B2 配置（§4.4）。

**不兼容变更**：
- `CrossAttnFusion` 的参数结构变了（新增 `norm_q`、`norm_kv`、`gamma`，删除了 `norm`），E07 的 cross-attn checkpoint 不能再加载到新代码。
- `branch_kd` 删除了 `objectness_mse` 和 `cls_kl`；`AOSampleDegLevel` 的 `seed` 默认值从 0 改为 None。

## 7. `test_uxo_projection` aim-line 失败的排查（2026-10-31）

**现象**：`test_calibrate_recovers_synthetic_tank[aim-line]` 报 `P error 1.246 m`（阈值 0.8 m）。包版本与 `env/pip-freeze.txt` 完全一致（numpy 1.26.4、scipy 1.15.3、opencv 4.11.0.86），但 1f5a7e0 的提交记录称该测试在另一台机器上通过。

**诊断**：

| 检查 | 结果 | 读法 |
|---|---|---|
| 拟合的 P | x/y 误差约 1 cm，z = +0.046 m（真值 −1.20 m） | 目标被放到了云台上方，物理上不可能 |
| 残差 | 距离 rms 1.7 mm，相机 0.51°，方位 0.34° | 所有残差都很好，问题不在拟合质量 |
| 与真值的关系 | z ≈ 真值关于传感器中心水平面（z ≈ −0.59）的镜像 | 镜像解 |
| 两个盆地的代价 | 真值盆地 70.65，镜像盆地 71.30（差约 1%） | 近退化：直线行程 + 逐帧瞄准时，镜像不改变任何距离，方位被逐帧瞄准吸收 |

**根因**有两层：
1. **近退化的镜像解**：两个极小值代价只差约 1%，最终落进哪个取决于浮点细节。这解释了同样的包版本、不同机器结果不同。
2. **盆地选择用错了代价**：瞄准模式下第一阶段只用相机残差挑选盆地。但云台每帧都对准 P，相机残差对 P 几乎是平的，第一阶段的 argmin 等于随机挑一个盆地，第二、三阶段只在这个盆地里精化。
   - 只加物理约束（下面第 1 条）后，原种子落到了离真值约 1 m、贴在约束边界上的另一个错误解。
   - 改成"按相机代价取前 4 个不同解"后，原种子通过，但另一个噪声种子（13）仍漏掉了真值盆地。

**修复**（`aodet/datasets/converters/uxo_projection.py`）：
1. **物理先验**：目标在声呐中心下方至少 `_P_BELOW_HEAD_M = 0.05` m（目标在池底，声呐头朝下看）。在所有会移动 P 的阶段（第一阶段、第三阶段）加 P_z 上界，初值裁剪到可行域内。
2. **盆地覆盖**：瞄准模式下，每个目标位置假设（`p_hyps`）各取最好的第一阶段解，合并相距不足 `_CAND_MIN_SEP_M = 0.15` m 的近重复解，全部跑完第二、三阶段，最后按**联合代价**（其中的距离项能区分盆地）选优。固定瞄准约定的模式不受影响，仍然只精化相机代价最优的那一个解。

**验证**：
- aim-line 原种子：P 误差 0.089 m，b = −0.026 m（真值 0.06）。
- 另外 3 个噪声种子（12/13/14）：P 误差 0.121 / 0.201 / 0.080 m，P_z 都在传感器平面以下。
- 测试中新增断言：拟合的 P_z 必须低于声呐中心。

**代价与影响**：
- 瞄准模式的标定变慢，每段录音要多精化几个候选（aim-line 合成例约 45 s）。
- **真实数据的 GT 可能变化**：E03 中 180722 和 123500 是瞄准模式（yaw = 2），它们的标定可能选到与之前不同的盆地。需要在有数据的机器上用 `tools/project_uxo_sonar.py` 重新标定（不能用 `--use-cache`，因为缓存键不包含代码版本），并与 `projection_report.json` 的旧结果逐段比较 P、b 和 dcol。若有变化，属于 GT 变更，后续训练需新开实验编号。
