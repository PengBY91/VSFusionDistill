# 2026-11-02 tau4 断点续训事故（weights_only）与恢复

 τ 消融链（E10 判据 6，`work_dirs/tau_ablation_chain.log`）的时间线、根因与处置。结论先行：tau1 已完整出数；tau4 自 epoch_10 续训成功（relaunch 4）；根因是 torch 2.7.1 的 `weights_only=True` 默认值撞上 mmengine 0.10.7 不带覆盖参数的 `torch.load`，`env/ENVIRONMENT.md` 2026-10-31 已预言该路径会炸；fa2777c 的 HistoryBuffer 类级 allowlist 单独不解决问题（实测），当前靠 `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1` 环境变量支撑，持久修复见文末待办。

## 时间线（均为 2026-11-02，时间取自日志）

| 时刻 | 事件 |
|---|---|
| 00:09 | tau4 训至 epoch_10/100 并落盘（11-01 23:50 起跑） |
| 00:23 | 上一个会话重启，链进程被杀（子进程随会话终止） |
| ~00:24 | relaunch 1：重挂即崩，`UnpicklingError: numpy.core.multiarray._reconstruct`（auto-resume 加载 epoch_10.pth） |
| 05:00:17 | relaunch 2：同样崩溃（此时 working tree 已加 HistoryBuffer allowlist，即 fa2777c 的内容） |
| 05:27:42 | relaunch 3：带环境变量启动，但漏了 `--cfg-options resume=True`，mmengine 从 epoch 1 重新开训（mmengine 0.10.7 只在 `cfg.resume` 为真时 auto-resume，runner.py:1686） |
| 05:29 | 人工发现 epoch 计数器为 [1]，杀掉；epoch_10.pth 未被覆写（fresh run 尚未到自己的 epoch 10 保存点） |
| 05:30:04 | 并行会话提交 fa2777c（train.py 加 allowlist）；prereq 5 同期落盘 6e2043a |
| 05:31 | relaunch 4：`TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1` + `--cfg-options resume=True`，Auto resumed epoch_10 成功，Epoch [11] 起训，eta 约 3 h；链尾自动接 D0–D∞ 扫描与 collect |

## 根因

1. torch 2.6 起默认 `torch.load(weights_only=True)`；mmengine 0.10.7 的 `load_from_local` 不传该参数。checkpoint 的 `meta.log_scalars` 里是 HistoryBuffer，其 pickle state 内含 numpy 标量，触发 `numpy.core.multiarray._reconstruct` 全局拒绝。
2. tau1 从头一次跑完 100 epoch，resume 路径在本机从未执行过；E01–E10 所有 run 也都是 fresh start。路径没测过，等于没有。
3. `env/ENVIRONMENT.md` 2026-10-31 节倒数第二段明确写过"A future `cfg.resume=True` on this env would need the same workaround"（test.py 已用 scoped `weights_only=False` 包装处理同一问题），预言命中，但没人把它变成 train 侧的修复或冒烟测试。

## fa2777c 为什么不够（实测）

`.venv-gpu/bin/python` 下，仅执行 fa2777c 的 `torch.serialization.add_safe_globals([HistoryBuffer])` 后直接 `torch.load(epoch_10.pth)`，仍然抛 WeightsOnlyLoadError。类级 allowlist 放行的是 HistoryBuffer 这个类型本身，放行不了其 state 内部的 numpy 全局。fa2777c 的提交信息声称修复 resume，单独不成立；当前续训进程真正依赖的是启动环境变量。

## 处置与待办

- 当前：tau4 由 `/tmp/tau4_resume.sh`（setsid 脱离会话）驱动，环境变量 + resume=True，链尾接扫描与 collect；出数按路线图只入账不解读。
- 待办一（E11 数据闸门前必须完成）：把 test.py 的 scoped `weights_only=False` 包装移植到 train/resume 路径（信任本地自产 checkpoint），替代环境变量方案；补一个"保存 1 epoch 后 resume 成功"的冒烟测试。E11 三臂 × 3 种子的长训练必然用到 resume，此路径必须有测试覆盖。
- 待办二：本事故的流程教训已进复盘 §8（无人值守链的 .ok 标记、重试两次停链、启动后人工确认、断点先冒烟）。
- 关联：`work_dirs/tau_ablation_chain.log`（relaunch 1–4 全程）、`env/ENVIRONMENT.md` GPU 节、`doc/experiments.md` E10 注记。
