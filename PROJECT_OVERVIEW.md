# 项目交接总览

## 1. 这份文档的用途
这不是面向外部用户的 README，而是给后续继续接手本项目的 Codex / 开发者看的交接说明。目标是让接手者快速回答以下问题：

1. 这个项目原始 baseline 在做什么
2. 当前代码已经在 baseline 上改了什么
3. 哪些功能已经能走通，哪些只是部分实现，哪些还只是想法
4. 后续应该优先从哪里继续接

本文档基于当前代码库实际状态整理，而不是只基于设想文档。

## 2. 当前项目在做什么

### 2.1 原始 baseline
项目来自 Afford-Motion，两阶段任务定义是：

1. **第一阶段 ADM / CDM**
   输入：`scene point cloud + text`
   输出：`affordance map`

2. **第二阶段 AMDM / CMDM**
   输入：`affordance map + text`
   输出：`motion sequence`

原始 baseline 的 affordance 是**静态 affordance**：对整段 motion 的逐帧几何关系做时间聚合后，得到一张单图。

### 2.2 当前代码上的新方向
当前代码已经不再是纯 baseline，而是在其上叠加了两个 idea：

1. **Idea 1：Temporal Affordance Sequence**
   把单张静态 affordance 改成固定 `K=4` 的 phase-wise affordance sequence。

2. **Idea 5：Contact-Consistent Diffusion Loss**
   在第二阶段 AMDM 的 motion MSE 基础上，增加几何一致性损失：
   - penetration loss
   - contact loss

## 3. baseline 是什么

### 3.1 原始 affordance 构造逻辑
baseline 的 affordance 不是人工标注，而是由 GT motion 和 scene 几何自动构造。

原始思路：
1. 对每个 scene point，计算它到整段 motion 中各 joint 轨迹的最近距离
2. 再转换成 contact / affordance 表示
3. 第一阶段用它监督 ADM
4. 第二阶段把它作为条件输入 AMDM

对应代码主入口：
- [prepare/generate_contact_data.py](/Users/kbyte/Desktop/afford-motion-main/prepare/generate_contact_data.py)

### 3.2 原始训练逻辑
- 第一阶段：`scene + text -> affordance`
- 第二阶段：`affordance + text -> motion`
- diffusion 训练核心 loss 在：
  - [diffusion/gaussian_diffusion.py](/Users/kbyte/Desktop/afford-motion-main/diffusion/gaussian_diffusion.py)

## 4. 当前已实现的 idea

### 4.1 Idea 1：Temporal Affordance Sequence
**状态：已实现 MVP**

当前实现不是 learned segmentation，也不是 language-driven segmentation，而是：
- 固定 `K = 4`
- 对 motion 做均匀分段
- 每段单独生成一张 phase-wise affordance

关键结果：
- 磁盘上保留旧的静态 `dist`
- 新增 `dist_seq: [4, N, 22]`
- 第一阶段训练内部把 `[4, N, 6]` 展平成 `[N, 24]`
- 第一阶段评测/保存时再恢复成 `[num_samples, 4, N, 6]`
- 第二阶段模型接收 `[B, 4, N, 6]`

对应说明文档：
- [baseline_summary.md](/Users/kbyte/Desktop/afford-motion-main/baseline_summary.md)
- [idea1_spec.md](/Users/kbyte/Desktop/afford-motion-main/idea1_spec.md)

### 4.2 Idea 5：Contact-Consistent Diffusion Loss
**状态：已实现 MVP，默认关闭**

当前实现只加在第二阶段 `CMDM` 上，不改 ADM，不改 Idea 1 的主流程。

当前 loss 形式：
- `L_total = L_mse + lambda_contact * L_contact + lambda_pen * L_penetration`

当前实现内容：
- `contact loss`
- `penetration loss`

当前未实现：
- foot sliding loss
- 更复杂的物理仿真
- 端到端联合训练

对应说明文档：
- [current_status.md](/Users/kbyte/Desktop/afford-motion-main/current_status.md)
- [idea5_spec.md](/Users/kbyte/Desktop/afford-motion-main/idea5_spec.md)

## 5. 现在的完整流程

## 5.1 数据预处理流程
官方预处理流程在：
- [README.md](/Users/kbyte/Desktop/afford-motion-main/README.md)
- [prepare/README.md](/Users/kbyte/Desktop/afford-motion-main/prepare/README.md)

主要步骤包括：
1. `prepare/process.py`
2. `prepare/smplx_to_vec.py`
3. `prepare/process_scene.py`
4. `prepare/generate_contact_data.py`
5. `prepare/split.py`

### 当前与 baseline 不同的关键点
[prepare/generate_contact_data.py](/Users/kbyte/Desktop/afford-motion-main/prepare/generate_contact_data.py) 已经改成：
- 保留静态 `dist`
- 新增 temporal `dist_seq`

关键函数：
- `compute_contact_distance_map()`：静态整段距离图
- `compute_temporal_contact_distance_map()`：按 `num_phases` 均匀切分后逐段生成 `[K, N, J]`

当前 `process()` 保存的是：
- `points`
- `mask`
- `dist`
- `dist_seq`

### 预处理状态判断
- **已实现**：Idea 1 所需的 temporal contact 预处理
- **待确认**：当前工作区里没有看到实际 `data/` 目录内容，运行时数据是否已重建要结合本机外部数据确认

## 5.2 第一阶段 ADM / CDM

### 当前目标
第一阶段现在不再预测单张静态 affordance，而是预测 4 段 temporal affordance 的展平表示。

### 当前数据形式
GT 原始形式：
- `dist_seq: [4, N, 22]`

选取 6 个 contact joints 后：
- `[4, N, 6]`

送入 ADM 前展平为：
- `[N, 24]`

### 数据集代码
- 通用多数据集版本：
  - [datasets/motionx.py](/Users/kbyte/Desktop/afford-motion-main/datasets/motionx.py)
  - 关键类：`ContactMapDataset`
- HumanML3D 单独版本：
  - [datasets/humanml3d.py](/Users/kbyte/Desktop/afford-motion-main/datasets/humanml3d.py)
  - 关键类：`ContactHumanML3DDataset`

当前做了这些事：
- 增加 `temporal_affordance` 开关
- 增加 `num_phases`
- 读取 `dist_seq`
- 支持 `recover_contact_sequence()`
- 单独保存 temporal 版本的 mean/std 文件

### 模型代码
- [models/cdm.py](/Users/kbyte/Desktop/afford-motion-main/models/cdm.py)

`CDM` 本体没有做大重构，核心思路是：
- 继续把输入当成 point-wise contact tensor
- 只是 `contact_dim` 从原来 `6` 变成 `24`

维度由这里统一计算：
- [utils/misc.py](/Users/kbyte/Desktop/afford-motion-main/utils/misc.py)
  - `compute_model_input_dimension()`

### 第一阶段训练入口
- [train.py](/Users/kbyte/Desktop/afford-motion-main/train.py)
- [train_ddp.py](/Users/kbyte/Desktop/afford-motion-main/train_ddp.py)

常用脚本：
- [scripts/t2m_contact/train.sh](/Users/kbyte/Desktop/afford-motion-main/scripts/t2m_contact/train.sh)
- [scripts/t2m_contact/train_ddp.sh](/Users/kbyte/Desktop/afford-motion-main/scripts/t2m_contact/train_ddp.sh)
- [scripts/ts2m_contact/train.sh](/Users/kbyte/Desktop/afford-motion-main/scripts/ts2m_contact/train.sh)
- [scripts/ts2m_contact/train_ddp.sh](/Users/kbyte/Desktop/afford-motion-main/scripts/ts2m_contact/train_ddp.sh)

### 第一阶段评测 / 推理
- [test.py](/Users/kbyte/Desktop/afford-motion-main/test.py)
- [utils/evaluate.py](/Users/kbyte/Desktop/afford-motion-main/utils/evaluate.py)

当前 evaluator 已经支持：
- 从 `[N, 24]` 恢复成 `[4, N, 6]`
- 保存为 `[num_samples, 4, N, 6]`

额外有一个 shape smoke test：
- [scripts/check_temporal_affordance_shapes.py](/Users/kbyte/Desktop/afford-motion-main/scripts/check_temporal_affordance_shapes.py)

## 5.3 第二阶段 AMDM / CMDM

### 当前目标
第二阶段输入已经从静态 affordance 改成 temporal affordance sequence。

### 当前输入形式
训练/测试时，第二阶段主链路读取的 contact 条件是：
- 单样本训练：`[B, 4, N, 6]`
- 评测时多 sample 输入：`[B, K_samples, 4, N, 6]`

其中：
- `test.py` 会在采样时取 `[:, k, ...]` 给 `CMDM`

### 数据集代码
- 通用多数据集版本：
  - [datasets/motionx.py](/Users/kbyte/Desktop/afford-motion-main/datasets/motionx.py)
  - 关键类：`ContactMotionDataset`
- HumanML3D 单独版本：
  - [datasets/humanml3d.py](/Users/kbyte/Desktop/afford-motion-main/datasets/humanml3d.py)
  - 关键类：`ContactMotionHumanML3DDataset`

当前已支持：
- 训练时读取 GT `dist_seq`
- 测试时读取 ADM 产出的 `[num_samples, 4, N, 6]`
- mixed training 时读取 `[4, N, 6]` 的预测 affordance

### 模型代码
- [models/cmdm.py](/Users/kbyte/Desktop/afford-motion-main/models/cmdm.py)

当前 `CMDM` 的 temporal affordance 逻辑是：
1. 每个 phase 用共享 `SceneMapEncoder` 单独编码
2. 加 phase embedding
3. 把 phase tokens flatten 成一串 memory / condition tokens
4. 喂给 `trans_enc`

### 当前结构边界
- **已实现**：`trans_enc` + temporal affordance
- **未实现**：`trans_dec` 下的 temporal affordance MVP
  - 代码里对 `trans_dec` 的 temporal 路径显式 `NotImplemented`

### 第二阶段新增几何损失（Idea 5）
`CMDM` 当前还实现了：
- `set_normalization_stats()`
- `denormalize_motion()`
- `compute_aux_losses()`

几何损失逻辑：
- `contact loss`
  - 从 GT motion 估计 soft contact weight
  - 约束预测接触关节靠近 scene points
- `penetration loss`
  - 预测 joints -> `JointsToSMPLX`
  - 得到 SMPL-X verts
  - 对采样 scene points 做 signed distance

相关工具文件：
- [utils/geometry_loss.py](/Users/kbyte/Desktop/afford-motion-main/utils/geometry_loss.py)
- [utils/joints_to_smplx.py](/Users/kbyte/Desktop/afford-motion-main/utils/joints_to_smplx.py)
- [utils/misc.py](/Users/kbyte/Desktop/afford-motion-main/utils/misc.py)

## 5.4 训练流程

### 训练主入口
- [train.py](/Users/kbyte/Desktop/afford-motion-main/train.py)
- [train_ddp.py](/Users/kbyte/Desktop/afford-motion-main/train_ddp.py)

当前训练时的几个关键点：
1. 先通过 `compute_model_input_dimension()` 计算当前模型真实输入维度
2. 构建 dataset / dataloader
3. 构建 model / diffusion
4. 如果模型实现了 `set_normalization_stats()`，则把 dataset 的 `mean/std` 注入模型
   - 当前主要是 `CMDM` 用它做几何 loss 的反归一化

训练 loop 在：
- [utils/training.py](/Users/kbyte/Desktop/afford-motion-main/utils/training.py)

关键行为：
- 只把 `x` 和 `c_*` 条件字段送给 diffusion / model
- 记录并保存各项 `terms`
- 保存 ckpt 时会跳过：
  - `scene_model`
  - `clip_model`
  - `text_model`
  - `bert_model`
  - `joints_to_smplx_model`
  - `smplx_body_model`

## 5.5 diffusion loss 入口
- [diffusion/gaussian_diffusion.py](/Users/kbyte/Desktop/afford-motion-main/diffusion/gaussian_diffusion.py)

当前状态：
- baseline 的 MSE 仍然保留
- 现在会恢复 `pred_xstart`
- 如果模型实现了 `compute_aux_losses()`，就把额外 loss 合入：
  - `contact_loss`
  - `penetration_loss`
  - `aux_loss`

因此当前第二阶段总 loss 实际上是：
- `loss = mse (+ vb) + aux_loss`

## 5.6 推理 / evaluate 流程

统一测试入口：
- [test.py](/Users/kbyte/Desktop/afford-motion-main/test.py)

评测和保存逻辑在：
- [utils/evaluate.py](/Users/kbyte/Desktop/afford-motion-main/utils/evaluate.py)

当前已知主链路：
1. 第一阶段测试
   - 保存 `pred_contact/*.npy`
   - Temporal 情况下保存形状是 `[num_samples, 4, N, 6]`
2. 第二阶段测试
   - 读取 `pred_contact`
   - 生成 motion
   - 保存 motion / metrics

### 已知边界
- **已实现**：训练主链路 + 第一阶段评测 + 第二阶段主评测链路
- **部分实现 / 待确认**：
  - `ContactMotionExampleDataset` 在 [datasets/motionx.py](/Users/kbyte/Desktop/afford-motion-main/datasets/motionx.py) 中仍然按 `contact.npy = xyz + dist` 的平面格式读取，未见明确 temporal 序列适配
  - custom/example 采样支线未作为这次改造的重点验证对象

## 6. 关键文件作用速查

### 6.1 说明与状态
- [baseline_summary.md](/Users/kbyte/Desktop/afford-motion-main/baseline_summary.md)：原始 baseline 与 Idea 1 动机说明
- [idea1_spec.md](/Users/kbyte/Desktop/afford-motion-main/idea1_spec.md)：Idea 1 需求说明
- [current_status.md](/Users/kbyte/Desktop/afford-motion-main/current_status.md)：当前阶段已做改动的摘要
- [idea5_spec.md](/Users/kbyte/Desktop/afford-motion-main/idea5_spec.md)：Idea 5 需求说明

### 6.2 预处理
- [prepare/generate_contact_data.py](/Users/kbyte/Desktop/afford-motion-main/prepare/generate_contact_data.py)：GT affordance / temporal affordance 生成核心

### 6.3 数据集
- [datasets/motionx.py](/Users/kbyte/Desktop/afford-motion-main/datasets/motionx.py)：通用多数据集版本
- [datasets/humanml3d.py](/Users/kbyte/Desktop/afford-motion-main/datasets/humanml3d.py)：HumanML3D 单独版本

### 6.4 模型
- [models/cdm.py](/Users/kbyte/Desktop/afford-motion-main/models/cdm.py)：第一阶段 ADM / CDM
- [models/cmdm.py](/Users/kbyte/Desktop/afford-motion-main/models/cmdm.py)：第二阶段 AMDM / CMDM + Idea 5 loss

### 6.5 几何与 SMPL-X 工具
- [utils/geometry_loss.py](/Users/kbyte/Desktop/afford-motion-main/utils/geometry_loss.py)：Idea 5 训练期几何工具
- [utils/joints_to_smplx.py](/Users/kbyte/Desktop/afford-motion-main/utils/joints_to_smplx.py)：joints -> SMPL-X 参数近似器
- [utils/misc.py](/Users/kbyte/Desktop/afford-motion-main/utils/misc.py)：输入维度计算、SMPL-X mesh/joints 工具

### 6.6 训练 / 测试 / 评测
- [train.py](/Users/kbyte/Desktop/afford-motion-main/train.py)
- [train_ddp.py](/Users/kbyte/Desktop/afford-motion-main/train_ddp.py)
- [test.py](/Users/kbyte/Desktop/afford-motion-main/test.py)
- [utils/training.py](/Users/kbyte/Desktop/afford-motion-main/utils/training.py)
- [utils/evaluate.py](/Users/kbyte/Desktop/afford-motion-main/utils/evaluate.py)

### 6.7 配置
- [configs/model/cdm.yaml](/Users/kbyte/Desktop/afford-motion-main/configs/model/cdm.yaml)
- [configs/model/cmdm.yaml](/Users/kbyte/Desktop/afford-motion-main/configs/model/cmdm.yaml)
- [configs/task/contact_gen.yaml](/Users/kbyte/Desktop/afford-motion-main/configs/task/contact_gen.yaml)
- [configs/task/text_to_motion_contact_gen.yaml](/Users/kbyte/Desktop/afford-motion-main/configs/task/text_to_motion_contact_gen.yaml)
- [configs/task/contact_motion_gen.yaml](/Users/kbyte/Desktop/afford-motion-main/configs/task/contact_motion_gen.yaml)
- [configs/task/text_to_motion_contact_motion_gen.yaml](/Users/kbyte/Desktop/afford-motion-main/configs/task/text_to_motion_contact_motion_gen.yaml)

## 7. 已实现 / 部分实现 / 仅想法

### 7.1 已实现
1. Idea 1 MVP
   - `K=4`
   - 均匀分段
   - `dist_seq` 预处理
   - ADM 展平训练
   - ADM 输出恢复为 `[num_samples, 4, N, 6]`
   - AMDM 接收 temporal affordance
2. Idea 5 MVP
   - `contact loss`
   - `penetration loss`
   - diffusion 训练已接线
3. shape / 语法层面的最小检查
   - [scripts/check_temporal_affordance_shapes.py](/Users/kbyte/Desktop/afford-motion-main/scripts/check_temporal_affordance_shapes.py)

### 7.2 部分实现
1. Temporal affordance 只打通了 MVP 路径
   - `trans_enc` 已支持
   - `trans_dec` 未支持
2. Idea 5 当前是“训练 loss 已接线”，但默认配置仍关闭
   - `model.geometry_loss.enable: false`
3. example/custom/sample 支线不是这轮改造重点
   - 主训练测试链路已改
   - example/custom 路径需额外确认

### 7.3 仅想法 / 未实现
1. Idea 1 的 learned segmentation / language-driven segmentation / contact-driven segmentation
2. Idea 5 的 foot sliding loss
3. 更复杂的 mesh SDF / 更精确物理约束
4. 端到端联合训练

## 8. 当前配置默认值的真实含义

这一点非常重要：**当前代码虽然已经实现了 Idea 1 和 Idea 5 的主逻辑，但默认配置并不会自动启用它们。**

### 默认关闭的开关
在 task config 中：
- `task.dataset.temporal_affordance: false`
- `task.dataset.num_phases: 4`

在 model config 中：
- `model.geometry_loss.enable: false`

因此如果直接按旧脚本原样跑，得到的仍然更接近 baseline 行为，而不是当前改造版行为。

## 9. 现在已经改到哪一步

### 可以认为已经完成的工程节点
1. Idea 1 的主训练/评测链路已经接通
2. Idea 5 的第二阶段训练 loss 已经接通
3. 配置层面已预留开关
4. 基础 smoke check 脚本已存在

### 还没有完成到“实验结论稳定”的部分
1. 没有在代码中看到完整实验结果归档
2. 没有看到完整训练 run 的产物目录被纳入仓库
3. 默认脚本仍偏 baseline，需要手动加 override
4. small-subset 实验支持没有专门做成配置开关

## 10. 后续建议从哪里继续接

### 10.1 如果目标是继续跑实验
优先顺序建议如下：

1. **先确认真实数据目录与依赖环境**
   - 当前工作区没有包含 `data/` 目录内容
   - 运行前需确认本机外部数据已经就位

2. **先跑 HumanML3D 的 short-run**
   - 先验证 `temporal_affordance=true`
   - 第二阶段再打开 `model.geometry_loss.enable=true`

3. **确保使用显式 override，而不是直接用旧脚本**
   必须显式打开：
   - `task.dataset.temporal_affordance=true`
   - `task.dataset.num_phases=4`
   - `task.dataset.mix_train_ratio=0.0`（建议做 Idea 5 首轮实验时先设 0）
   - `model.geometry_loss.enable=true`

4. **先做小数据集 / 少步数对比**
   当前训练按 `max_steps` 驱动，真正省时间要同时缩小：
   - 训练集规模
   - `max_steps`

### 10.2 如果目标是继续补代码
推荐优先级：

1. **把 HumanML3D 的小数据集子集支持做成正式配置**
   - 当前 HumanML3D 专用数据集没有像通用 `MotionXDataset` 那样直接暴露 `ratio`
   - 现在做小数据集实验主要靠替换 `train.txt`

2. **补 temporal affordance 的 example/custom/sample 路径**
   - 重点看 [datasets/motionx.py](/Users/kbyte/Desktop/afford-motion-main/datasets/motionx.py) 里的 `ContactMotionExampleDataset`

3. **给 Idea 5 增加更明确的 smoke test**
   - 当前有 shape smoke test
   - 但缺少真正的 model forward / loss 数值 smoke test

4. **视需要再补 `trans_dec` 或 foot sliding**
   - 这两项都不应早于主实验验证

## 11. 运行接手时的风险提示

1. **最容易误判的地方**
   - 看到代码里有 Idea 1 / Idea 5 逻辑，就误以为默认运行已经启用
   - 实际上默认配置仍是关闭状态

2. **最容易出 shape 问题的地方**
   - 第一阶段 flatten / recover 的 `[4, N, 6] <-> [N, 24]`
   - 第二阶段测试时 `c_pc_contact[:, k, ...]`

3. **最容易出数值/速度问题的地方**
   - `penetration loss` 依赖 `JointsToSMPLX + signed distance`
   - 第二阶段训练速度会明显慢于 baseline

4. **最容易出 train-test gap 的地方**
   - `mix_train_ratio`
   - 如果第一轮只是验证方向，建议先设为 `0.0`

## 12. 一句话总结
当前项目已经从原始的“静态 affordance 两阶段框架”改成了：

**“Temporal Affordance Sequence + Geometry-aware AMDM Loss”的两阶段框架。**

其中：
- Idea 1 的主链路已经接通
- Idea 5 的 MVP 训练损失已经接通
- 默认配置仍需显式打开
- 主实验链路可继续推进
- example/custom 支线与更复杂物理项仍可视为后续工作
