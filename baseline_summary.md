# Baseline总结：Afford-Motion

## 一、任务定义
该 baseline 解决的是：**在 3D 场景中，根据文本描述生成合理的人体动作序列**。

输入包括：
- 3D scene point cloud
- 文本描述 text

输出包括：
- human motion sequence

---

## 二、baseline整体框架
baseline 是一个两阶段框架：

### 第一阶段：ADM（Affordance Diffusion Model）
输入：
- scene point cloud
- text

输出：
- affordance map

### 第二阶段：AMDM（Affordance-to-Motion Diffusion Model）
输入：
- affordance map
- text

输出：
- motion sequence

---

## 三、baseline中的affordance是如何构造的
baseline 中的 affordance 不是人工标注的，而是由真实 motion 和真实 scene 自动计算得到。

具体流程如下：

1. 对 motion 的每一帧：
   - 计算 scene 中每个点到 skeleton 中每个 joint 的距离场

2. 将距离场转换成 per-frame affordance map

3. 对整段 motion 的所有帧，在时间维度上做 max-pool

4. 得到整段 motion 对应的一张静态 affordance map

因此，baseline 对于每一条 motion，只使用 **一张静态 affordance map**。

---

## 四、baseline的第一阶段训练逻辑
第一阶段 ADM 的目标是：
- 输入 scene + text
- 预测 affordance map

训练时会使用 GT affordance 作为监督目标。

ADM 当前主要优化的是：
- 预测 affordance 与 GT affordance 的重建误差（MSE）

---

## 五、baseline的第二阶段训练逻辑
第二阶段 AMDM 的目标是：
- 输入 affordance + text
- 生成 motion

训练时主要优化：
- 预测 motion 与 GT motion 的重建误差（MSE）

---

## 六、baseline的核心优点
1. 使用 affordance 作为中间表示，连接 scene grounding 与 motion generation
2. 相比直接做 scene + text -> motion，更容易学习
3. 在泛化到新场景和新描述时有一定优势

---

## 七、baseline当前的主要局限
### 局限1：affordance是静态图
当前 affordance 是对整段 motion 做 max-pool 后得到的一张静态图，会丢失交互的阶段顺序信息。

例如：
- 接近目标
- 调整朝向
- 接触目标
- 稳定/离开

这些过程都被压缩到了同一张图中。



---

## 八、本轮改造目标
本轮只实现 **Idea 1：Temporal Affordance Sequence**。



目标是：
- 将 baseline 中的单张静态 affordance map
- 改为 K-step temporal affordance sequence
- 先验证 temporal affordance 是否有效