# 当前项目状态说明

## 一、当前项目基础
当前项目最初基于 Afford-Motion baseline。

baseline 原始结构为：
1. ADM：scene + text -> affordance
2. AMDM：affordance + text -> motion

---

## 二、当前已经完成的改动
目前我已经实现了 **Idea 1：Temporal Affordance Sequence**。

当前代码不再使用单张静态 affordance，而是使用时序 affordance sequence。

具体来说：
- 原来：每条 motion 对应 1 张静态 affordance map
- 现在：每条 motion 对应 K 张 phase-wise affordance map
- 当前 MVP 设置为：K = 4，均匀分段
- ADM 当前输出的是 temporal affordance sequence
- AMDM 当前也已经改为接收 temporal affordance sequence

因此，请不要再按原始 static affordance baseline 去理解当前代码。

---

## 三、本轮新目标
我现在准备在已经实现 Idea 1 的基础上，继续实现 **Idea 5：Contact-Consistent Diffusion Loss**。

本轮目标是：
- 只修改第二阶段 AMDM 的训练目标
- 不修改 Idea 1 的基本结构
- 不改 ADM

---

## 四、本轮重点
希望在 AMDM 上增加物理一致性约束，包括：
- penetration loss
- contact loss

第一版先不做：
- foot sliding loss
- 复杂联合训练
- 大规模结构重构

---

## 五、对你的要求
请你先理解当前项目已经实现了 temporal affordance sequence，再分析如何在当前代码上叠加 Idea 5。

不要默认当前代码还是原始 baseline。