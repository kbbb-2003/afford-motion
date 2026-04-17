# Idea 5 需求说明：Contact-Consistent Diffusion Loss

## 一、目标
在当前已经实现 Idea 1（Temporal Affordance Sequence）的项目基础上，继续为第二阶段 AMDM 增加物理一致性训练目标。

本轮目标是：
- 在 AMDM 的原始 MSE motion loss 基础上
- 增加额外的物理一致性 loss
- 提升接触准确性和非穿模能力

---

## 二、核心动机
当前 AMDM 主要通过 MSE 学习 motion reconstruction。

这意味着模型更关注：
- 整体 motion 是否接近 GT

但没有显式约束：
- 接触是否准确
- 是否发生穿模
- 是否满足更合理的场景交互几何关系

因此，本轮希望在训练目标中加入 contact-consistent losses。

---

## 三、本轮实现范围（MVP）
第一版只实现最小可行版本：

### 先实现
1. penetration loss
2. contact loss

### 暂时不实现
1. foot sliding loss
2. 复杂的物理仿真
3. 大规模训练流程重构
4. 对 ADM 的修改

---

## 四、希望修改的位置
本轮改动应主要发生在：
- 第二阶段 AMDM 的训练代码
- 与 motion loss 相关的模块
- 如有必要，可增加少量 geometry utility 函数

本轮不希望大改：
- ADM
- temporal affordance 数据结构
- Idea 1 的主流程

---

## 五、penetration loss 的目标
希望增加一个简单可运行的 penetration loss，用来惩罚预测人体进入场景内部的情况。

请优先结合当前代码条件，选择最容易落地的实现方式，例如：
- 基于 scene geometry 的最近距离 / signed distance / 近似碰撞约束
- 如果项目中已有 scene distance / SDF / nearest neighbor 相关模块，请优先复用

要求：
- 第一版优先可实现、可运行
- 暂时不追求最复杂最精确的碰撞检测

---

## 六、contact loss 的目标
希望增加一个简单的 contact-consistent loss，用来鼓励预测 motion 在关键接触部位更接近目标场景表面。

contact target 的来源可以优先考虑：
1. 从 GT motion 中估计接触关系
2. 或从当前 temporal affordance 中提取 soft contact prior

请优先选择：
- 最容易和当前代码兼容
- 最清晰可解释
- 最适合 MVP 验证的方案

---

## 七、AMDM 总损失的目标形式
希望新的 AMDM loss 形式类似：

L_total = L_mse + lambda_pen * L_penetration + lambda_contact * L_contact

其中：
- L_mse 是当前已有的 motion reconstruction loss
- L_penetration 是新增穿模惩罚
- L_contact 是新增接触约束

第一版先不要加入 foot sliding loss。

---

## 八、实现要求
请遵守以下原则：

1. 先理解当前代码，再改
2. 先输出修改计划，不要直接大改代码
3. 尽量复用当前项目已有的数据结构和 geometry 工具
4. 不要破坏当前 Idea 1 的 temporal affordance 主流程
5. 先做 MVP，优先可运行和可调试
6. 对新增 loss 的输入张量形状、来源、计算方式做清楚说明
7. 如存在多种实现路线，请先列出方案并说明推荐理由

---

## 九、你需要先完成的事情
在正式修改代码前，请先完成：

1. 阅读 current_status.md
2. 阅读本文件 idea5_spec.md
3. 阅读当前项目代码
4. 找到以下内容：
   - 第二阶段 AMDM 的训练脚本
   - 当前 motion loss 定义位置
   - scene geometry 在训练阶段如何访问
   - temporal affordance 当前在 AMDM 中的输入形式
   - 是否已有 mesh / SDF / nearest distance / joint index 工具
5. 输出一个详细修改计划，包括：
   - 需要修改哪些文件
   - 每个文件改什么
   - penetration loss 怎么实现最稳
   - contact loss 怎么实现最稳
   - 哪些地方最容易出 bug
   - 先做哪一步最合适

在我确认方案前，不要直接开始大改代码。