# Idea 1 需求说明：Temporal Affordance Sequence

## 一、目标
在当前 Afford-Motion baseline 的基础上，实现 **Temporal Affordance Sequence**。

当前 baseline 对每条 motion 只构造 **一张静态 affordance map**。  
我的目标是将其改为：

- 一条 motion 对应 **K 张 phase-wise affordance map**
- 形成一个时序 affordance 序列：
  {C^(1), C^(2), ..., C^(K)}

每一张 C^(k) 对应 motion 的一个阶段（phase）。

---

## 二、核心动机
当前 baseline 的 affordance 是对整段 motion 的 per-frame affordance 做时间 max-pool 得到的静态图。

这样做虽然能保留“整段动作中哪些区域曾经重要”，但会丢失：
- 阶段顺序
- 接近过程
- 转向过程
- 接触过程
- 离开过程

对于复杂 HSI（如 walk to chair and sit down），单张静态 affordance 不够表达交互过程。

因此，我希望将静态 affordance 升级为时序 affordance sequence。

---

## 三、本次实现范围（MVP版本）
第一版只实现最小可行版本（MVP）：

### 固定设置
- K = 4
- motion 序列做 **均匀分段**
- 不实现 learned segmentation
- 不实现语言驱动的子动作切分
- 不实现 contact-driven segmentation

---

## 四、GT时序affordance的构造方式
对于每条训练样本：

### 输入
- scene
- text
- GT motion sequence

### 处理流程
1. 将整段 motion 均匀切分成 4 段
2. 对每一段：
   - 按 baseline 原始方式，逐帧计算 scene-point 到 skeleton-joint 的距离场
   - 转换为 per-frame affordance map
   - 仅在该段内部做 max-pool
3. 最终得到 4 张 phase-wise GT affordance：

- C^(1)
- C^(2)
- C^(3)
- C^(4)

也就是说：
- baseline 原来是一张静态 affordance
- 现在变成 4 张按时间阶段划分的 affordance

---

## 五、对ADM的修改要求
当前 ADM 预测一张 affordance map。  
现在需要改为：

- 输入仍然保持 scene + text + noisy affordance 的逻辑
- 输出改为 **4 张 affordance map**

建议输出形式：
- [K, N, J]
或者
- [N, J, K]

具体格式请根据当前代码结构选择最兼容的方案，并在修改计划中说明。

---

## 六、ADM的loss修改
当前 baseline 的第一阶段 loss 是单张 affordance 的 MSE。

现在需要改成 phase-wise MSE：

L_aff = sum_k MSE(C_pred^(k), C_gt^(k))

要求：
- 保持实现清晰
- 优先保证代码可读性
- 暂时不引入额外复杂 regularization

---

## 七、对AMDM的修改要求
当前 AMDM 只接收一张静态 affordance map。  
现在需要改为接收 4 张 phase-wise affordance。

MVP 版本要求：
- 先分别编码每一张 phase affordance
- 再通过一个尽量简单、和当前架构兼容的方式融合
- 推荐使用 phase-aware conditioning / 多token conditioning / 简单 cross-attention 方式
- 不需要过度设计复杂的新结构

目标是：
- 尽量小改动地让 AMDM 能消费 temporal affordance sequence

---

## 八、训练策略要求
本轮先保持 baseline 的两阶段训练思路，不做大规模端到端联合训练。

建议训练顺序：
1. 先训练 ADM（使用 temporal GT affordance）
2. 再训练 AMDM（使用 temporal affordance）
---

## 九、代码修改原则
请严格遵守以下原则：

1. 先理解当前项目代码结构，再修改
2. 尽量减少与本轮目标无关的代码改动
3. 不要静默修改原有训练逻辑
4. 修改后请保留清晰注释
5. 如果某处存在多种实现方案，请先说明再改
6. 优先保证：
   - 可运行
   - 易调试
   - 易检查张量维度
7. 如果可能，请保留 static affordance 的兼容开关，方便做对比实验

---

## 十、需要你先完成的事情
在正式修改代码前，请先完成以下工作，不要直接改代码：

1. 阅读当前项目代码
2. 找到以下内容：
   - affordance 数据构造在哪里
   - 数据集加载逻辑在哪里
   - ADM 模块定义在哪里
   - AMDM 模块定义在哪里
   - 第一阶段训练脚本在哪里
   - 第二阶段训练脚本在哪里
3. 输出一个详细修改计划，包括：
   - 需要修改哪些文件
   - 每个文件大概改什么
   - 张量维度会如何变化
   - 哪些地方最容易出错
   - 你建议的最小可行实现顺序

在我确认方案前，不要直接开始大改代码。

---

## 十一、最终交付要求
正式修改代码后，希望你做到：

1. 每改完一个关键模块，说明：
   - 改了哪个文件
   - 改了什么
   - 为什么这样改

2. 对关键张量形状做说明，例如：
   - static affordance 的原始形状
   - temporal affordance 的新形状
   - ADM 输出形状
   - AMDM 输入形状

3. 如有必要，请增加简单的调试输出或断言，帮助检查 shape 是否正确

4. 如果你认为某个地方设计存在歧义，请先提出再继续实现