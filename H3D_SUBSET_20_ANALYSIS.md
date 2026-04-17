# HumanML3D（H3D）抽取 20% 子集说明

## 1. 文档目的
这份文档专门回答一个问题：

> 如果想从 HumanML3D / `H3D` 中抽出 20% 训练子集，用来做 baseline 与改进版模型的快速对比实验，应该改哪一部分？

本文档基于当前项目代码真实读取逻辑整理，不是基于猜测。

---

## 2. 先说结论

**最推荐的做法不是去裁剪整个 `H3D` 文件夹，也不是手动删 `contacts/`、`new_joint_vecs/`、`texts/` 里的文件。**

当前代码下，**最稳、最省事、最不容易做歪实验**的方法是：

1. 保留 `H3D` 目录里的原始数据文件不动
2. 只从 `train.txt` 里抽出 20% 样本，生成一个新的 `train_20.txt`
3. 做实验时让训练代码读这份 20% 的训练列表
4. `test.txt` 保持不变

也就是说：

- **要改的核心是 split 文件**
- **不是数据文件本体**

---

## 3. 你截图里的 H3D 目录，大致哪些东西有用

从当前代码看，HumanML3D 主流程主要依赖这些内容：

- `contacts/`
  - 第一阶段 GT affordance
  - 第二阶段 GT / test-time affordance 条件
- `new_joint_vecs/`
  - motion 向量
- `texts/`
  - 文本描述
- `train.txt`
  - 训练样本 ID 列表
- `test.txt`
  - 测试样本 ID 列表
- `all.txt`
  - 统计 / 全量遍历时用
- `Mean.npy`
  - motion 归一化均值
- `Std.npy`
  - motion 归一化方差
- `pred_contact/`
  - 第二阶段 mixed training / test-time 读取 ADM 生成的 affordance

你截图里这些文件：
- `val.txt`
- `train_val.txt`
- `texts_enhanced/`
- `question.txt`
- `test_single.txt`

从**当前主训练链路**来看，不是核心依赖项，至少 HumanML3D 的 ADM / AMDM 主流程没有直接依赖它们。

---

## 4. 当前代码到底读了哪些 split 文件

### 4.1 通用 motion 数据集 `HumanML3DDataset`
代码位置：
- [datasets/humanml3d.py](/Users/kbyte/Desktop/afford-motion-main/datasets/humanml3d.py)

关键逻辑：
- `split_file = os.path.join(self.data_dir, 'H3D', f'{self.phase}.txt')`

也就是说：
- `phase='train'` -> 读 `H3D/train.txt`
- `phase='test'` -> 读 `H3D/test.txt`
- `phase='all'` -> 读 `H3D/all.txt`

对应代码：
- [datasets/humanml3d.py](/Users/kbyte/Desktop/afford-motion-main/datasets/humanml3d.py:53)

这个类本身支持一个 `ratio` 参数：
- [datasets/humanml3d.py](/Users/kbyte/Desktop/afford-motion-main/datasets/humanml3d.py:35)
- [datasets/humanml3d.py](/Users/kbyte/Desktop/afford-motion-main/datasets/humanml3d.py:56)

但要注意：**你现在训练 ADM / AMDM 用的并不是这个基础类。**

---

### 4.2 第一阶段 HumanML3D 数据集 `ContactHumanML3DDataset`
关键逻辑：
- 读 `H3D/{phase}.txt`
- 训练通常读 `H3D/train.txt`
- 测试读 `H3D/test.txt`

对应代码：
- [datasets/humanml3d.py](/Users/kbyte/Desktop/afford-motion-main/datasets/humanml3d.py:356)

这个类**没有 `ratio` 开关**，所以它不会像基础 `HumanML3DDataset` 那样自动按概率抽样训练集。

---

### 4.3 第二阶段 HumanML3D 数据集 `ContactMotionHumanML3DDataset`
关键逻辑同样是：
- 读 `H3D/{phase}.txt`

对应代码：
- [datasets/humanml3d.py](/Users/kbyte/Desktop/afford-motion-main/datasets/humanml3d.py:660)

这个类同样**没有 `ratio` 开关**。

---

### 4.4 `all.txt` 的用途
在第一阶段的 `ContactHumanML3DDataset` 里，`all.txt` 主要用于：
- 计算 / 缓存 contact mean/std

对应代码：
- [datasets/humanml3d.py](/Users/kbyte/Desktop/afford-motion-main/datasets/humanml3d.py:447)

所以：

- **训练子集实验不建议去改 `all.txt`**
- 否则会影响统计文件的含义，增加额外变量

---

## 5. 如果你只想做 20% 训练子集实验，最推荐改哪一部分

## 5.1 推荐做法：只改 `train.txt`

最推荐的方式是：

1. 保留完整的：
   - `contacts/`
   - `new_joint_vecs/`
   - `texts/`
   - `test.txt`
   - `all.txt`
   - `Mean.npy`
   - `Std.npy`

2. 单独从 `train.txt` 中抽 20%，生成一份：
   - `train_20.txt`

3. 训练实验时：
   - 临时让 `train.txt` 指向这 20% 子集
   - 或者让代码读取 `train_20.txt`

### 为什么这是最好的方式
因为当前代码是“先按 split 文件拿样本 ID，再去各目录读取对应文件”。

也就是说，只要 `train.txt` 里没有某个 ID：
- 它对应的 `contacts/*.npz`
- `new_joint_vecs/*.npy`
- `texts/*.txt`

就不会被训练集访问到。

所以没有必要把这些物理文件删掉。

---

## 5.2 不建议直接删目录文件

不建议你去做这些事：

- 从 `contacts/` 里删 80%
- 从 `new_joint_vecs/` 里删 80%
- 从 `texts/` 里删 80%

原因：

1. 容易把 test 样本也删掉
2. 容易破坏 `all.txt` 和已有统计文件的对应关系
3. 以后恢复完整数据集更麻烦
4. 第二阶段 test / evaluate / pred_contact 也更容易混乱

---

## 6. 实际上“需要改哪一部分”

### 方案 A：零代码方案
这是最推荐的。

你不改任何 Python 代码，只改 split 文件：

1. 备份原始训练列表
   - `train.txt -> train_full.txt`
2. 生成 20% 子集
   - `train_20.txt`
3. 做实验时：
   - 把 `train_20.txt` 复制覆盖成 `train.txt`
4. 跑完实验后再恢复：
   - `train_full.txt -> train.txt`

### 这个方案需要动的部分
- **只动 `H3D/train.txt`**

### 这个方案不需要动的部分
- `test.txt`
- `all.txt`
- `contacts/`
- `new_joint_vecs/`
- `texts/`
- 训练代码本身

---

### 方案 B：轻量代码方案
如果你不想每次手动覆盖 `train.txt`，可以让代码支持读取自定义 split 文件。

这时需要改的地方主要是：

#### 第一阶段 HumanML3D 数据集
- [datasets/humanml3d.py](/Users/kbyte/Desktop/afford-motion-main/datasets/humanml3d.py)
  - `ContactHumanML3DDataset._load_datasets()`

当前代码：
- 固定读 `os.path.join(self.data_dir, 'H3D', f'{self.phase}.txt')`

可改成：
- 优先读 `cfg.split_file`
- 没有时再回退到默认 `train.txt / test.txt / all.txt`

#### 第二阶段 HumanML3D 数据集
- [datasets/humanml3d.py](/Users/kbyte/Desktop/afford-motion-main/datasets/humanml3d.py)
  - `ContactMotionHumanML3DDataset._load_datasets()`

同样改法：
- 增加 `split_file` 支持

#### 如果你还会用基础 HumanML3D dataset
- [datasets/humanml3d.py](/Users/kbyte/Desktop/afford-motion-main/datasets/humanml3d.py)
  - `HumanML3DDataset._load_datasets()`

但如果你当前主要跑的是：
- 第一阶段 ADM
- 第二阶段 AMDM

那最关键的还是前两个类。

---

## 7. 哪些地方不要改

### 7.1 不建议改 `test.txt`
如果你要做 baseline vs improved 的公平对比：

- 训练集可以缩成 20%
- **测试集必须保持一致**

否则结果没有可比性。

---

### 7.2 不建议改 `all.txt`
原因：
- 它被第一阶段数据集用来统计 mean/std
- 改掉以后会把“训练子集影响”与“统计分布变化”混在一起

如果你的目标是快速公平对比，建议：
- `all.txt` 保持原样
- 只改训练集样本列表

---

### 7.3 不建议改 `Mean.npy` / `Std.npy`
第二阶段 motion normalization 依赖它们：
- [datasets/humanml3d.py](/Users/kbyte/Desktop/afford-motion-main/datasets/humanml3d.py:117)
- [datasets/humanml3d.py](/Users/kbyte/Desktop/afford-motion-main/datasets/humanml3d.py:741)

做 20% 子集实验时，建议保持：
- `Mean.npy`
- `Std.npy`

不变。

---

## 8. 20% 子集实验时，第一阶段和第二阶段分别会受什么影响

### 8.1 第一阶段 ADM
第一阶段训练样本会减少到 `train_20.txt` 中出现的那些 HumanML3D ID。

但注意：
- `ContactHumanML3DDataset` 还会基于文本片段切出额外的子 motion
- 所以最终样本数并不是 “ID 数量 × 1”

换句话说：
- 你从 `train.txt` 中抽 20% 的 ID
- 最终训练样本条数仍然会大于这个 ID 数量

这很正常。

---

### 8.2 第二阶段 AMDM
第二阶段同样会只读取 `train_20.txt` 里的样本 ID。

此外它还会在训练时读：
- `contacts/`
- `pred_contact/`（如果 `mix_train_ratio > 0`）

因此如果你要做最干净的小子集实验，建议：
- `task.dataset.mix_train_ratio = 0.0`

这样第二阶段只用 GT affordance，不引入额外变量。

---

## 9. 如果想做公平对比实验，最推荐的流程

### 9.1 baseline vs improved 的公平比较
推荐：

1. 从 `train.txt` 抽一份固定的 `train_20.txt`
2. baseline 和 improved 都使用这同一份 `train_20.txt`
3. 两边都保持：
   - 相同 `test.txt`
   - 相同 `max_steps`
   - 相同 `batch_size`
   - 相同随机种子

### 9.2 两阶段都要用同一份子集
因为你是两阶段方法，所以对比时应该是：

1. baseline ADM：用 20% 子集训练
2. baseline AMDM：也用同一份 20% 子集训练
3. improved ADM：用同一份 20% 子集训练
4. improved AMDM：也用同一份 20% 子集训练

不要出现：
- 第一阶段用 20%
- 第二阶段又用 full train

否则结果不公平。

---

## 10. 当前代码下最推荐的落地方案

### 最推荐
**不改代码，只改 `train.txt`。**

你现在最值得做的是：

1. 备份：
   - `train.txt -> train_full.txt`
2. 生成：
   - `train_20.txt`
3. 做实验时：
   - 用 `train_20.txt` 覆盖 `train.txt`
4. 跑 baseline 和 improved
5. 跑完再恢复 `train_full.txt`

### 为什么这是当前最合适的
因为：
- 当前 HumanML3D 专用数据集没有现成的 `ratio` 配置
- 但 split 文件驱动的数据读取逻辑非常清晰
- 改 split 文件最稳，也最不容易把目录搞乱

---

## 11. 如果后面想把这件事做成正式功能，建议改哪里

如果你以后想长期做：
- `10% / 20% / 50%` 子集实验
- 不想每次手动覆盖 `train.txt`

建议做成正式配置：

### 建议新增的配置项
- `task.dataset.split_file`

### 需要改的代码位置
1. [datasets/humanml3d.py](/Users/kbyte/Desktop/afford-motion-main/datasets/humanml3d.py)
   - `ContactHumanML3DDataset._load_datasets()`
2. [datasets/humanml3d.py](/Users/kbyte/Desktop/afford-motion-main/datasets/humanml3d.py)
   - `ContactMotionHumanML3DDataset._load_datasets()`
3. 视需要再补：
   - `HumanML3DDataset._load_datasets()`

### 推荐逻辑
- 如果 `cfg.split_file` 非空：
  - 直接读这份文件
- 否则：
  - 继续读 `H3D/{phase}.txt`

这样就能非常方便地切换：
- `train.txt`
- `train_20.txt`
- `train_10.txt`

而且不需要再手动覆盖原文件。

---

## 12. 一句话总结

如果你只是想从 HumanML3D / `H3D` 里抽 20% 子集做快速实验：

**最该改的是 `train.txt` 这份 split 文件，而不是 `H3D` 下面的原始数据目录。**

对当前代码来说：
- `train.txt` 决定训练集样本范围
- `test.txt` 决定测试集
- `all.txt` 用于统计，不建议动
- `contacts/`、`new_joint_vecs/`、`texts/` 保持完整即可

所以当前最推荐的做法是：

**保留完整数据目录，只生成一份 `train_20.txt`，并让训练阶段使用它。**
