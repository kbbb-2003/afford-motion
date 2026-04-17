# HumanML3D 20% 子集实验操作手册

## 1. 目标
这份文档用于快速完成两件事：

1. 生成 `train_20.txt`
2. 在 **同一份 20% 训练子集** 上，分别运行：
   - baseline（原始 static affordance）
   - improved（Idea 1 + Idea 5）

这里默认你要跑的是 **HumanML3D / H3D** 这条线。

---

## 2. 实验原则

为了让 baseline 和 improved 的对比尽量公平，建议保持下面几项一致：

1. 使用同一份 `train_20.txt`
2. 使用同一个 `test.txt`
3. 两边第一阶段用相同步数
4. 两边第二阶段用相同步数
5. 两边 batch size 保持一致
6. 随机种子固定

---

## 3. 当前实验设定建议

这是一套适合 first-pass 对比的设置：

### 第一阶段 ADM
- `max_steps = 20000`

### 第二阶段 AMDM
- `max_steps = 20000`

### baseline
- static affordance
- 不开 geometry loss

### improved
- temporal affordance
- `num_phases = 4`
- 开 Idea 5 geometry loss
- `mix_train_ratio = 0.0`

---

## 4. 数据预处理阶段：哪些要重做，哪些可以沿用

### 4.1 如果你已经有官方预处理好的 H3D 数据
通常**不用**从头重跑整套预处理。

一般可以沿用的内容包括：
- `H3D/new_joint_vecs/`
- `H3D/texts/`
- `H3D/Mean.npy`
- `H3D/Std.npy`
- `H3D/train.txt`
- `H3D/test.txt`
- `H3D/all.txt`

也就是说，下面这些官方预处理步骤通常可以不重新做：

```bash
python prepare/process.py --dataset HumanML3D --data_dir ${YOUR_AMASS_SMPLX_PATH}
python prepare/smplx_to_vec.py --dataset HumanML3D
python prepare/process_scene.py
python prepare/split.py
```

### 4.2 当前代码下建议必须重跑的一步
由于当前代码已经实现了 Idea 1，`contacts/*.npz` 不再只需要 baseline 的静态 `dist`，还需要新的 temporal 字段：

- `dist`
- `dist_seq`

因此建议至少重跑：

```bash
python prepare/generate_contact_data.py --random_segment
```

### 4.3 这一步处理的是整个 H3D，还是只处理 20% 子集？
这一步默认处理的是**整个 H3D**，不是只处理 `train_20.txt` 里的 20%。

这是正常的，原因是：
- `generate_contact_data.py` 的职责是生成全量 `contacts/*.npz`
- 真正决定训练只用哪些样本的是后面的 split 文件：
  - `train.txt`
  - `test.txt`

所以推荐策略是：

1. `generate_contact_data.py` 先全量跑一次，得到完整 `contacts/`
2. 后面训练时再通过 `train_20.txt` 只取 20% 样本

### 4.4 为什么不建议只给 20% 子集单独预处理
因为这样做并不会明显节省后面训练时间，而且会带来额外管理成本：

1. 你以后切回 full training 时又要重新补全 `contacts/`
2. 更容易把 train / test / all 的文件关系搞混
3. 当前代码主逻辑天然是“全量缓存 + 按 split 使用”

### 4.5 建议的最小预处理顺序
如果你已经有现成的 H3D 数据目录，最推荐先做：

```bash
python prepare/generate_contact_data.py --random_segment
python scripts/check_temporal_affordance_shapes.py
```

说明：
- 第一条会重建当前代码所需的 `contacts/*.npz`
- 第二条是一个轻量 shape smoke test，用来确认 temporal affordance 主链路的数据形状没问题

---

## 5. 生成 train_20.txt

## 5.1 先备份原始训练集
在项目根目录执行：

```bash
cp data/H3D/train.txt data/H3D/train_full.txt
```

如果 `train_full.txt` 已经存在，可以先确认内容后再决定是否覆盖。

---

## 5.2 固定随机种子，生成 20% 子集
在项目根目录执行：

```bash
python3 - <<'PY'
import random
from pathlib import Path

seed = 2023
ratio = 0.2

src = Path("data/H3D/train_full.txt")
dst = Path("data/H3D/train_20.txt")

lines = [x.strip() for x in src.read_text().splitlines() if x.strip()]
random.seed(seed)
random.shuffle(lines)

k = max(1, int(len(lines) * ratio))
subset = lines[:k]

dst.write_text("\n".join(subset) + "\n")
print(f"Total train ids: {len(lines)}")
print(f"Subset ids: {len(subset)}")
print(f"Saved to: {dst}")
PY
```

---

## 5.3 开始实验前，把 train_20.txt 作为当前训练集
```bash
cp data/H3D/train_20.txt data/H3D/train.txt
```

注意：
- 这里只改训练集
- **不要改 `test.txt`**
- **不要改 `all.txt`**

---

## 5.4 跑完实验后恢复完整训练集
```bash
cp data/H3D/train_full.txt data/H3D/train.txt
```

---

## 6. baseline（原始 static）完整命令清单

## 6.1 baseline 第一阶段 ADM 训练
```bash
python train.py hydra/job_logging=none hydra/hydra_logging=none \
  exp_name=BASELINE20_ADM_H3D \
  output_dir=outputs \
  platform=TensorBoard \
  diffusion.steps=500 \
  task=text_to_motion_contact_gen \
  task.dataset.sigma=0.8 \
  task.dataset.temporal_affordance=false \
  task.train.batch_size=64 \
  task.train.max_steps=20000 \
  task.train.save_every_step=5000 \
  model=cdm \
  model.arch=Perceiver \
  model.scene_model.use_scene_model=False \
  model.text_model.max_length=20
```

说明：
- 这里明确关闭 `temporal_affordance`
- 这就是“当前代码里的 baseline 模式”

---

## 6.2 baseline 第一阶段测试，预生成 affordance
```bash
python test.py hydra/job_logging=none hydra/hydra_logging=none \
  exp_dir=outputs/BASELINE20_ADM_H3D \
  seed=2023 \
  output_dir=outputs \
  diffusion.steps=500 \
  task=text_to_motion_contact_gen \
  task.dataset.sigma=0.8 \
  task.dataset.temporal_affordance=false \
  task.evaluator.k_samples=0 \
  task.evaluator.eval_nbatch=32 \
  task.evaluator.num_k_samples=128 \
  model=cdm \
  model.arch=Perceiver \
  model.scene_model.use_scene_model=False \
  model.text_model.max_length=20
```

跑完后记下生成的 affordance 目录，例如：

```bash
outputs/BASELINE20_ADM_H3D/eval/test-xxxxxx
```

后面记作：

```bash
BASELINE_AFFORD_DIR=outputs/BASELINE20_ADM_H3D/eval/test-xxxxxx
```

---

## 6.3 baseline 第二阶段 AMDM 训练
```bash
python train.py hydra/job_logging=none hydra/hydra_logging=none \
  exp_name=BASELINE20_AMDM_H3D \
  output_dir=outputs \
  platform=TensorBoard \
  diffusion.steps=1000 \
  task=text_to_motion_contact_motion_gen \
  task.dataset.sigma=0.8 \
  task.dataset.temporal_affordance=false \
  task.dataset.mix_train_ratio=0.0 \
  task.train.batch_size=32 \
  task.train.max_steps=20000 \
  task.train.save_every_step=5000 \
  task.dataset.train_transforms=['RandomEraseLang','RandomEraseContact','NumpyToTensor'] \
  model=cmdm \
  model.arch=trans_enc \
  model.data_repr=h3d \
  model.text_model.max_length=20 \
  model.geometry_loss.enable=false
```

说明：
- 这里显式关掉 `geometry_loss`
- 这样才是和 improved 对比时的 baseline

---

## 6.4 baseline 第二阶段测试，生成 motion
把下面命令中的 `BASELINE_AFFORD_DIR` 替换成真实目录：

```bash
python test.py hydra/job_logging=none hydra/hydra_logging=none \
  exp_dir=outputs/BASELINE20_AMDM_H3D \
  seed=2023 \
  output_dir=outputs \
  diffusion.steps=1000 \
  task=text_to_motion_contact_motion_gen \
  task.dataset.sigma=0.8 \
  task.dataset.temporal_affordance=false \
  task.test.contact_folder=BASELINE_AFFORD_DIR \
  task.evaluator.k_samples=0 \
  task.evaluator.eval_nbatch=32 \
  task.evaluator.num_k_samples=128 \
  model=cmdm \
  model.arch=trans_enc \
  model.data_repr=h3d \
  model.text_model.max_length=20
```

---

## 7. improved（Idea 1 + Idea 5）完整命令清单

## 7.1 improved 第一阶段 ADM 训练
```bash
python train.py hydra/job_logging=none hydra/hydra_logging=none \
  exp_name=IMPROVED20_ADM_H3D \
  output_dir=outputs \
  platform=TensorBoard \
  diffusion.steps=500 \
  task=text_to_motion_contact_gen \
  task.dataset.sigma=0.8 \
  task.dataset.temporal_affordance=true \
  task.dataset.num_phases=4 \
  task.train.batch_size=64 \
  task.train.max_steps=20000 \
  task.train.save_every_step=5000 \
  model=cdm \
  model.arch=Perceiver \
  model.scene_model.use_scene_model=False \
  model.text_model.max_length=20
```

说明：
- 这里打开了 Idea 1
- 第一阶段输出会是 temporal affordance 的展平表示

---

## 7.2 improved 第一阶段测试，预生成 temporal affordance
```bash
python test.py hydra/job_logging=none hydra/hydra_logging=none \
  exp_dir=outputs/IMPROVED20_ADM_H3D \
  seed=2023 \
  output_dir=outputs \
  diffusion.steps=500 \
  task=text_to_motion_contact_gen \
  task.dataset.sigma=0.8 \
  task.dataset.temporal_affordance=true \
  task.dataset.num_phases=4 \
  task.evaluator.k_samples=0 \
  task.evaluator.eval_nbatch=32 \
  task.evaluator.num_k_samples=128 \
  model=cdm \
  model.arch=Perceiver \
  model.scene_model.use_scene_model=False \
  model.text_model.max_length=20
```

跑完后记下生成目录，例如：

```bash
outputs/IMPROVED20_ADM_H3D/eval/test-xxxxxx
```

后面记作：

```bash
IMPROVED_AFFORD_DIR=outputs/IMPROVED20_ADM_H3D/eval/test-xxxxxx
```

---

## 7.3 improved 第二阶段 AMDM 训练
```bash
python train.py hydra/job_logging=none hydra/hydra_logging=none \
  exp_name=IMPROVED20_AMDM_H3D \
  output_dir=outputs \
  platform=TensorBoard \
  diffusion.steps=1000 \
  task=text_to_motion_contact_motion_gen \
  task.dataset.sigma=0.8 \
  task.dataset.temporal_affordance=true \
  task.dataset.num_phases=4 \
  task.dataset.mix_train_ratio=0.0 \
  task.train.batch_size=32 \
  task.train.max_steps=20000 \
  task.train.save_every_step=5000 \
  task.dataset.train_transforms=['RandomEraseLang','RandomEraseContact','NumpyToTensor'] \
  model=cmdm \
  model.arch=trans_enc \
  model.data_repr=h3d \
  model.text_model.max_length=20 \
  model.geometry_loss.enable=true
```

说明：
- 这里同时打开了 Idea 1 和 Idea 5
- `mix_train_ratio=0.0` 是为了让 first-pass 对比更干净

---

## 7.4 improved 第二阶段测试，生成 motion
把下面命令中的 `IMPROVED_AFFORD_DIR` 替换成真实目录：

```bash
python test.py hydra/job_logging=none hydra/hydra_logging=none \
  exp_dir=outputs/IMPROVED20_AMDM_H3D \
  seed=2023 \
  output_dir=outputs \
  diffusion.steps=1000 \
  task=text_to_motion_contact_motion_gen \
  task.dataset.sigma=0.8 \
  task.dataset.temporal_affordance=true \
  task.dataset.num_phases=4 \
  task.test.contact_folder=IMPROVED_AFFORD_DIR \
  task.evaluator.k_samples=0 \
  task.evaluator.eval_nbatch=32 \
  task.evaluator.num_k_samples=128 \
  model=cmdm \
  model.arch=trans_enc \
  model.data_repr=h3d \
  model.text_model.max_length=20
```

---

## 8. 最后应该比较什么

建议至少比较以下几个层面：

### 8.1 第一阶段
- 生成的 affordance 是否合理
- static vs temporal 的可视化差异

### 8.2 第二阶段
- `mse`
- `contact_loss`
- `penetration_loss`
- 采样 motion 的视觉质量

### 8.3 最重要的 qualitative 观察
- improved 是否明显减少穿模
- improved 是否更贴合场景
- improved 是否保持动作自然性

---

## 9. 训练时推荐重点盯的日志

### baseline
- `loss`
- `mse`

### improved
- `loss`
- `mse`
- `contact_loss`
- `penetration_loss`
- `aux_loss`

如果出现这些情况要警惕：

1. `contact_loss` / `penetration_loss` 很大且一直不降
2. `aux_loss` 远大于 `mse`
3. loss 变成 NaN
4. 采样 motion 明显变僵

---

## 10. 推荐的实际执行顺序

建议按下面顺序做：

1. 确认 `data/H3D/` 已经存在基础预处理结果
2. 重跑 `python prepare/generate_contact_data.py --random_segment`
3. 可选跑 `python scripts/check_temporal_affordance_shapes.py`
4. 生成 `train_20.txt`
5. 覆盖 `train.txt`
6. 跑 baseline 第一阶段
7. 跑 baseline 第一阶段测试
8. 跑 baseline 第二阶段
9. 跑 baseline 第二阶段测试
10. 跑 improved 第一阶段
11. 跑 improved 第一阶段测试
12. 跑 improved 第二阶段
13. 跑 improved 第二阶段测试
14. 对比结果
15. 恢复 `train_full.txt`

---

## 11. 一句话总结

如果你现在想最快完成一次 HumanML3D 的 20% 子集对比实验：

1. **先全量重建一次 `contacts/*.npz`，确保里面有 `dist_seq`**
2. **再把 `train.txt` 抽成 `train_20.txt`**
3. **baseline 用 static + 无 geometry loss**
4. **improved 用 temporal affordance + geometry loss**
5. **两边都用相同步数和同一份子集**

这样你就能在尽量省时间的前提下，得到一版比较有意义的方向性结果。
