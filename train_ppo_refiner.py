# import os
# import hydra
# import torch
# import random
# import numpy as np
# from omegaconf import DictConfig, OmegaConf
# from loguru import logger

# from datasets.base import create_dataset
# from datasets.misc import collate_fn_general
# from models.base import create_model_and_diffusion
# from models.ppo_refiner import PPORefiner
# from utils.io import mkdir_if_not_exists, Board
# from utils.misc import compute_repr_dimesion
# from utils.ppo_utils import compute_gae, ppo_update
# from utils.rl_reward import compute_total_reward

# # 确保CMDMM模型被导入并注册
# import models.cmdm
# # ---------------------------------------------------------------------

# # Configurable constants (tune these)
# BATCH_MOTION_KEYS = ['x', 'motion', 'gt_motion', 'target']  # candidate keys to find GT motion in dataset batch
# NOISE_STD = 0.05
# DEFAULT_PPO_HIDDEN = 1024
# # ---------------------------------------------------------------------

# @hydra.main(version_base=None, config_path="./configs", config_name="default")
# def main(cfg: DictConfig):
#     # 根据数据表示计算输入特征维度
#     cfg.model.input_feats = compute_repr_dimesion(cfg.model.data_repr)
    
#     # 初始化设备、目录和日志
#     device = f"cuda:{cfg.gpu}" if cfg.gpu is not None else "cpu"
#     mkdir_if_not_exists(cfg.log_dir)
#     mkdir_if_not_exists(cfg.ckpt_dir)
#     mkdir_if_not_exists(cfg.eval_dir)
#     logger.add(cfg.log_dir + '/ppo_refiner_runtime.log')
#     Board().create_board(cfg.platform, project=cfg.project + "_ppo", log_dir=cfg.log_dir)
#     logger.info('[PPO] Config\n' + OmegaConf.to_yaml(cfg))

#     # dataset
#     train_dataset = create_dataset(cfg.task.dataset, cfg.task.train.phase, gpu=cfg.gpu)
#     train_loader = train_dataset.get_dataloader(
#         batch_size=cfg.task.train.batch_size,
#         collate_fn=collate_fn_general,
#         num_workers=cfg.task.train.num_workers,
#         pin_memory=True,
#         shuffle=True,
#     )

#     # 加载预训练的 CMDM 模型 (冻结参数)
#     model, diffusion = create_model_and_diffusion(cfg, device=device)
#     model.to(device)
#     model.eval()
#     for p in model.parameters():
#         p.requires_grad = False
    
#     # -------------------------------------------------------------
#     # 关键部分：获取运动维度和序列长度。
#     # -------------------------------------------------------------
#     sample = next(iter(train_loader))
#     motion_key = None
#     for k in BATCH_MOTION_KEYS:
#         if k in sample:
#             motion_key = k
#             break
#     if motion_key is None:
#         for k, v in sample.items():
#             if hasattr(v, 'ndim') and v.ndim == 3:
#                 motion_key = k
#                 break
#     if motion_key is None:
#         raise RuntimeError("Cannot detect motion key in dataset batch. Please set BATCH_MOTION_KEYS properly.")

#     batch_motion = sample[motion_key]  # tensor [B, seq_len, motion_dim]
#     seq_len = batch_motion.shape[1]
    
#     # 获取 PPO 模型的运动维度，确保与 CMDM 模型的输出维度一致 (66)
#     motion_dim = cfg.model.input_feats
    
#     # 获取 GT 运动的完整维度，用于奖励计算
#     gt_motion_full_dim = batch_motion.shape[2]
    
#     logger.info(f"[PPO] Detected motion key '{motion_key}', seq_len={seq_len}, "
#                 f"motion_dim (PPO/CMDM)={motion_dim}, "
#                 f"GT motion_dim (full)={gt_motion_full_dim}")

#     # 创建 PPO Refiner 模型
#     # ppo_model的输入维度现在也是66，与obs_t匹配
#     ppo_model = PPORefiner(seq_len=seq_len, motion_dim=motion_dim, hidden_dim=DEFAULT_PPO_HIDDEN).to(device)
#     optimizer = torch.optim.Adam(ppo_model.parameters(), lr=cfg.get('ppo_lr', 1e-5))
    
#     # 训练循环
#     epochs = cfg.get('ppo_epochs', 50)
#     steps_per_epoch = cfg.get('ppo_steps_per_epoch', 1024)
#     minibatch = cfg.get('ppo_minibatch', 64)
#     clip_eps = cfg.get('ppo_clip', 0.2)
    
#     global_step = 0
#     for epoch in range(epochs):
#         obs_buf, act_buf, rew_buf, val_buf, logp_buf = [], [], [], [], []
#         processed = 0
        
#         data_iterator = iter(train_loader)
#         while processed < steps_per_epoch:
#             try:
#                 batch = next(data_iterator)
#             except StopIteration:
#                 data_iterator = iter(train_loader)
#                 batch = next(data_iterator)

#             gt_motion_full = batch[motion_key].to(device)
#             B = gt_motion_full.shape[0]

#             with torch.no_grad():
#                 model_kwargs = {
#                     'c_text': batch.get('text', None),
#                     'c_pc_xyz': batch.get('c_pc_xyz', None).to(device) if 'c_pc_xyz' in batch else None,
#                     'c_pc_contact': batch.get('c_pc_contact', None).to(device) if 'c_pc_contact' in batch else None,
#                     'x_mask': torch.zeros(B, seq_len, dtype=torch.bool, device=device)
#                 }
                
#                 # CMDM 采样生成初步观测值
#                 obs_t = diffusion.p_sample_loop(
#                     model,
#                     (B, seq_len, motion_dim), # 注意这里的维度
#                     clip_denoised=True,
#                     device=device,
#                     model_kwargs=model_kwargs
#                 )
                
#             # PPO Refiner 动作采样
#             with torch.no_grad():
#                 mean, logstd, values = ppo_model(obs_t)
#                 std = logstd.exp()
#                 dist = torch.distributions.Normal(mean, std)
#                 actions = dist.sample()
#                 logp = dist.log_prob(actions).sum(dim=[1,2])
#                 values = values.cpu().numpy()

#             # 将张量移到 CPU 并转为 NumPy 数组，用于奖励计算
#             obs_np = obs_t.cpu().numpy()
#             actions_np = actions.cpu().numpy()
#             gt_np = gt_motion_full.cpu().numpy()
            
#             # **改进点**：对 GT 运动进行切片，确保维度与 PPO 输出匹配
#             # 假设gt_motion_full_dim = 263，而cmdm/ppo的motion_dim = 66
#             # 这里需要根据你的数据表示（representation）来确定正确的切片方式
#             # 比如，如果你只关心关节旋转和平移，这部分通常是前66维
#             gt_np_subset = gt_np[:, :, :motion_dim] 
            
#             # 应用修正：refined = obs + action
#             refined_np = obs_np + actions_np
            
#             # 计算奖励
#             for i in range(B):
#                 # 传入维度匹配的参数
#                 r_total, _ = compute_total_reward(
#                     refined_np[i],
#                     gt_np_subset[i],
#                     contact_info=batch.get('c_pc_contact', None),
#                     refine_phase=True # 传递一个标志，告诉奖励函数现在是修正阶段
#                 )
#                 obs_buf.append(obs_np[i])
#                 act_buf.append(actions_np[i])
#                 rew_buf.append(r_total)
#                 val_buf.append(values[i])
#                 logp_buf.append(logp[i].cpu().numpy())

#             processed += B
#             global_step += B
        
#         # compute returns and advantages
#         returns = compute_gae(rew_buf, val_buf, np.zeros_like(rew_buf), gamma=cfg.get('ppo_gamma', 0.99), lam=cfg.get('ppo_lam', 0.95))
#         obs_b = torch.tensor(np.array(obs_buf)).to(device).float()
#         act_b = torch.tensor(np.array(act_buf)).to(device).float()
#         logp_b = torch.tensor(np.array(logp_buf)).to(device).float()
#         val_b = torch.tensor(np.array(val_buf)).to(device).float()
#         ret_b = torch.tensor(np.array(returns)).to(device).float()
#         adv_b = ret_b - val_b
#         adv_b = (adv_b - adv_b.mean()) / (adv_b.std() + 1e-8)

#         # PPO update
#         pl, vl, ent = ppo_update(ppo_model, optimizer, obs_b, act_b, logp_b, adv_b, ret_b,
#                                  clip_eps=clip_eps,
#                                  value_coef=cfg.get('ppo_value_coef', 0.2),
#                                  ent_coef=cfg.get('ppo_ent_coef', 0.0),
#                                  epochs=cfg.get('ppo_update_epochs', 4),
#                                  minibatch_size=minibatch,
#                                  device=device)

#         logger.info(f"[PPO][Epoch {epoch}] policy_loss={pl:.4f} value_loss={vl:.4f} ent={ent:.4f}")
        
#         # 保存检查点
#         save_path = os.path.join(cfg.ckpt_dir, f"ppo_refiner_epoch{epoch}.pth")
#         torch.save({'model': ppo_model.state_dict(), 'optimizer': optimizer.state_dict()}, save_path)
#         logger.info(f"[PPO] Saved {save_path}")
        
#     Board().close()

# if __name__ == "__main__":
#     main()




import os
import hydra
import torch
import random
import numpy as np
from omegaconf import DictConfig, OmegaConf
from loguru import logger

from datasets.base import create_dataset
from datasets.misc import collate_fn_general
from models.base import create_model_and_diffusion
from models.ppo_refiner import PPORefiner
from utils.io import mkdir_if_not_exists, Board
from utils.misc import compute_repr_dimesion
from utils.ppo_utils import compute_gae, ppo_update
from utils.rl_reward import compute_total_reward

# 确保CMDMM模型被导入并注册
import models.cmdm
# ---------------------------------------------------------------------

# Configurable constants (tune these)
BATCH_MOTION_KEYS = ['x', 'motion', 'gt_motion', 'target']  # candidate keys to find GT motion in dataset batch
NOISE_STD = 0.05
DEFAULT_PPO_HIDDEN = 1024
# ---------------------------------------------------------------------

@hydra.main(version_base=None, config_path="./configs", config_name="default")
def main(cfg: DictConfig):
    # 初始化设备、目录和日志
    device = f"cuda:{cfg.gpu}" if cfg.gpu is not None else "cpu"
    mkdir_if_not_exists(cfg.log_dir)
    mkdir_if_not_exists(cfg.ckpt_dir)
    mkdir_if_not_exists(cfg.eval_dir)
    logger.add(cfg.log_dir + '/ppo_refiner_runtime.log')
    Board().create_board(cfg.platform, project=cfg.project + "_ppo", log_dir=cfg.log_dir)
    logger.info('[PPO] Config\n' + OmegaConf.to_yaml(cfg))

    # dataset - 先创建数据集获取真实的运动维度
    train_dataset = create_dataset(cfg.task.dataset, cfg.task.train.phase, gpu=cfg.gpu)
    train_loader = train_dataset.get_dataloader(
        batch_size=cfg.task.train.batch_size,
        collate_fn=collate_fn_general,
        num_workers=cfg.task.train.num_workers,
        pin_memory=True,
        shuffle=True,
    )

    # 获取运动维度和序列长度
    sample = next(iter(train_loader))
    motion_key = None
    for k in BATCH_MOTION_KEYS:
        if k in sample:
            motion_key = k
            break
    if motion_key is None:
        for k, v in sample.items():
            if hasattr(v, 'ndim') and v.ndim == 3:
                motion_key = k
                break
    if motion_key is None:
        raise RuntimeError("Cannot detect motion key in dataset batch. Please set BATCH_MOTION_KEYS properly.")

    batch_motion = sample[motion_key]  # tensor [B, seq_len, motion_dim]
    seq_len = batch_motion.shape[1]
    
    # **关键修正**：使用数据集中的真实运动维度，而不是cfg.model.input_feats
    full_motion_dim = batch_motion.shape[2]  # 这应该是263
    
    # 现在根据数据表示计算CMDM模型期望的维度
    cfg.model.input_feats = compute_repr_dimesion(cfg.model.data_repr)
    cmdm_motion_dim = cfg.model.input_feats  # 这可能是66或其他值
    
    logger.info(f"[PPO] Detected motion key '{motion_key}', seq_len={seq_len}")
    logger.info(f"[PPO] Full motion_dim from dataset: {full_motion_dim}")
    logger.info(f"[PPO] CMDM motion_dim from config: {cmdm_motion_dim}")
    
    # 决定PPO使用的维度 - 使用完整维度以保持一致性
    ppo_motion_dim = full_motion_dim
    logger.info(f"[PPO] PPO will use motion_dim: {ppo_motion_dim}")
    
    # 加载预训练的 CMDM 模型 (冻结参数)
    # **重要**：临时修改cfg中的input_feats以匹配完整维度
    original_input_feats = cfg.model.input_feats
    cfg.model.input_feats = full_motion_dim
    
    model, diffusion = create_model_and_diffusion(cfg, device=device)
    model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    
    # 恢复原始配置值
    cfg.model.input_feats = original_input_feats
    
    # 创建 PPO Refiner 模型 - 现在使用完整的运动维度
    ppo_model = PPORefiner(seq_len=seq_len, motion_dim=ppo_motion_dim, hidden_dim=DEFAULT_PPO_HIDDEN).to(device)
    optimizer = torch.optim.Adam(ppo_model.parameters(), lr=cfg.get('ppo_lr', 1e-5))
    
    # 训练循环
    epochs = cfg.get('ppo_epochs', 50)
    steps_per_epoch = cfg.get('ppo_steps_per_epoch', 1024)
    minibatch = cfg.get('ppo_minibatch', 64)
    clip_eps = cfg.get('ppo_clip', 0.2)
    
    global_step = 0
    for epoch in range(epochs):
        obs_buf, act_buf, rew_buf, val_buf, logp_buf = [], [], [], [], []
        processed = 0
        
        data_iterator = iter(train_loader)
        while processed < steps_per_epoch:
            try:
                batch = next(data_iterator)
            except StopIteration:
                data_iterator = iter(train_loader)
                batch = next(data_iterator)

            gt_motion_full = batch[motion_key].to(device)
            B = gt_motion_full.shape[0]

            with torch.no_grad():
                model_kwargs = {
                    'c_text': batch.get('text', None),
                    'c_pc_xyz': batch.get('c_pc_xyz', None).to(device) if 'c_pc_xyz' in batch else None,
                    'c_pc_contact': batch.get('c_pc_contact', None).to(device) if 'c_pc_contact' in batch else None,
                    'x_mask': torch.zeros(B, seq_len, dtype=torch.bool, device=device)
                }
                
                # CMDM 采样生成初步观测值 - 现在使用完整维度
                obs_t = diffusion.p_sample_loop(
                    model,
                    (B, seq_len, ppo_motion_dim), # 使用完整维度
                    clip_denoised=True,
                    device=device,
                    model_kwargs=model_kwargs
                )
                
            # PPO Refiner 动作采样
            with torch.no_grad():
                mean, logstd, values = ppo_model(obs_t)
                std = logstd.exp()
                dist = torch.distributions.Normal(mean, std)
                actions = dist.sample()
                logp = dist.log_prob(actions).sum(dim=[1,2])
                values = values.cpu().numpy()

            # 将张量移到 CPU 并转为 NumPy 数组，用于奖励计算
            obs_np = obs_t.cpu().numpy()
            actions_np = actions.cpu().numpy()
            gt_np = gt_motion_full.cpu().numpy()
            
            # 应用修正：refined = obs + action
            refined_np = obs_np + actions_np
            
            # 计算奖励 - 现在维度匹配
            for i in range(B):
                r_total, _ = compute_total_reward(
                    refined_np[i],
                    gt_np[i],  # 现在维度匹配
                    contact_info=batch.get('c_pc_contact', None),
                    refine_phase=True
                )
                obs_buf.append(obs_np[i])
                act_buf.append(actions_np[i])
                rew_buf.append(r_total)
                val_buf.append(values[i])
                logp_buf.append(logp[i].cpu().numpy())

            processed += B
            global_step += B
        
        # compute returns and advantages
        returns = compute_gae(rew_buf, val_buf, np.zeros_like(rew_buf), gamma=cfg.get('ppo_gamma', 0.99), lam=cfg.get('ppo_lam', 0.95))
        obs_b = torch.tensor(np.array(obs_buf)).to(device).float()
        act_b = torch.tensor(np.array(act_buf)).to(device).float()
        logp_b = torch.tensor(np.array(logp_buf)).to(device).float()
        val_b = torch.tensor(np.array(val_buf)).to(device).float()
        ret_b = torch.tensor(np.array(returns)).to(device).float()
        adv_b = ret_b - val_b
        adv_b = (adv_b - adv_b.mean()) / (adv_b.std() + 1e-8)

        # PPO update
        pl, vl, ent = ppo_update(ppo_model, optimizer, obs_b, act_b, logp_b, adv_b, ret_b,
                                 clip_eps=clip_eps,
                                 value_coef=cfg.get('ppo_value_coef', 0.2),
                                 ent_coef=cfg.get('ppo_ent_coef', 0.0),
                                 epochs=cfg.get('ppo_update_epochs', 4),
                                 minibatch_size=minibatch,
                                 device=device)

        logger.info(f"[PPO][Epoch {epoch}] policy_loss={pl:.4f} value_loss={vl:.4f} ent={ent:.4f}")
        
        # 保存检查点时也保存维度信息
        save_path = os.path.join(cfg.ckpt_dir, f"ppo_refiner_epoch{epoch}.pth")
        torch.save({
            'model': ppo_model.state_dict(), 
            'optimizer': optimizer.state_dict(),
            'motion_dim': ppo_motion_dim,
            'seq_len': seq_len
        }, save_path)
        logger.info(f"[PPO] Saved {save_path} with motion_dim={ppo_motion_dim}")
        
    Board().close()

if __name__ == "__main__":
    main()
