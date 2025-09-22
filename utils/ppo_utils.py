# utils/ppo_utils.py
import torch
import torch.nn.functional as F
from torch.distributions import Normal

def compute_gae(rewards, values, masks, gamma=0.99, lam=0.95):
    """
    rewards, values, masks are python lists or 1D tensors per step
    masks: 0 if done else 1  (so multiply next value)
    returns list of returns (target for critic)
    """
    returns = []
    gae = 0.0
    next_value = 0.0
    for step in reversed(range(len(rewards))):
        delta = rewards[step] + gamma * next_value * masks[step] - values[step]
        gae = delta + gamma * lam * gae * masks[step]
        next_value = values[step]
        returns.insert(0, gae + values[step])
    return returns

def ppo_update(model, optimizer, obs_b, act_b, logp_old_b, adv_b, ret_b,
               clip_eps=0.2, value_coef=0.5, ent_coef=0.01, epochs=4, minibatch_size=64, device='cpu'):
    """
    A minibatch PPO update loop. obs_b: [N, seq, dim], act_b same, logp_old_b shape [N], adv_b [N], ret_b [N]
    """
    N = obs_b.shape[0]
    idxs = torch.arange(N)
    mean_loss = 0.0
    mean_vloss = 0.0
    mean_ent = 0.0
    num_updates = 0  # 用于统计实际更新次数

    for _ in range(epochs):
        perm = torch.randperm(N)
        for start in range(0, N, minibatch_size):
            mb_idx = perm[start:start+minibatch_size]
            
            # 将小批量数据移到设备上
            obs_mb = obs_b[mb_idx].to(device)
            act_mb = act_b[mb_idx].to(device)
            logp_old_mb = logp_old_b[mb_idx].to(device)
            adv_mb = adv_b[mb_idx].to(device)
            ret_mb = ret_b[mb_idx].to(device)

            # --- PPO 核心：计算损失 ---
            mean, logstd, values = model(obs_mb)
            std = logstd.exp()
            dist = Normal(mean, std)
            
            # log prob total per sample
            logp = dist.log_prob(act_mb).sum(dim=[1,2])
            entropy = dist.entropy().sum(dim=[1,2]).mean()

            # PPO 策略损失
            ratio = (logp - logp_old_mb).exp()
            surr1 = ratio * adv_mb
            surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * adv_mb
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # 价值网络损失
            value_loss = F.mse_loss(values, ret_mb)

            # 总损失
            loss = policy_loss + value_coef * value_loss - ent_coef * entropy
            
            # --- 改进点：在反向传播前检查损失是否为 NaN ---
            if torch.isnan(loss):
                print(f"Warning: NaN detected in loss, skipping this minibatch. "
                      f"Policy Loss: {policy_loss.item():.4f}, "
                      f"Value Loss: {value_loss.item():.4f}, "
                      f"Entropy: {entropy.item():.4f}")
                continue # 跳过当前小批量，防止 NaN 扩散

            # --- 优化器更新 ---
            optimizer.zero_grad()
            loss.backward()
            
            # 梯度裁剪：防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()

            # 统计损失，用于日志记录
            mean_loss += policy_loss.item()
            mean_vloss += value_loss.item()
            mean_ent += entropy.item()
            num_updates += 1
            
    # 返回平均损失和熵
    return mean_loss / max(1, num_updates), mean_vloss / max(1, num_updates), mean_ent / max(1, num_updates)
