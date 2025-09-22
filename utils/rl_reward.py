# utils/rl_reward.py
import numpy as np
import torch

def reward_imitation(refined: np.ndarray, target: np.ndarray):
    """负均方误差 (MSE)，值越高越好。"""
    return -float(np.mean((refined - target) ** 2))

def reward_smoothness(refined: np.ndarray):
    """惩罚二阶差分过高的动作，值越高越好。"""
    if refined.shape[0] < 3:
        return 0.0
    vel = np.diff(refined, axis=0)
    acc = np.diff(vel, axis=0)
    return -float(np.mean(acc**2))

def reward_contact_stub(refined: np.ndarray, contact_info):
    """
    接触奖励的占位符。
    - contact_info: 预期的接触标签或可供性图。
    目前返回 0.0。
    """
    return 0.0

# -------------------------------------------------------------
# 核心改进：添加 `refine_phase` 参数以匹配调用方
# -------------------------------------------------------------
def compute_total_reward(refined: np.ndarray, target: np.ndarray, contact_info=None, refine_phase=False, weights=None):
    """
    计算总奖励。

    Args:
        refined (np.ndarray): PPO 修正后的运动，形状为 [seq_len, motion_dim_refined]。
        target (np.ndarray): 原始的真实运动，形状为 [seq_len, motion_dim_full]。
        contact_info: 接触信息。
        refine_phase (bool): 标识当前是否在修正阶段。
        weights (dict): 各项奖励的权重。

    Returns:
        tuple: 总奖励和各项奖励的字典。
    """
    if weights is None:
        weights = {'fidelity': 1.0, 'smooth': 0.5, 'contact': 1.0}

    # -------------------------------------------------------------
    # 维度处理：这是防止 nan 的关键步骤。
    # 确保用于模仿奖励的 target 维度与 refined 匹配。
    # -------------------------------------------------------------
    if refine_phase and refined.shape[-1] != target.shape[-1]:
        # 如果是修正阶段且维度不匹配，则对 target 进行切片
        # 我们假设 refined 的维度是 target 的一个子集
        # 例如，将 263 维切片为 66 维
        target = target[..., :refined.shape[-1]]
        # 此时，refined 和 target 的形状都为 [seq_len, 66]

    # 计算各项奖励
    r_fid = reward_imitation(refined, target)
    r_smooth = reward_smoothness(refined)
    r_contact = reward_contact_stub(refined, contact_info)
    
    # 计算总奖励
    total = weights['fidelity'] * r_fid + weights['smooth'] * r_smooth + weights['contact'] * r_contact
    
    return total, {'fidelity': r_fid, 'smooth': r_smooth, 'contact': r_contact}

# import numpy as np
# import torch

# def reward_imitation(refined: np.ndarray, target: np.ndarray):
#     """负均方误差 (MSE)，值越高越好。"""
#     if refined.shape != target.shape:
#         raise ValueError(f"Shape mismatch in reward_imitation: refined {refined.shape} vs target {target.shape}")
    
#     mse = np.mean((refined - target) ** 2)
#     # 使用负MSE作为奖励，并进行适当缩放
#     return -float(mse)

# def reward_smoothness(refined: np.ndarray):
#     """
#     惩罚二阶差分过高的动作，值越高越好。
#     **关键修复**: 降低惩罚权重，避免过度约束
#     """
#     if refined.shape[0] < 3:
#         return 0.0
    
#     vel = np.diff(refined, axis=0)
#     acc = np.diff(vel, axis=0)
    
#     # **修复**: 使用更温和的平滑性惩罚
#     smoothness_penalty = np.mean(acc**2)
#     return -float(smoothness_penalty * 0.01)  # 进一步降低权重

# def reward_contact_stub(refined: np.ndarray, contact_info):
#     """
#     接触奖励的占位符。
#     - contact_info: 预期的接触标签或可供性图。
#     目前返回 0.0。
#     """
#     return 0.0

# def compute_total_reward(refined: np.ndarray, target: np.ndarray, 
#                         contact_info=None, refine_phase=False, weights=None):
#     """
#     计算总奖励 - 修复版本
    
#     Args:
#         refined: PPO修正后的运动 [seq_len, motion_dim_refined]
#         target: 原始的真实运动 [seq_len, motion_dim_full]  
#         contact_info: 接触信息
#         refine_phase: 标识当前是否在修正阶段
#         weights: 各项奖励的权重
#     Returns:
#         tuple: 总奖励和各项奖励的字典
#     """
#     if weights is None:
#         # **关键修复**: 调整奖励权重，减少过度平滑化
#         weights = {
#             'fidelity': 1.0,    # 模仿奖励 - 主要目标
#             'smooth': 0.01,     # 平滑性奖励 - 从0.5降到0.01，大幅减少约束  
#             'contact': 0.0      # 接触奖励 - 暂时关闭
#         }
    
#     # **数值稳定性检查**
#     if np.isnan(refined).any() or np.isinf(refined).any():
#         print("Warning: NaN or Inf detected in refined motion")
#         return -100.0, {'fidelity': -100.0, 'smooth': 0.0, 'contact': 0.0}
    
#     if np.isnan(target).any() or np.isinf(target).any():
#         print("Warning: NaN or Inf detected in target motion")
#         return -100.0, {'fidelity': -100.0, 'smooth': 0.0, 'contact': 0.0}
    
#     # **关键修复**: 维度处理
#     # 因为你的训练代码现在使用完整维度，所以refined和target应该匹配
#     if refined.shape != target.shape:
#         print(f"Warning: Dimension mismatch: refined {refined.shape} vs target {target.shape}")
#         if refine_phase and refined.shape[-1] != target.shape[-1]:
#             # 如果维度不匹配，截取target的前refined.shape[-1]维
#             if refined.shape[-1] < target.shape[-1]:
#                 target = target[..., :refined.shape[-1]]
#                 print(f"Adjusted target shape to {target.shape}")
#             else:
#                 print(f"Unexpected dimension relationship")
#                 return -50.0, {'fidelity': -50.0, 'smooth': 0.0, 'contact': 0.0}
    
#     try:
#         # 计算各项奖励
#         r_fid = reward_imitation(refined, target)
#         r_smooth = reward_smoothness(refined)
#         r_contact = reward_contact_stub(refined, contact_info)
        
#         # **数值检查和限制**
#         if np.isnan(r_fid) or np.isinf(r_fid):
#             print(f"Invalid fidelity reward: {r_fid}")
#             r_fid = -10.0
#         if np.isnan(r_smooth) or np.isinf(r_smooth): 
#             print(f"Invalid smoothness reward: {r_smooth}")
#             r_smooth = 0.0
#         if np.isnan(r_contact) or np.isinf(r_contact):
#             print(f"Invalid contact reward: {r_contact}")
#             r_contact = 0.0
        
#         # 计算总奖励
#         total = (weights['fidelity'] * r_fid + 
#                 weights['smooth'] * r_smooth + 
#                 weights['contact'] * r_contact)
        
#         # **关键修复**: 限制奖励范围，防止极端值影响训练
#         total = np.clip(total, -20.0, 2.0)  # 更合理的范围
        
#         reward_dict = {'fidelity': r_fid, 'smooth': r_smooth, 'contact': r_contact}
        
#         return float(total), reward_dict
        
#     except Exception as e:
#         print(f"Error in reward computation: {e}")
#         return -10.0, {'fidelity': -10.0, 'smooth': 0.0, 'contact': 0.0}