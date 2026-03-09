import torch
import torch.nn as nn
import torch.nn.functional as F
from .backbone import get_content_encoder, get_distortion_encoder

class AdaptiveFeatureFusion(nn.Module):
    """
    Adaptive Feature Fusion using Gated Attention.
    Learns to dynamically weight Content and Distortion features.
    """
    def __init__(self, feature_dim=768):
        super().__init__()
        # Attention weights
        self.attn_fc = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim // 2),
            nn.ReLU(),
            nn.Linear(feature_dim // 2, 2), # Output 2 weights (alpha, beta)
            nn.Softmax(dim=1)
        )
        
    def forward(self, feat_c, feat_d):
        # feat_c, feat_d: [B, D]
        combined = torch.cat([feat_c, feat_d], dim=1)
        weights = self.attn_fc(combined) # [B, 2]
        
        alpha = weights[:, 0].unsqueeze(1) # Weight for content
        beta = weights[:, 1].unsqueeze(1)  # Weight for distortion
        
        # Weighted fusion
        # We still concat them, but weighted, or sum them?
        # Concatenation preserves more info. Let's weight them before concat.
        # Or better: Fused = alpha * C + beta * D (if dimensions match)
        # But C and D represent different things.
        # Let's use the weights to scale the features before concatenation.
        
        feat_c_weighted = feat_c * (1 + alpha) # Residual-like scaling
        feat_d_weighted = feat_d * (1 + beta)
        
        return torch.cat([feat_c_weighted, feat_d_weighted], dim=1)

class DisNeRFQA_Advanced(nn.Module):
    def __init__(self, num_subscores=4, use_fusion=True):
        super().__init__()
        self.use_fusion = use_fusion
        
        # 1. Backbones
        self.content_encoder = get_content_encoder(pretrained=True)
        self.distortion_encoder = get_distortion_encoder(pretrained=True)
        
        # Feature dimensions
        self.feat_dim = 768 # ViT-Base and Swin-Tiny (projected)

        # 3. Adaptive Fusion (Innovation)
        if self.use_fusion:
            self.fusion = AdaptiveFeatureFusion(self.feat_dim)
        
        # 4. Heads
        # Main Quality Regressor (0-1)
        # If fusion, input is 2*D (weighted). If no fusion, input is 2*D (concat).
        # Dimensions are same, just logic differs.
        self.regressor = nn.Sequential(
            nn.Linear(self.feat_dim * 2, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid() 
        )

        self.num_subscores = num_subscores
        # [修改 1/2] 子分数预测头只接收失真分支特征 (feat_dim)
        # 这样做可以强制 Swin 分支学习失真相关的特征
        self.subscore_head = nn.Sequential(
            nn.Linear(self.feat_dim, 256),  # 从 feat_dim*2 改为 feat_dim
            nn.ReLU(),
            nn.Dropout(0.2),  # 添加 Dropout 防止小数据集过拟合
            nn.Linear(256, num_subscores),
            nn.Sigmoid()
        )

    def forward(self, x_content, x_distortion):
        # x: [B, T, C, H, W]
        b, t, c, h, w = x_content.shape
        
        # --- Feature Extraction ---
        x_c_flat = x_content.view(b * t, c, h, w)
        x_d_flat = x_distortion.view(b * t, c, h, w)
        
        # Content Branch
        feat_c_raw = self.content_encoder.forward_features(x_c_flat)
        if hasattr(self.content_encoder, 'global_pool'):
             feat_c_seq = self.content_encoder.forward_head(feat_c_raw, pre_logits=True)
        else:
             feat_c_seq = feat_c_raw[:, 0]
             
        # Distortion Branch
        feat_d_raw = self.distortion_encoder.forward_features(x_d_flat)
        
        # ========================================
        # [修改 2/2] Soft Top-K Pooling (空间注意力机制)
        # ========================================
        # 目的：用 Softmax 生成平滑的注意力权重，让所有位置都能获得梯度
        # 避免硬性 Top-K 导致的梯度稀疏问题
        # ========================================
        
        # 假设 feat_d_raw 是 [B*T, H, W, C]
        if len(feat_d_raw.shape) == 4:
            # [B*T, H, W, C] -> [B*T, C, H*W]
            feat_d_flat_spatial = feat_d_raw.permute(0, 3, 1, 2).flatten(2)
        else:
            # [B*T, L, C] -> [B*T, C, L]
            feat_d_flat_spatial = feat_d_raw.transpose(1, 2)
        
        # 计算每个空间位置的失真分数（L2 范数）
        # score_map: [B*T, H*W] 或 [B*T, L]
        score_map = torch.norm(feat_d_flat_spatial, dim=1, keepdim=True)  # [B*T, 1, H*W]
        
        # 使用 Softmax 将分数转为注意力权重
        # temperature=0.1: 温度参数，越小越尖锐（接近 Hard Top-K），越大越平滑
        temperature = 0.1
        attention_weights = F.softmax(score_map / temperature, dim=-1)  # [B*T, 1, H*W]
        
        # 加权求和：分数高的区域获得更大权重
        # 所有位置都参与计算，梯度不再稀疏
        feat_d_seq = (feat_d_flat_spatial * attention_weights).sum(dim=-1)  # [B*T, C]
        # ========================================
        
        # Reshape back
        feat_c_seq = feat_c_seq.view(b, t, -1)
        feat_d_seq = feat_d_seq.view(b, t, -1)
        
        # Temporal Pooling
        feat_c = feat_c_seq.mean(dim=1)
        feat_d = feat_d_seq.mean(dim=1)
        
        # # ==========================================
        # # [修改这里] 实验 A: 去掉内容分支 (Only Swin
        # # ==========================================
        # # 解释：把 feat_c 全部变成 0，模拟模型“看不见”内容
        # # 注意：这里必须用 zeros_like，保持维度形状不变，否则后面拼接会报错
        # feat_c = torch.zeros_like(feat_c).to(feat_c.device) 
        # # ==========================================

        # # ==========================================
        # # [修改这里] 实验 B: 去掉失真分支 (Only ViT
        # # ==========================================
        # # 解释：把 feat_d 全部变成 0，模拟模型“看不见”失真细节
        # feat_d = torch.zeros_like(feat_d).to(feat_d.device)
        # # # ==========================================

        # --- Adaptive Fusion ---
        if self.use_fusion:
            feat_fused = self.fusion(feat_c, feat_d)
        else:
            # Simple Concatenation (Ablation)
            feat_fused = torch.cat([feat_c, feat_d], dim=1)
        
        # --- Predictions ---
        # 1. Quality Score (使用融合特征)
        score = self.regressor(feat_fused)
        
        # 2. Sub-scores (只用失真分支预测)
        # 强制 Swin 分支学习失真相关特征（模糊、伪影、光照、不适感）
        sub_scores = self.subscore_head(feat_d)
        
        return score, sub_scores, feat_c, feat_d

class MultiTaskLoss(nn.Module):
    # 基于同方差不确定性(Homoscedastic Uncertainty)的自适应多任务Loss权重
    # Reference: Kendall et al. "Multi-Task Learning Using Uncertainty to Weigh Losses", CVPR 2018.
    def __init__(self, num_tasks=4):
        super(MultiTaskLoss, self).__init__()
        # log_vars 是可学习的参数 (log(sigma^2))
        # 初始化为0，即初始权重为 1.0
        self.log_vars = nn.Parameter(torch.zeros(num_tasks))

    def forward(self, input_losses):
        # input_losses: list of losses [L_reg, L_rank, L_mi, L_sub]
        # 确保输入是列表
        loss_sum = 0
        for i, loss in enumerate(input_losses):
            # 核心公式: L = (1 / 2*sigma^2) * L_i + log(sigma)
            # log_vars[i] = log(sigma^2)
            precision = torch.exp(-self.log_vars[i])
            loss_sum += 0.5 * precision * loss + 0.5 * self.log_vars[i]

        return loss_sum

