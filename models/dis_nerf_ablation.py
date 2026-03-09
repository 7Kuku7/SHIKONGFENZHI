"""
支持消融实验的模型
可以通过配置灵活控制各个模块的启用/禁用
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from .backbone import get_content_encoder, get_distortion_encoder

class AdaptiveFeatureFusion(nn.Module):
    """自适应门控特征融合"""
    def __init__(self, feature_dim=768):
        super().__init__()
        self.attn_fc = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim // 2),
            nn.ReLU(),
            nn.Linear(feature_dim // 2, 2),
            nn.Softmax(dim=1)
        )
        
    def forward(self, feat_c, feat_d):
        combined = torch.cat([feat_c, feat_d], dim=1)
        weights = self.attn_fc(combined)
        
        alpha = weights[:, 0].unsqueeze(1)
        beta = weights[:, 1].unsqueeze(1)
        
        feat_c_weighted = feat_c * (1 + alpha)
        feat_d_weighted = feat_d * (1 + beta)
        
        return torch.cat([feat_c_weighted, feat_d_weighted], dim=1)

class DisNeRFQA_Advanced(nn.Module):
    def __init__(self, num_subscores=4, use_fusion=True,
                 use_soft_topk=True, soft_topk_temperature=0.1,
                 subscore_use_distortion_only=True,
                 ablate_content_branch=False,
                 ablate_distortion_branch=False):
        """
        Args:
            num_subscores: 子分数数量
            use_fusion: 是否使用门控融合
            use_soft_topk: 是否使用 Soft Top-K（否则用 GAP）
            soft_topk_temperature: Soft Top-K 的温度参数
            subscore_use_distortion_only: 子分数预测头是否只用失真特征
            ablate_content_branch: 是否消融内容分支
            ablate_distortion_branch: 是否消融失真分支
        """
        super().__init__()
        self.use_fusion = use_fusion
        self.use_soft_topk = use_soft_topk
        self.soft_topk_temperature = soft_topk_temperature
        self.subscore_use_distortion_only = subscore_use_distortion_only
        self.ablate_content_branch = ablate_content_branch
        self.ablate_distortion_branch = ablate_distortion_branch
        
        # Backbones
        self.content_encoder = get_content_encoder(pretrained=True)
        self.distortion_encoder = get_distortion_encoder(pretrained=True)
        
        self.feat_dim = 768
        
        # Adaptive Fusion
        if self.use_fusion:
            self.fusion = AdaptiveFeatureFusion(self.feat_dim)
        
        # Main Quality Regressor
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
        
        # Sub-score Head（根据配置决定输入维度）
        subscore_input_dim = self.feat_dim if subscore_use_distortion_only else (self.feat_dim * 2)
        self.subscore_head = nn.Sequential(
            nn.Linear(subscore_input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_subscores),
            nn.Sigmoid()
        )

    def forward(self, x_content, x_distortion):
        b, t, c, h, w = x_content.shape
        
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
        # 池化策略：Soft Top-K vs. GAP
        # ========================================
        if self.use_soft_topk:
            # Soft Top-K Pooling
            if len(feat_d_raw.shape) == 4:
                feat_d_flat_spatial = feat_d_raw.permute(0, 3, 1, 2).flatten(2)
            else:
                feat_d_flat_spatial = feat_d_raw.transpose(1, 2)
            
            score_map = torch.norm(feat_d_flat_spatial, dim=1, keepdim=True)
            attention_weights = F.softmax(score_map / self.soft_topk_temperature, dim=-1)
            feat_d_seq = (feat_d_flat_spatial * attention_weights).sum(dim=-1)
        else:
            # Global Average Pooling（传统方法）
            if len(feat_d_raw.shape) == 4:
                feat_d_seq = feat_d_raw.mean(dim=[1, 2])  # [B*T, C]
            else:
                feat_d_seq = feat_d_raw.mean(dim=1)
        # ========================================
        
        # Reshape back
        feat_c_seq = feat_c_seq.view(b, t, -1)
        feat_d_seq = feat_d_seq.view(b, t, -1)
        
        # Temporal Pooling
        feat_c = feat_c_seq.mean(dim=1)
        feat_d = feat_d_seq.mean(dim=1)
        
        # ========================================
        # 消融实验：屏蔽分支
        # ========================================
        if self.ablate_content_branch:
            # 消融内容分支：将特征置零
            feat_c = torch.zeros_like(feat_c).to(feat_c.device)
        
        if self.ablate_distortion_branch:
            # 消融失真分支：将特征置零
            feat_d = torch.zeros_like(feat_d).to(feat_d.device)
        # ========================================
        
        # Adaptive Fusion
        if self.use_fusion:
            feat_fused = self.fusion(feat_c, feat_d)
        else:
            feat_fused = torch.cat([feat_c, feat_d], dim=1)
        
        # Predictions
        score = self.regressor(feat_fused)
        
        # Sub-scores（根据配置选择输入）
        if self.subscore_use_distortion_only:
            sub_scores = self.subscore_head(feat_d)
        else:
            sub_scores = self.subscore_head(feat_fused)
        
        return score, sub_scores, feat_c, feat_d

class MultiTaskLoss(nn.Module):
    def __init__(self, num_tasks=4):
        super(MultiTaskLoss, self).__init__()
        self.log_vars = nn.Parameter(torch.zeros(num_tasks))

    def forward(self, input_losses):
        loss_sum = 0
        for i, loss in enumerate(input_losses):
            precision = torch.exp(-self.log_vars[i])
            loss_sum += 0.5 * precision * loss + 0.5 * self.log_vars[i]
        return loss_sum
