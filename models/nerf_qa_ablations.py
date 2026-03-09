import torch
import torch.nn as nn
import torch.nn.functional as F
from .backbone import get_distortion_encoder
import torchvision.models.video as video_models

class NeRFQA_SpatialOnly(nn.Module):
    """
    消融实验 A: 纯空间单流 Baseline (Spatial Only)
    - 只使用 2D Swin + Soft Top-K Pooling 处理 Dense Crops。
    - 极致解耦：彻底删除时间方差。
    """
    def __init__(self, soft_topk_temperature=0.1):
        super().__init__()
        self.soft_topk_temperature = soft_topk_temperature
        self.spatial_encoder = get_distortion_encoder(pretrained=True)
        self.spatial_feat_dim = 768
        
        # 回归头：只接受空间特征的 均值
        self.regressor = nn.Sequential(
            nn.Linear(self.spatial_feat_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x_s):
        # x_s: [B, T, N, C, H, W]
        b, t, n, c, h, w = x_s.shape
        
        x_s_flat = x_s.view(b * t * n, c, h, w)
        feat_s_raw = self.spatial_encoder.forward_features(x_s_flat) 
        
        if len(feat_s_raw.shape) == 4:
            feat_s_spatial = feat_s_raw.permute(0, 3, 1, 2).flatten(2)
        else:
            feat_s_spatial = feat_s_raw.transpose(1, 2)
            
        score_map = torch.norm(feat_s_spatial, dim=1, keepdim=True)
        attention_weights = F.softmax(score_map / self.soft_topk_temperature, dim=-1)
        feat_s_patch = (feat_s_spatial * attention_weights).sum(dim=-1)
        
        feat_s_patch = feat_s_patch.view(b, t, n, -1)
        feat_s_frame = feat_s_patch.mean(dim=2)
        
        feat_s_mean = feat_s_frame.mean(dim=1)
        
        score = self.regressor(feat_s_mean)
        
        return score

class NeRFQA_TemporalOnly(nn.Module):
    """
    消融实验 B: 纯时间单流 Baseline (Temporal Only)
    - 只使用 预训练的 R3D_18 处理下采样全图。
    """
    def __init__(self):
        super().__init__()
        self.temporal_encoder = video_models.r3d_18(weights='KINETICS400_V1')
        in_features = self.temporal_encoder.fc.in_features
        self.temporal_feat_dim = 256
        self.temporal_encoder.fc = nn.Linear(in_features, self.temporal_feat_dim)
        
        self.regressor = nn.Sequential(
            nn.Linear(self.temporal_feat_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x_t):
        # x_t: [B, C_in, T, H', W']
        feat_t = self.temporal_encoder(x_t)
        score = self.regressor(feat_t)
        return score
