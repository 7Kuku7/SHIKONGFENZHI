import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models.video as video_models
from .backbone import get_distortion_encoder

class Lightweight3DCNN(nn.Module):
    """
    轻量级 3D CNN，用于捕捉低分辨率全图序列的时域不一致性 (View-inconsistency)
    """
    def __init__(self, in_channels=3, out_dim=256):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, 32, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.bn1 = nn.BatchNorm3d(32)
        # 第一层：时空同时下采样 [T/2, H/2, W/2]
        self.pool1 = nn.MaxPool3d(kernel_size=(2, 2, 2))
        
        self.conv2 = nn.Conv3d(32, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.bn2 = nn.BatchNorm3d(64)
        # 第二层：时空同时下采样 [T/4, H/4, W/4]
        self.pool2 = nn.MaxPool3d(kernel_size=(2, 2, 2))
        
        self.conv3 = nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.bn3 = nn.BatchNorm3d(128)
        # 第三层：只在空间下采样，保留时间维度 [T/4, H/8, W/8]
        self.pool3 = nn.MaxPool3d(kernel_size=(1, 2, 2))
        
        # 空间全局池化，保留时间序列。假设输入 T=96，最后输出的时间维度是 T'=24
        self.spatial_global_pool = nn.AdaptiveAvgPool3d((None, 1, 1))
        self.fc = nn.Linear(128 * 2, out_dim) # 乘以 2 因为要拼接均值和方差

    def forward(self, x):
        # x: [B, C, T, H, W]
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool3(x)
        
        # [B, 128, T', 1, 1]
        x = self.spatial_global_pool(x)
        x = x.squeeze(-1).squeeze(-1) # -> [B, 128, T']
        
        # 计算时序均值和方差，捕捉动态突变
        feat_mean = x.mean(dim=2) # [B, 128]
        feat_std = x.std(dim=2)   # [B, 128]
        
        # [B, 256]
        x_fused = torch.cat([feat_mean, feat_std], dim=1)
        x_out = self.fc(x_fused)
        return x_out

class NeRFQA_AsymmetricDualStream(nn.Module):
    """
    非对称双流架构 (Asymmetric Two-Stream Architecture) + 晚期融合 (Late Fusion)
    - 空间流 (Spatial Stream): 2D Swin + Soft Top-K Pooling，处理 Dense Crops，抓取高频噪点和局部伪影。
    - 时间流 (Temporal Stream): 预训练的 r3d_18，处理下采样全图序列，抓取宏观的闪烁和视角不一致 (View-inconsistency)。
    """
    def __init__(self, soft_topk_temperature=0.1):
        super().__init__()
        
        # === 空间分支 (Spatial Stream) ===
        self.soft_topk_temperature = soft_topk_temperature
        self.spatial_encoder = get_distortion_encoder(pretrained=True)
        self.spatial_feat_dim = 768
        
        # === 时间分支 (Temporal Stream) ===
        # 方案一：使用预训练的 r3d_18 提取时域特征，利用其在大规模数据上的泛化能力
        # self.temporal_encoder = video_models.r3d_18(pretrained=True)
        self.temporal_encoder = video_models.r3d_18(weights='KINETICS400_V1')
        # 修改最后的全连接层以输出我们需要的时间特征维度
        in_features = self.temporal_encoder.fc.in_features
        self.temporal_feat_dim = 256
        self.temporal_encoder.fc = nn.Linear(in_features, self.temporal_feat_dim)
        
        # === 方案二：晚期融合回归头 (Late Fusion Heads) ===
        # 空间流单独的分数预测头 (输入维度为 feat_dim，因为删除了方差拼接)
        self.spatial_regressor = nn.Sequential(
            nn.Linear(self.spatial_feat_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
        
        # 时间流单独的分数预测头
        self.temporal_regressor = nn.Sequential(
            nn.Linear(self.temporal_feat_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x_s, x_t):
        # x_s (Spatial): [B, T, N, C, H, W]
        # x_t (Temporal): [B, C_in, T, H', W']
        
        b, t, n, c, h, w = x_s.shape
        
        # ==========================================
        # 1. 空间流特征提取与分数预测
        # ==========================================
        x_s_flat = x_s.view(b * t * n, c, h, w)
        feat_s_raw = self.spatial_encoder.forward_features(x_s_flat) 
        
        # Soft Top-K Pooling
        if len(feat_s_raw.shape) == 4:
            feat_s_spatial = feat_s_raw.permute(0, 3, 1, 2).flatten(2)
        else:
            feat_s_spatial = feat_s_raw.transpose(1, 2)
            
        score_map = torch.norm(feat_s_spatial, dim=1, keepdim=True)
        attention_weights = F.softmax(score_map / self.soft_topk_temperature, dim=-1)
        feat_s_patch = (feat_s_spatial * attention_weights).sum(dim=-1) # [B*T*N, C_feat]
        
        feat_s_patch = feat_s_patch.view(b, t, n, -1)
        feat_s_frame = feat_s_patch.mean(dim=2) # [B, T, C_feat]
        
        # 空间分支的时序聚合 (彻底删除时间方差，实现极致特征解耦)
        feat_s_mean = feat_s_frame.mean(dim=1) # [B, C_feat]
        
        score_s = self.spatial_regressor(feat_s_mean)
        
        # ==========================================
        # 2. 时间流特征提取与分数预测
        # ==========================================
        feat_t = self.temporal_encoder(x_t) # [B, 256]
        score_t = self.temporal_regressor(feat_t)
        
        # 返回双重分数，供双重 Loss 和 晚期融合 加权使用
        return score_s, score_t
