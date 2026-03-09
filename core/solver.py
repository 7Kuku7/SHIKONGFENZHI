import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import json
import os
from utils import calculate_srcc, calculate_plcc, calculate_krcc
from torch.optim.lr_scheduler import CosineAnnealingLR

# ==========================================
# [关键修复] 真正的 Pairwise Rank Loss
# ==========================================
class RankLoss(nn.Module):
    def forward(self, preds, targets):
        """
        在 Batch 内部构建所有可能的 Pair 进行排序学习。
        preds: [B]
        targets: [B]
        """
        # 扩展成矩阵 [B, B]
        # preds_diff[i][j] = preds[i] - preds[j]
        preds_diff = preds.unsqueeze(1) - preds.unsqueeze(0)
        targets_diff = targets.unsqueeze(1) - targets.unsqueeze(0)
        
        # 符号矩阵：如果 target[i] > target[j]，则 sign 为 1
        S = torch.sign(targets_diff)
        
        # 找出有效的 pair (target 不相等的)
        mask = (S != 0) & (S.abs() > 0)
        
        if mask.sum() == 0:
            return torch.tensor(0.0).to(preds.device)
            
        # RankNet / MarginRank Loss 变体
        # Loss = ReLU( - S * preds_diff + margin )
        loss = torch.relu(-S * preds_diff + 0.1)
        
        return (loss * mask).sum() / (mask.sum() + 1e-6)

class Solver:
    def __init__(self, model, config, train_loader, val_loader):
        self.model = model
        self.cfg = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = torch.device(f"cuda:{config.GPU_ID}" if torch.cuda.is_available() else "cpu")
        
        self.model.to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=config.LR, weight_decay=1e-4)
        
        # 定义损失函数
        self.mse_crit = nn.MSELoss()
        self.rank_crit = RankLoss()     # 使用新版 RankLoss

        # 增加 weight_decay 防止过拟合
        self.optimizer = optim.Adam(self.model.parameters(), lr=config.LR, weight_decay=1e-4) 
        
        # 新增：余弦退火调度器，让 LR 从 1e-4 慢慢降到 1e-6
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=config.EPOCHS, eta_min=1e-6)

        # 打印一下权重配置，方便 debug
        print(f"[Solver] Weights -> MSE: {self.cfg.LAMBDA_MSE}, Rank: {self.cfg.LAMBDA_RANK}, "
              f"SSL: {self.cfg.LAMBDA_SSL}")

    def train_epoch(self, epoch):
        self.model.train()
        total_loss_avg = 0
        
        pbar = tqdm(self.train_loader, desc=f"Ep {epoch}/{self.cfg.EPOCHS}", leave=False)
        
        for batch in pbar:
            # 数据解包
            x_c, x_d, score, sub_gt, _, x_c_aug, x_d_aug = batch
            
            x_c, x_d = x_c.to(self.device), x_d.to(self.device)
            score, sub_gt = score.to(self.device), sub_gt.to(self.device)
            x_c_aug, x_d_aug = x_c_aug.to(self.device), x_d_aug.to(self.device)

            # ==========================================
            # [新增核心] Mixup 数据增强
            # ==========================================
            use_mixup = True  # Mixup 总开关，可放到 config 中配置
            mixup_prob = 0.8  # 50% 概率触发 Mixup
            mixup_alpha = 1.0 # Beta 分布参数，0.2 是常用值
            did_mixup = False # 标记是否触发了 Mixup
            
            if use_mixup and np.random.rand() < mixup_prob:
                did_mixup = True
                # 1. 生成 Beta 分布的混合系数 lam
                lam = np.random.beta(mixup_alpha, mixup_alpha)
                # 确保 lam 在 0.3~0.7 之间，避免混合比例极端
                lam = max(min(lam, 0.7), 0.3)
                
                # 2. 生成 batch 内的乱序索引（Mixup 是 batch 内样本混合）
                batch_size = x_c.size(0)
                index = torch.randperm(batch_size).to(self.device)
                
                # 3. 混合主输入 (x_c, x_d)
                x_c = lam * x_c + (1 - lam) * x_c[index, :]
                x_d = lam * x_d + (1 - lam) * x_d[index, :]
                
                # 4. 混合标签 (score 是标量，sub_gt 是子分数向量)
                score = lam * score + (1 - lam) * score[index]
                sub_gt = lam * sub_gt + (1 - lam) * sub_gt[index, :]
                
                # 注意：SSL 增强图可选方案：
                # 方案1（推荐）：SSL 增强图不参与 Mixup，保持原样
                # 方案2：对 SSL 增强图执行相同的 Mixup
                # 这里选择方案1，避免过度增强导致 SSL Loss 不稳定
                # 如果要方案2，取消下面注释：
                # x_c_aug = lam * x_c_aug + (1 - lam) * x_c_aug[index, :]
                # x_d_aug = lam * x_d_aug + (1 - lam) * x_d_aug[index, :]

            # --- Forward Pass ---
            pred_score, pred_subs, feat_c, feat_d = self.model(x_c, x_d)
            pred_score = pred_score.view(-1)
            
            # --- Calculate Losses ---
            
            # 1. Main MSE (如果 LAMBDA_MSE=0，则跳过)
            loss_mse = torch.tensor(0.0).to(self.device)
            if self.cfg.LAMBDA_MSE > 0:
                loss_mse = self.mse_crit(pred_score, score)
            
            # 2. Rank Loss
            loss_rank = torch.tensor(0.0).to(self.device)
            if self.cfg.LAMBDA_RANK > 0:
                loss_rank = self.rank_crit(pred_score, score)
                
            # 4. Sub-score Loss
            loss_sub = torch.tensor(0.0).to(self.device)
            if self.cfg.LAMBDA_SUB > 0:
                loss_sub = self.mse_crit(pred_subs, sub_gt)

            # ==========================================
            # 5. SSL Loss - View-Consistency (多视角一致性约束)
            # ==========================================
            loss_ssl = torch.tensor(0.0).to(self.device)
            if self.cfg.LAMBDA_SSL > 0:
                # 增强图的前向
                pred_score_aug, _, feat_c_aug, feat_d_aug = self.model(x_c_aug, x_d_aug)
                pred_score_aug = pred_score_aug.view(-1)
                
                # [核心创新] View-Consistency Loss (多视角一致性损失)
                # 物理先验：高质量 NeRF 在相邻视角间应该保持：
                # 1. 预测分数一致（整体质量一致）
                # 2. 特征表示一致（局部细节一致）
                # 
                # 相比传统图像增强（加噪声后质量下降），NeRF 的相邻视角
                # 质量是相同的，因此我们约束它们的分数和特征尽可能接近
                
                # Loss 1: 分数一致性（Score Consistency）
                # 使用 L1 或 L2 距离，而非 ReLU Hinge Loss
                loss_score_consistency = torch.mean(torch.abs(pred_score - pred_score_aug))
                
                # Loss 2: 特征一致性（Feature Consistency）
                # 约束内容特征和失真特征在相邻视角间保持稳定
                loss_feat_c_consistency = torch.mean(torch.abs(feat_c - feat_c_aug))
                loss_feat_d_consistency = torch.mean(torch.abs(feat_d - feat_d_aug))
                
                # 组合 SSL Loss
                # 权重可调：特征一致性通常权重更大（因为特征空间更大）
                loss_ssl = (loss_score_consistency + 
                           0.1 * loss_feat_c_consistency + 
                           0.1 * loss_feat_d_consistency)
            # ==========================================

            # --- Total Loss ---
            total_loss = (self.cfg.LAMBDA_MSE * loss_mse +
                          self.cfg.LAMBDA_RANK * loss_rank +
                          self.cfg.LAMBDA_SUB * loss_sub +
                          self.cfg.LAMBDA_SSL * loss_ssl)

            # --- Backward ---
            self.optimizer.zero_grad()
            total_loss.backward()
            
            # 梯度裁剪防止不稳定
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
            
            self.optimizer.step()
            
            total_loss_avg += total_loss.item()
            pbar.set_postfix({'loss': f"{total_loss.item():.4f}"})

        # 在 epoch 结束时更新 LR
        self.scheduler.step()
        current_lr = self.scheduler.get_last_lr()[0]
        print(f"  [LR Update] Epoch {epoch} finished. Current LR: {current_lr:.6f}")
            
        return total_loss_avg / len(self.train_loader)

    def evaluate(self):
        self.model.eval()
        preds, targets, keys = [], [], []
        
        with torch.no_grad():
            for batch in self.val_loader:
                x_c, x_d, score, _, key, _, _ = batch
                x_c, x_d = x_c.to(self.device), x_d.to(self.device)
                
                pred_score, _, _, _ = self.model(x_c, x_d)
                
                preds.extend(pred_score.cpu().numpy().flatten())
                targets.extend(score.numpy().flatten())
                keys.extend(key)

        preds = np.array(preds)
        targets = np.array(targets)
        
        metrics = {
            "srcc": calculate_srcc(preds, targets),
            "plcc": calculate_plcc(preds, targets),
            "krcc": calculate_krcc(preds, targets),
            "rmse": np.sqrt(np.mean((preds*100 - targets*100)**2))
        }
        return metrics, preds, targets, keys

    # ▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼
    # [这就是你缺失的部分，务必确保这一段代码存在！]
    # ▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼
    def save_model(self, path, epoch, metrics):
        """
        保存模型到 main.py 指定的 path 目录
        """
        state = {
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'metrics': metrics,
            'config': self.cfg.__dict__
        }
        torch.save(state, os.path.join(path, "best_model.pth"))
