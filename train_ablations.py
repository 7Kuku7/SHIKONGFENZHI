import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
import random
import os
import json
import datetime
from tqdm import tqdm
import argparse

from config11 import Config
from datasets.nerf_loader_dual import NerfDatasetDual
from models.nerf_qa_ablations import NeRFQA_SpatialOnly, NeRFQA_TemporalOnly
from utils import calculate_srcc, calculate_plcc, calculate_krcc

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def worker_init_fn(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

class RankLoss(nn.Module):
    def forward(self, preds, targets):
        preds_diff = preds.unsqueeze(1) - preds.unsqueeze(0)
        targets_diff = targets.unsqueeze(1) - targets.unsqueeze(0)
        S = torch.sign(targets_diff)
        mask = (S != 0) & (S.abs() > 0)
        if mask.sum() == 0: return torch.tensor(0.0).to(preds.device)
        loss = torch.relu(-S * preds_diff + 0.1)
        return (loss * mask).sum() / (mask.sum() + 1e-6)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, required=True, choices=['spatial', 'temporal'], help="Ablation mode")
    args = parser.parse_args()

    cfg = Config()
    set_seed(cfg.SEED)
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    mode_name = "SpatialOnly" if args.mode == 'spatial' else "TemporalOnly"
    output_dir = os.path.join("results/Ablations", mode_name, f"seed_{cfg.SEED}", f"run_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    from consts_scene_split import build_scene_split, TRAIN_SCENES, VAL_SCENES, TEST_SCENES
    import shutil
    
    train_paths, val_paths, test_paths = build_scene_split(cfg.ROOT_DIR, seed=cfg.SEED)
    
    split_record = {
        "seed_used_for_split": cfg.SEED,
        "split_type": "scene_level",
        "train_scenes": TRAIN_SCENES,
        "val_scenes": VAL_SCENES,
        "test_scenes": TEST_SCENES
    }
    
    with open(os.path.join(output_dir, "dataset_split.json"), "w") as f:
        json.dump(split_record, f, indent=4)
    
    shutil.copy("config11.py", os.path.join(output_dir, "config11_backup.py"))
    
    print("="*80)
    print(f"🚀 开始消融实验训练：{mode_name} 单流模型")
    print(f"输出目录: {output_dir}")
    print("="*80)
    
    NUM_FRAMES_SPATIAL = 24
    NUM_CROPS_TRAIN = 5
    NUM_FRAMES_TEMPORAL = 80
    
    train_set = NerfDatasetDual(cfg.ROOT_DIR, cfg.MOS_FILE, mode='train', 
                                num_frames_spatial=NUM_FRAMES_SPATIAL, 
                                num_crops=NUM_CROPS_TRAIN,
                                num_frames_temporal=NUM_FRAMES_TEMPORAL)
    val_set = NerfDatasetDual(cfg.ROOT_DIR, cfg.MOS_FILE, mode='val', 
                              num_frames_spatial=NUM_FRAMES_SPATIAL, 
                              num_crops=NUM_CROPS_TRAIN,
                              num_frames_temporal=NUM_FRAMES_TEMPORAL)
    
    # B = 2
    # accumulation_steps = 4 
    
    # 针对不同模式，采用不同的 Batch Size 以充分利用显存
    # 时间流 (3D CNN) 参数量和激活值都较小，可以适当调大 batch_size
    # 空间流 (Swin-Tiny) 显存占用大，需要维持较小的 batch_size
    if args.mode == 'temporal':
        B = 8  # 纯时间流显存充裕，调大 B 加速训练
        accumulation_steps = 1 # 因为 B 够大了，不需要梯度累积
    else:
        B = 2  # 纯空间流由于 Dense Crop 极占显存，保持 B=2
        accumulation_steps = 4 

    train_loader = DataLoader(train_set, batch_size=B, shuffle=True, num_workers=4, worker_init_fn=worker_init_fn, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=B, shuffle=False, num_workers=4, worker_init_fn=worker_init_fn, drop_last=True)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    if args.mode == 'spatial':
        model = NeRFQA_SpatialOnly(soft_topk_temperature=0.1)
    else:
        model = NeRFQA_TemporalOnly()
        
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
        
    model = model.to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=70, eta_min=1e-6)
    mse_crit = nn.MSELoss()
    rank_crit = RankLoss()
    
    best_score = -1.0
    
    for epoch in range(1, 71):
        model.train()
        total_loss = 0
        pbar = tqdm(train_loader, desc=f"Ep {epoch}/70", leave=False)
        
        optimizer.zero_grad()
        
        for batch_idx, batch in enumerate(pbar):
            x_s, x_t, score, _ = batch
            x_s, x_t, score = x_s.to(device), x_t.to(device), score.to(device)
            
            # Mixup
            if np.random.rand() < 0.8 and score.size(0) >= 2:
                lam = max(min(np.random.beta(1.0, 1.0), 0.7), 0.3)
                index = torch.randperm(score.size(0)).to(device)
                score = lam * score + (1 - lam) * score[index]
                if args.mode == 'spatial':
                    x_s = lam * x_s + (1 - lam) * x_s[index]
                else:
                    x_t = lam * x_t + (1 - lam) * x_t[index]
            
            # if np.random.rand() < 0.8 and x_s.size(0) >= 2: # 必须依赖输入的 batch 大小
            #     lam = max(min(np.random.beta(1.0, 1.0), 0.7), 0.3)
            #     index = torch.randperm(x_s.size(0)).to(device)
                
            #     # 先保存原图和原标签，再做混合，绝对不复用变量名
            #     mixed_x_s = lam * x_s + (1 - lam) * x_s[index]
            #     mixed_x_t = lam * x_t + (1 - lam) * x_t[index]
            #     mixed_score = lam * score + (1 - lam) * score[index]
                
            #     # 混合后再赋回去
            #     x_s, x_t, score = mixed_x_s, mixed_x_t, mixed_score
                
            if args.mode == 'spatial':
                pred = model(x_s).view(-1)
            else:
                pred = model(x_t).view(-1)
                
            loss = mse_crit(pred, score) + 0.2 * rank_crit(pred, score)
            
            loss = loss / accumulation_steps
            loss.backward()
            
            if (batch_idx + 1) % accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
                optimizer.zero_grad()
            
            total_loss += loss.item() * accumulation_steps
            pbar.set_postfix({'loss': f"{loss.item() * accumulation_steps:.4f}"})
        
        optimizer.step()
        optimizer.zero_grad()
            
        scheduler.step()
        
        model.eval()
        preds, targets, keys = [], [], []
        with torch.no_grad():
            for batch in val_loader:
                x_s, x_t, score, key = batch
                x_s, x_t, score = x_s.to(device), x_t.to(device), score.to(device)
                
                if args.mode == 'spatial':
                    pred = model(x_s).view(-1)
                else:
                    pred = model(x_t).view(-1)
                
                preds.extend(pred.cpu().numpy())
                targets.extend(score.cpu().numpy())
                keys.extend(key)
                
        preds, targets = np.array(preds), np.array(targets)
        srcc = calculate_srcc(preds, targets)
        plcc = calculate_plcc(preds, targets)
        krcc = calculate_krcc(preds, targets)
        combined = srcc + plcc
        
        print(f"Epoch {epoch} | Loss: {total_loss/len(train_loader):.4f} | Val SRCC: {srcc:.4f} | PLCC: {plcc:.4f} | Combined: {combined:.4f}")
        
        if combined > best_score:
            best_score = combined
            print(f"  >>> New Best Combined: {combined:.4f} (Epoch {epoch}) -> Saving...")
            
            model_to_save = model.module if hasattr(model, 'module') else model
            
            torch.save({
                'epoch': epoch,
                'state_dict': model_to_save.state_dict(),
                'optimizer': optimizer.state_dict(),
                'best_score': best_score,
                'srcc': srcc,
                'plcc': plcc
            }, os.path.join(output_dir, "best_model.pth"))
            
            with open(os.path.join(output_dir, "best_val_results.json"), "w") as f:
                json.dump({
                    "epoch": epoch,
                    "metrics": {
                        "srcc": float(srcc), 
                        "plcc": float(plcc), 
                        "krcc": float(krcc)
                    }
                }, f, indent=4)

if __name__ == "__main__":
    main()
