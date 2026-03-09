# python test_dual.py --run_dir results/AsymmetricDualStream/seed_3407/run_20260306_204942 --num_crops 9

import torch
from torch.utils.data import DataLoader
import numpy as np
import os
import json
import argparse

from config11 import Config
from datasets.nerf_loader_dual import NerfDatasetDual
from models.nerf_qa_dual import NeRFQA_AsymmetricDualStream
from utils import calculate_srcc, calculate_plcc, calculate_krcc, calculate_rmse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", type=str, required=True, help="Path to run directory")
    parser.add_argument("--gpu", type=str, default="0,1", help="GPU ID")
    parser.add_argument("--num_crops", type=int, default=5, help="Number of dense crops per frame")
    parser.add_argument("--num_frames_spatial", type=int, default=24, help="Number of spatial frames to sample")
    parser.add_argument("--num_frames_temporal", type=int, default=80, help="Number of temporal frames to sample")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    cfg = Config()

    print("="*80)
    print(f"🚀 开始非对称双流架构测试 (Asymmetric Dual Stream)")
    print(f"空间流帧数: {args.num_frames_spatial} | 时间流帧数: {args.num_frames_temporal} | 每帧裁剪数: {args.num_crops}")
    print("="*80)

    test_set = NerfDatasetDual(
        cfg.ROOT_DIR, 
        cfg.MOS_FILE, 
        mode='test', 
        num_frames_spatial=args.num_frames_spatial, 
        num_crops=args.num_crops,
        num_frames_temporal=args.num_frames_temporal
    )
    
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=4)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = NeRFQA_AsymmetricDualStream(soft_topk_temperature=0.1)
    
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
        
    model = model.to(device)
    
    ckpt_path = os.path.join(args.run_dir, "best_model.pth")
    print(f"Loading weights from: {ckpt_path}")
    
    checkpoint = torch.load(ckpt_path, map_location=device)['state_dict']
    
    if isinstance(model, torch.nn.DataParallel) and not any(k.startswith('module.') for k in checkpoint.keys()):
        checkpoint = {'module.' + k: v for k, v in checkpoint.items()}
    elif not isinstance(model, torch.nn.DataParallel) and any(k.startswith('module.') for k in checkpoint.keys()):
        checkpoint = {k.replace('module.', ''): v for k, v in checkpoint.items()}
        
    model.load_state_dict(checkpoint)
    model.eval()

    preds, targets, keys = [], [], []
    
    with torch.no_grad():
        for batch in test_loader:
            x_s, x_t, score, key = batch
            x_s, x_t = x_s.to(device), x_t.to(device)
            pred_s, pred_t = model(x_s, x_t)
            pred_s = pred_s.view(-1)
            pred_t = pred_t.view(-1)
            
            # 晚期融合加权
            pred = 0.8 * pred_s + 0.2 * pred_t
            
            preds.extend(pred.cpu().numpy())
            targets.extend(score.numpy())
            keys.extend(key)

    preds = np.array(preds)
    targets = np.array(targets)
    
    srcc = calculate_srcc(preds, targets)
    plcc = calculate_plcc(preds, targets)
    krcc = calculate_krcc(preds, targets)
    rmse = calculate_rmse(preds, targets)

    print("\n" + "="*50)
    print("       TEST RESULTS (Dual Stream)      ")
    print("="*50)
    print(f"SRCC: {srcc:.4f}")
    print(f"PLCC: {plcc:.4f}")
    print(f"KRCC: {krcc:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print("="*50)

    save_path = os.path.join(args.run_dir, f"test_results_dual-{args.num_frames_spatial}-{args.num_frames_temporal}-{args.num_crops}.json")
    
    unique_scenes = sorted(list(set([k.split('+')[0] for k in keys])))
    
    sample_details = []
    for i, key in enumerate(keys):
        sample_details.append({
            "sample_id": i,
            "key": key,
            "scene": key.split('+')[0],
            "method": key.split('+')[1] if len(key.split('+')) > 1 else "unknown",
            "predicted_score": float(preds[i]),
            "ground_truth": float(targets[i]),
            "error": float(abs(preds[i] - targets[i]))
        })
    
    with open(save_path, "w") as f:
        json.dump({
            "test_config": {
                "num_frames_spatial": args.num_frames_spatial, 
                "num_frames_temporal": args.num_frames_temporal,
                "num_crops": args.num_crops
            },
            "test_scenes": unique_scenes,
            "metrics": {
                "srcc": float(srcc), 
                "plcc": float(plcc), 
                "krcc": float(krcc),
                "rmse": float(rmse)
            },
            "sample_details": sample_details
        }, f, indent=4)
        
    print(f"[Test] 详细结果已保存至: {save_path}")

if __name__ == "__main__":
    main()
