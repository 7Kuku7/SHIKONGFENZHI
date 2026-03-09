# python test_ablations.py --run_dir RUN_DIR --mode {spatial,temporal} [--gpu GPU] [--num_crops NUM_CROPS]
#                          [--num_frames_spatial NUM_FRAMES_SPATIAL] [--num_frames_temporal NUM_FRAMES_TEMPORAL]
# python test_ablations.py --run_dir results/Ablations/SpatialOnly/seed_3407/run_20260305_093135 --mode spatial --num_crops 5

import torch
from torch.utils.data import DataLoader
import numpy as np
import os
import json
import argparse

from config11 import Config
from datasets.nerf_loader_dual import NerfDatasetDual
from models.nerf_qa_ablations import NeRFQA_SpatialOnly, NeRFQA_TemporalOnly
from utils import calculate_srcc, calculate_plcc, calculate_krcc, calculate_rmse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", type=str, required=True, help="Path to run directory")
    parser.add_argument("--mode", type=str, required=True, choices=['spatial', 'temporal'], help="Ablation mode")
    parser.add_argument("--gpu", type=str, default="0,1", help="GPU ID")
    parser.add_argument("--num_crops", type=int, default=5, help="Number of dense crops per frame")
    parser.add_argument("--num_frames_spatial", type=int, default=24, help="Number of spatial frames to sample")
    parser.add_argument("--num_frames_temporal", type=int, default=80, help="Number of temporal frames to sample")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    cfg = Config()

    print("="*80)
    print(f"🚀 开始消融实验测试: {args.mode.capitalize()} Only")
    print(f"参数: {args.num_frames_spatial} F_s | {args.num_frames_temporal} F_t | {args.num_crops} Crops")
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
    
    if args.mode == 'spatial':
        model = NeRFQA_SpatialOnly(soft_topk_temperature=0.1)
    else:
        model = NeRFQA_TemporalOnly()
    
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
            
            if args.mode == 'spatial':
                pred = model(x_s).view(-1)
            else:
                pred = model(x_t).view(-1)
            
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
    print(f"    TEST RESULTS ({args.mode.capitalize()} Only)    ")
    print("="*50)
    print(f"SRCC: {srcc:.4f}")
    print(f"PLCC: {plcc:.4f}")
    print(f"KRCC: {krcc:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print("="*50)

    save_path = os.path.join(args.run_dir, f"test_results_{args.mode}-{args.num_frames_spatial}-{args.num_frames_temporal}-{args.num_crops}.json")
    
    unique_scenes = sorted(list(set([k.split('+')[0] for k in keys])))
    
    with open(save_path, "w") as f:
        json.dump({
            "test_config": {
                "mode": args.mode,
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
            }
        }, f, indent=4)
        
    print(f"[Test] 详细结果已保存至: {save_path}")

if __name__ == "__main__":
    main()
