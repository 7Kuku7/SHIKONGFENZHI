# 实现非对称双流采样：空间分支使用密集裁剪 (Dense Crop)，时间分支使用下采样全图序列。
import torch
from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image
import json
import os
import random
import torchvision.transforms as T
from datasets.nerf_loader_dense import TemporalDenseCrop

class NerfDatasetDual(Dataset):
    def __init__(self, root_dir, mos_file, mode='train', 
                 num_frames_spatial=24, num_crops=3, 
                 num_frames_temporal=96, temporal_size=112):
        self.root_dir = Path(root_dir)
        self.num_frames_spatial = num_frames_spatial
        self.num_frames_temporal = num_frames_temporal
        self.mode = mode
        
        from consts_scene_split import build_scene_split
        train, val, test = build_scene_split(self.root_dir)
        self.samples = {'train': train, 'val': val, 'test': test}[mode]
        
        with open(mos_file, 'r') as f:
            self.mos_labels = json.load(f)
            
        self.valid_samples = [s for s in self.samples if self._get_key_from_path(s) in self.mos_labels]
        
        # 空间分支 (Spatial Stream) 的裁剪
        self.dense_crop = TemporalDenseCrop(size=224, num_crops=num_crops, mode=mode)
        
        # 空间分支的 Transform
        self.spatial_transform = T.Compose([
            T.ToTensor(), 
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        # 时间分支 (Temporal Stream) 的 Transform (下采样全图)
        self.temporal_transform = T.Compose([
            T.Resize((temporal_size, temporal_size)),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def _get_key_from_path(self, path):
        parts = path.name.split("__")
        if len(parts) == 4: return "+".join(parts)
        return path.name

    def _load_frames_pil(self, folder_path, mode_type):
        """
        mode_type: 'spatial' 或 'temporal'
        """
        all_frames = sorted(list(folder_path.glob("frame_*.png")))
        if not all_frames: all_frames = sorted(list(folder_path.glob("frame_*.jpg")))
        total_frames = len(all_frames)
        
        if mode_type == 'spatial':
            # 空间流：全局均匀稀疏采样 (例如 24 帧)
            indices = torch.linspace(0, total_frames - 1, self.num_frames_spatial).long()
        else:
            # 时间流：前 80% 帧中密集采样 (例如 96 帧)
            # 丢弃后 20% 的尾部帧（通常是相机停止或者无意义重复段落）
            end_idx = int(total_frames * 0.8)
            # 如果总帧数小于要采样的帧数，就全采；否则在前 80% 内均匀提 96 帧
            if end_idx <= self.num_frames_temporal:
                indices = torch.linspace(0, end_idx - 1, self.num_frames_temporal).long()
            else:
                indices = torch.linspace(0, end_idx - 1, self.num_frames_temporal).long()
                
        return [Image.open(all_frames[i]).convert('RGB') for i in indices]

    def _get_laplacian(self, tensor_img):
        kernel = torch.tensor([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype=torch.float32)
        kernel = kernel.view(1, 1, 3, 3).repeat(3, 1, 1, 1)
        t, n, c, h, w = tensor_img.shape
        flat_img = tensor_img.view(t * n, c, h, w)
        with torch.no_grad():
            out = torch.nn.functional.conv2d(flat_img, kernel, groups=3, padding=1)
        return out.view(t, n, c, h, w)

    def __getitem__(self, idx):
        folder_path = self.valid_samples[idx]
        key = self._get_key_from_path(folder_path)
        
        entry = self.mos_labels[key]
        score = entry['mos'] / 100.0 if isinstance(entry, dict) else entry / 100.0
        score_tensor = torch.tensor(score, dtype=torch.float32)
            
        # ===== 1. 空间流 (Spatial Stream) 数据 =====
        # 稀疏采样 (例如 24 帧)
        frames_spatial_pil = self._load_frames_pil(folder_path, mode_type='spatial')
        cropped_frames_pil = self.dense_crop(frames_spatial_pil)
        t_imgs = []
        for frame_crops in cropped_frames_pil:
            crops_tensor = torch.stack([self.spatial_transform(c) for c in frame_crops]) 
            t_imgs.append(crops_tensor)
        content_input = torch.stack(t_imgs) # [T_s, N, C, H, W]
        x_s = self._get_laplacian(content_input) # [T_s, N, C, H, W]
        
        # ===== 2. 时间流 (Temporal Stream) 数据 =====
        # 密集采样前 80% (例如 96 帧)
        frames_temporal_pil = self._load_frames_pil(folder_path, mode_type='temporal')
        t_frames = []
        for img in frames_temporal_pil:
            t_frames.append(self.temporal_transform(img))
        x_t = torch.stack(t_frames) # [T_t, C, H', W']
        x_t = x_t.permute(1, 0, 2, 3) # [C, T_t, H', W']
                
        return x_s, x_t, score_tensor, key

    def __len__(self):
        return len(self.valid_samples)
