# datasets/nerf_loader.py
#拉普拉斯算子 20260204新加模块，对图像进行卷积，提取高频边缘和噪声作为“失真特征”的输入。
import torch
from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image
import json
import sys
import os

# 尝试引用 split 工具
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from consts_scene_split import build_scene_split
except ImportError:
    print("Warning: consts_scene_split not found. Trying consts_simple_split...")
    try:
        from consts_simple_split import build_simple_split as build_scene_split
    except ImportError:
        print("Error: No split module found!")

class NerfDataset(Dataset):
    def __init__(self, root_dir, mos_file, mode='train', 
                 basic_transform=None, 
                 ssl_transform=None,  
                 distortion_sampling=False, 
                 num_frames=8,
                 use_subscores=True):
        
        self.root_dir = Path(root_dir)
        self.basic_transform = basic_transform
        self.ssl_transform = ssl_transform 
        self.distortion_sampling = distortion_sampling
        self.num_frames = num_frames
        self.use_subscores = use_subscores
        
        # 加载数据集划分（使用场景级划分）
        train, val, test = build_scene_split(self.root_dir)
        self.samples = {'train': train, 'val': val, 'test': test}[mode]
        
        # 加载 MOS 标签
        with open(mos_file, 'r') as f:
            self.mos_labels = json.load(f)
            
        # 过滤有效样本
        self.valid_samples = []
        for s in self.samples:
            key = self._get_key_from_path(s)
            if key in self.mos_labels:
                self.valid_samples.append(s)

    def _get_key_from_path(self, path):
        parts = path.name.split("__")
        if len(parts) == 4:
            return "+".join(parts)
        return path.name

    def _load_frames_pil(self, folder_path):
        all_frames = sorted(list(folder_path.glob("frame_*.png")))
        if not all_frames: all_frames = sorted(list(folder_path.glob("frame_*.jpg")))
        if not all_frames: all_frames = sorted([f for f in folder_path.iterdir() if f.suffix.lower() in ['.png', '.jpg', '.jpeg']])
        
        if not all_frames: 
            raise ValueError(f"No frames found in {folder_path}")
            
        indices = torch.linspace(0, len(all_frames)-1, self.num_frames).long()
        return [Image.open(all_frames[i]).convert('RGB') for i in indices]

    def _get_distortion_input(self, tensor_img):
        """
        [修正] 使用拉普拉斯算子提取高频残差图，替代原来的打乱 Patch。
        这能捕捉 NeRF 的锯齿、噪点和边缘伪影。
        Input: [T, C, H, W]
        """
        # 拉普拉斯核
        kernel = torch.tensor([[-1, -1, -1], 
                               [-1,  8, -1], 
                               [-1, -1, -1]], dtype=torch.float32)
        kernel = kernel.view(1, 1, 3, 3).repeat(3, 1, 1, 1)
        
        # 卷积操作 (保持梯度断开，仅作为预处理)
        with torch.no_grad():
            out = torch.nn.functional.conv2d(tensor_img, kernel, groups=3, padding=1)
        
        return out

    def _load_adjacent_frames_pil(self, folder_path, offset=1):
        """
        [新增] 加载相邻视角的帧序列，用于 View-Consistency SSL
        
        Args:
            folder_path: 帧目录
            offset: 时序偏移量（默认为1，即加载下一帧）
        
        Returns:
            List[PIL.Image]: 相邻视角的帧列表
        """
        all_frames = sorted(list(folder_path.glob("frame_*.png")))
        if not all_frames: all_frames = sorted(list(folder_path.glob("frame_*.jpg")))
        if not all_frames: all_frames = sorted([f for f in folder_path.iterdir() if f.suffix.lower() in ['.png', '.jpg', '.jpeg']])
        
        if not all_frames: 
            raise ValueError(f"No frames found in {folder_path}")
        
        # 原始采样索引
        indices = torch.linspace(0, len(all_frames)-1, self.num_frames).long()
        
        # 相邻帧索引（向后偏移 offset 帧，循环到开头）
        adjacent_indices = (indices + offset) % len(all_frames)
        
        return [Image.open(all_frames[i]).convert('RGB') for i in adjacent_indices]

    def __getitem__(self, idx):
        folder_path = self.valid_samples[idx]
        key = self._get_key_from_path(folder_path)
        
        entry = self.mos_labels[key]
        if isinstance(entry, dict):
            score = entry['mos'] / 100.0
            sub_data = entry.get('sub_scores', {})
        else:
            score = entry / 100.0
            sub_data = {}
            
        score_tensor = torch.tensor(score, dtype=torch.float32)
        
        sub_scores_tensor = torch.zeros(4, dtype=torch.float32)
        if self.use_subscores:
             sub_scores_tensor = torch.tensor([
                sub_data.get("discomfort", 0), sub_data.get("blur", 0),
                sub_data.get("lighting", 0), sub_data.get("artifacts", 0)
            ], dtype=torch.float32) / 5.0
            
        frames_pil = self._load_frames_pil(folder_path)
        
        # Main Branch
        t_imgs = [self.basic_transform(img) for img in frames_pil]
        content_input = torch.stack(t_imgs)
        
        # [调用修正后的失真输入生成]
        if self.distortion_sampling:
            distortion_input = self._get_distortion_input(content_input)
        else:
            distortion_input = content_input.clone()
            
        # ==========================================
        # SSL Branch - View-Consistency (多视角一致性)
        # ==========================================
        # 创新点：利用 NeRF 的渲染物理先验，使用相邻视角作为自然增强
        # 高质量 NeRF：相邻视角特征一致（3D 几何稳定）
        # 低质量 NeRF：相邻视角特征不一致（存在伪影、浮漂、几何崩塌）
        # ==========================================
        content_input_aug = torch.tensor(0.0) 
        distortion_input_aug = torch.tensor(0.0)
        
        # [核心修改] 判断是否使用 View-Consistency
        # 当 ssl_transform 不为 None 时才启用 SSL（与 config 的 LAMBDA_SSL 对应）
        use_view_consistency = True  # 可以通过 config 控制
        
        if self.ssl_transform is not None:
            # [方案选择]
            # 方案 A（传统）：使用人工图像增强（ColorJitter、模糊等）
            # 方案 B（创新）：使用相邻视角帧作为自然增强（View-Consistency）
            
            if use_view_consistency:
                # [创新] 加载相邻视角的帧（时序偏移）
                # 例如：原帧序列 [f0, f8, f16, f24, f32, f40, f48, f56]
                #       邻帧序列 [f1, f9, f17, f25, f33, f41, f49, f57]
                frames_aug_pil = self._load_adjacent_frames_pil(folder_path, offset=1)
            else:
                # [传统] 使用图像增强（如果需要对比实验）
                frames_aug_pil = self.ssl_transform(frames_pil)
            
            t_imgs_aug = [self.basic_transform(img) for img in frames_aug_pil]
            content_input_aug = torch.stack(t_imgs_aug)
            
            if self.distortion_sampling:
                distortion_input_aug = self._get_distortion_input(content_input_aug)
            else:
                distortion_input_aug = content_input_aug.clone()
        # ==========================================
                
        return content_input, distortion_input, score_tensor, sub_scores_tensor, key, content_input_aug, distortion_input_aug

    def __len__(self):
        return len(self.valid_samples)
