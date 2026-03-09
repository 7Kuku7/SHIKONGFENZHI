# ================= consts_scene_split.py (基于场景隔离的划分) =================
"""
场景级划分策略 (Scene-Level Split)：
- 严格按场景分组，同一场景的所有样本必须在同一个集合中
- 训练集和测试集没有任何场景重叠，符合质量评价领域的学术标准
- 解决数据泄露问题，确保测试的是模型对"未见过场景"的泛化能力

数据集概况 (共182个样本，11个场景)：
- 大场景 (28样本/场景): Architecture, LiZhaoJi_Building, office1b, riverview
- 小场景 (10样本/场景): Center_Garden, Sqaure1, Sqaure2, apartment, office_view1, office_view2, raf_furnishedroom

划分方案：
- 训练集 (7场景, 142样本): Architecture, LiZhaoJi_Building, office1b, riverview, Sqaure2, Center_Garden, office_view1
- 验证集 (2场景, 20样本): office_view2, raf_furnishedroom
- 测试集 (2场景, 20样本): apartment, Sqaure1
"""
from pathlib import Path
import random

# -------------------------
# 场景级划分定义
# -------------------------
# 训练场景 (7个): 4大 + 3小 = 28*4 + 10*3 = 142 样本
TRAIN_SCENES = [
    "Architecture",        # 28
    "LiZhaoJi_Building",   # 28
    "office1b",            # 28
    "riverview",           # 28
    "Sqaure2",             # 10
    "Center_Garden",       # 10
    "office_view1",        # 10
]

# 验证场景 (2个): 2小 = 20 样本
VAL_SCENES = [
    "office_view2",        # 10
    "raf_furnishedroom",   # 10
]

# 测试场景 (2个): 2小 = 20 样本 (完全未见过的场景)
TEST_SCENES = [
    "apartment",           # 10
    "Sqaure1",             # 10
]

# 所有场景
ALL_SCENES = TRAIN_SCENES + VAL_SCENES + TEST_SCENES

# -------------------------
# 方法 / 条件 / 轨迹（与目录名一致）
# -------------------------
METHODS = [
    "instant-ngp",
    "nerfacto",
    "mipnerf",
    "tensorf",
    "nerf",
]

CONDITIONS = [
    "baseline",
    "clip_0.4", "clip_0.7", "clip_1.5", "clip_2.5",
    "gamma_0.6", "gamma_1.6",
]

TRAJECTORIES = [
    "path1",
    "path2",
]

# -------------------------
# 场景级划分函数
# -------------------------
def build_scene_split(renders_root="renders", seed=None):
    """
    构建基于场景隔离的数据集划分
    
    Args:
        renders_root: 渲染数据根目录
        seed: 随机种子 (仅用于场景内样本顺序打乱，不影响场景划分)
    
    Returns:
        train_samples, val_samples, test_samples: 三个列表，每个元素是Path对象
    """
    if seed is not None:
        random.seed(seed)
    
    renders_root = Path(renders_root)
    
    # 收集所有有效样本，并按场景分组
    scene_samples = {scene: [] for scene in ALL_SCENES}
    
    for d in sorted([p for p in renders_root.iterdir() if p.is_dir()]):
        parts = d.name.split("__")
        if len(parts) != 4:
            continue
        scene, method, cond, path_name = parts
        
        # 只处理已知场景
        if scene not in ALL_SCENES:
            continue
        
        # 检查是否有足够的帧
        imgs = sorted(d.glob("frame_*.png"))
        if len(imgs) < 50:  # 至少50帧
            continue
        
        scene_samples[scene].append(d)
    
    # 根据场景类型分配到训练/验证/测试集
    train_samples = []
    val_samples = []
    test_samples = []
    
    for scene in TRAIN_SCENES:
        samples = scene_samples[scene]
        if seed is not None:
            random.shuffle(samples)
        train_samples.extend(samples)
    
    for scene in VAL_SCENES:
        samples = scene_samples[scene]
        if seed is not None:
            random.shuffle(samples)
        val_samples.extend(samples)
    
    for scene in TEST_SCENES:
        samples = scene_samples[scene]
        if seed is not None:
            random.shuffle(samples)
        test_samples.extend(samples)
    
    # 打乱训练集顺序（增加随机性）
    if seed is not None:
        random.shuffle(train_samples)
    
    # 统计信息
    n_total = len(train_samples) + len(val_samples) + len(test_samples)
    
    print(f"[场景级划分] 总样本: {n_total}")
    print(f"  - 训练集: {len(train_samples)} ({len(train_samples)/n_total*100:.1f}%) | 场景: {TRAIN_SCENES}")
    print(f"  - 验证集: {len(val_samples)} ({len(val_samples)/n_total*100:.1f}%) | 场景: {VAL_SCENES}")
    print(f"  - 测试集: {len(test_samples)} ({len(test_samples)/n_total*100:.1f}%) | 场景: {TEST_SCENES}")
    
    # 验证场景隔离
    train_scenes_actual = set([p.name.split("__")[0] for p in train_samples])
    val_scenes_actual = set([p.name.split("__")[0] for p in val_samples])
    test_scenes_actual = set([p.name.split("__")[0] for p in test_samples])
    
    train_val_overlap = train_scenes_actual & val_scenes_actual
    train_test_overlap = train_scenes_actual & test_scenes_actual
    val_test_overlap = val_scenes_actual & test_scenes_actual
    
    if train_val_overlap or train_test_overlap or val_test_overlap:
        print(f"  [警告] 检测到场景重叠!")
        if train_val_overlap: print(f"    训练-验证重叠: {train_val_overlap}")
        if train_test_overlap: print(f"    训练-测试重叠: {train_test_overlap}")
        if val_test_overlap: print(f"    验证-测试重叠: {val_test_overlap}")
    else:
        print(f"  [✓] 场景隔离验证通过: 训练/验证/测试集无场景重叠")
    
    return train_samples, val_samples, test_samples


def get_scene_from_sample(sample_path):
    """从样本路径中提取场景名"""
    parts = Path(sample_path).name.split("__")
    if len(parts) >= 1:
        return parts[0]
    return None


def print_split_details(train_samples, val_samples, test_samples):
    """打印划分的详细信息"""
    print("\n" + "="*60)
    print("数据集划分详情")
    print("="*60)
    
    for name, samples, scenes in [
        ("训练集", train_samples, TRAIN_SCENES),
        ("验证集", val_samples, VAL_SCENES),
        ("测试集", test_samples, TEST_SCENES)
    ]:
        print(f"\n{name} ({len(samples)} 样本):")
        scene_counts = {}
        for s in samples:
            scene = get_scene_from_sample(s)
            scene_counts[scene] = scene_counts.get(scene, 0) + 1
        
        for scene in scenes:
            count = scene_counts.get(scene, 0)
            print(f"  - {scene}: {count} 样本")
    
    print("\n" + "="*60)


# -------------------------
# 兼容性函数 (与 consts_simple_split 保持接口一致)
# -------------------------
def build_simple_split(renders_root="renders", train_ratio=None, val_ratio=None, seed=777):
    """
    兼容性函数：直接调用场景级划分
    train_ratio 和 val_ratio 参数被忽略（使用固定的场景划分）
    """
    return build_scene_split(renders_root, seed=seed)


# -------------------------
# 测试代码
# -------------------------
if __name__ == "__main__":
    train, val, test = build_scene_split("renders", seed=3407)
    print_split_details(train, val, test)
