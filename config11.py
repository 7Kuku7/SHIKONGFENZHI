# config.py
import os

class Config:
    # ================= 路径设置 =================
    # 数据集根目录
    ROOT_DIR = "/home/abc/wq/QQQ/renders"
    # MOS 标签文件
    MOS_FILE = "/home/abc/wq/QQQ/mos_advanced.json"
    # 结果保存主目录
    OUTPUT_DIR = "/home/abc/wq/QQQ/results/"
    
    # ================= 实验名称与描述 =================
    EXP_NAME = "Exp_v13"  # 实验组名
    DESCRIPTION = "单swin+拉普拉斯+密度裁剪+时空一致性"

    # ================= 训练超参数 =================
    SEED = 3407             # 随机种子，设置为 None 则随机
    GPU_ID = "1"          # 指定 GPU
    BATCH_SIZE = 1
    NUM_WORKERS = 4
    EPOCHS = 100
    LR = 1e-4             # 学习率
    
    # ================= 损失函数权重 (Ablation Control) =================
    # 通过设置权重为 0.0 来实现消融实验 (Ablation)
    LAMBDA_MSE = 1.0      # 主分数回归损失
    LAMBDA_RANK = 0.2     # 排序损失
    LAMBDA_SUB = 0.05      # 子任务分数损失 -> 设为0即为 w/o Multi-task
    LAMBDA_SSL = 0.1      # 自监督一致性损失 -> 设为0即为 w/o SSL

    # ================= 模型配置 =================
    NUM_FRAMES = 16        # 每个视频采样的帧数
    USE_SUBSCORES = True  # 是否使用子分数头
    USE_FUSION = True     # 是否使用特征融合模块
    
    # ================= 功能开关 =================
    ENABLE_WANDB = False        # 是否使用 WandB 记录
    WANDB_PROJECT = "OF-NeRF-QA"
    SAVE_PER_VIDEO_RESULT = True # 是否保存每个视频的预测结果JSON

    @classmethod
    def get_output_path(cls):
        """自动生成当前实验的保存路径"""
        path = os.path.join(cls.OUTPUT_DIR, cls.EXP_NAME, f"seed_{cls.SEED}")
        os.makedirs(path, exist_ok=True)
        return path

# export http_proxy=http://127.0.0.1:7897
# export https_proxy=http://127.0.0.1:7897
# export all_proxy=http://127.0.0.1:7897