#!/usr/bin/env python3
"""
验证场景级划分是否正确
运行: python verify_scene_split.py

检查项目：
1. 场景是否正确划分到训练/验证/测试集
2. 是否存在场景重叠（数据泄露）
3. 每个集合的样本数量统计
"""
from consts_scene_split import (
    build_scene_split, 
    print_split_details,
    TRAIN_SCENES, 
    VAL_SCENES, 
    TEST_SCENES
)
from pathlib import Path

def verify():
    print("="*70)
    print("场景级划分验证工具")
    print("="*70)
    
    # 1. 打印预定义的场景划分
    print("\n[预定义场景划分]")
    print(f"  训练场景 ({len(TRAIN_SCENES)}个): {TRAIN_SCENES}")
    print(f"  验证场景 ({len(VAL_SCENES)}个): {VAL_SCENES}")
    print(f"  测试场景 ({len(TEST_SCENES)}个): {TEST_SCENES}")
    
    # 2. 调用划分函数
    print("\n" + "-"*70)
    train_samples, val_samples, test_samples = build_scene_split("renders", seed=3407)
    
    # 3. 打印详细信息
    print_split_details(train_samples, val_samples, test_samples)
    
    # 4. 验证场景隔离
    print("\n[场景隔离验证]")
    train_scenes_actual = set([p.name.split("__")[0] for p in train_samples])
    val_scenes_actual = set([p.name.split("__")[0] for p in val_samples])
    test_scenes_actual = set([p.name.split("__")[0] for p in test_samples])
    
    train_test_overlap = train_scenes_actual & test_scenes_actual
    train_val_overlap = train_scenes_actual & val_scenes_actual
    val_test_overlap = val_scenes_actual & test_scenes_actual
    
    all_ok = True
    if train_test_overlap:
        print(f"  [错误] 训练-测试场景重叠: {train_test_overlap}")
        all_ok = False
    else:
        print(f"  [通过] 训练-测试集无场景重叠")
    
    if train_val_overlap:
        print(f"  [错误] 训练-验证场景重叠: {train_val_overlap}")
        all_ok = False
    else:
        print(f"  [通过] 训练-验证集无场景重叠")
    
    if val_test_overlap:
        print(f"  [错误] 验证-测试场景重叠: {val_test_overlap}")
        all_ok = False
    else:
        print(f"  [通过] 验证-测试集无场景重叠")
    
    # 5. 总结
    print("\n" + "="*70)
    if all_ok:
        print("验证结果: 通过！所有场景隔离检查均通过。")
        print("可以放心使用此划分方案进行训练。")
    else:
        print("验证结果: 失败！存在场景重叠，请检查划分配置。")
    print("="*70)
    
    # 6. 提供示例命令
    print("\n[下一步]")
    print("如果验证通过，运行以下命令开始训练:")
    print("  python main11.py")
    print("\n训练完成后，运行以下命令测试:")
    print("  python test11.py --run_dir <你的run目录路径>")
    
    return all_ok

if __name__ == "__main__":
    verify()
