"""
测试特征选择功能
验证是否只输出选中的特征
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.psp.psplib_io import load_psplib_sm
from src.psp.features import FeatureExtractor

def test_feature_selection():
    """测试特征选择功能"""
    
    print("="*60)
    print("测试特征选择功能")
    print("="*60)
    
    # 加载实例
    instance_file = "data/psplib_raw/j30/J30_1.RCP"
    print(f"\n加载实例: {instance_file}")
    inst = load_psplib_sm(instance_file)
    
    # 计算horizon
    n = inst.n_activities
    es = [0] * n
    for j in range(n):
        for pred in inst.predecessors[j]:
            es[j] = max(es[j], es[pred] + inst.durations[pred])
    critical_path_length = max([es[i] + inst.durations[i] for i in range(n)])
    horizon = int(critical_path_length)
    
    print(f"实例信息: {n}个活动, horizon={horizon}")
    
    # 提取所有特征
    extractor = FeatureExtractor(inst, horizon)
    all_features = extractor.extract_all()
    
    print(f"\n提取的所有特征 (共{len(all_features)}个):")
    print("-" * 60)
    for i, (key, value) in enumerate(all_features.items(), 1):
        print(f"{i:2d}. {key:30s} = {value}")
    
    # 模拟GUI中选择部分特征
    print("\n" + "="*60)
    print("测试场景1: 只选择结构特征和资源特征")
    print("="*60)
    
    selected_features = [
        "n_activities", "n_resources", "n_edges", "critical_path_len",
        "capacity_mean", "demand_mean", "resource_factor"
    ]
    
    print(f"\n选中的特征 ({len(selected_features)}个):")
    for i, feature in enumerate(selected_features, 1):
        print(f"{i}. {feature}")
    
    # 过滤特征
    filtered_features = {k: v for k, v in all_features.items() if k in selected_features}
    
    print(f"\n输出的特征 ({len(filtered_features)}个):")
    print("-" * 60)
    for i, (key, value) in enumerate(filtered_features.items(), 1):
        print(f"{i}. {key:30s} = {value}")
    
    # 验证
    print("\n验证结果:")
    if len(filtered_features) == len(selected_features):
        print(f"✓ 正确! 输出特征数量 ({len(filtered_features)}) 等于选中特征数量 ({len(selected_features)})")
    else:
        print(f"✗ 错误! 输出特征数量 ({len(filtered_features)}) 不等于选中特征数量 ({len(selected_features)})")
    
    # 测试场景2
    print("\n" + "="*60)
    print("测试场景2: 只选择松弛时间特征")
    print("="*60)
    
    selected_features_2 = [
        "slack_mean", "slack_std", "slack_min", "slack_max", "critical_activity_ratio"
    ]
    
    print(f"\n选中的特征 ({len(selected_features_2)}个):")
    for i, feature in enumerate(selected_features_2, 1):
        print(f"{i}. {feature}")
    
    # 过滤特征
    filtered_features_2 = {k: v for k, v in all_features.items() if k in selected_features_2}
    
    print(f"\n输出的特征 ({len(filtered_features_2)}个):")
    print("-" * 60)
    for i, (key, value) in enumerate(filtered_features_2.items(), 1):
        print(f"{i}. {key:30s} = {value}")
    
    # 验证
    print("\n验证结果:")
    if len(filtered_features_2) == len(selected_features_2):
        print(f"✓ 正确! 输出特征数量 ({len(filtered_features_2)}) 等于选中特征数量 ({len(selected_features_2)})")
    else:
        print(f"✗ 错误! 输出特征数量 ({len(filtered_features_2)}) 不等于选中特征数量 ({len(selected_features_2)})")
    
    # 测试场景3
    print("\n" + "="*60)
    print("测试场景3: 跨类别选择特征")
    print("="*60)
    
    selected_features_3 = [
        "n_activities", "critical_path_len",  # 结构特征
        "capacity_mean", "resource_factor",    # 资源特征
        "slack_mean",                          # 松弛时间特征
        "duration_mean", "network_complexity"  # 网络拓扑特征
    ]
    
    print(f"\n选中的特征 ({len(selected_features_3)}个):")
    for i, feature in enumerate(selected_features_3, 1):
        print(f"{i}. {feature}")
    
    # 过滤特征
    filtered_features_3 = {k: v for k, v in all_features.items() if k in selected_features_3}
    
    print(f"\n输出的特征 ({len(filtered_features_3)}个):")
    print("-" * 60)
    for i, (key, value) in enumerate(filtered_features_3.items(), 1):
        print(f"{i}. {key:30s} = {value}")
    
    # 验证
    print("\n验证结果:")
    if len(filtered_features_3) == len(selected_features_3):
        print(f"✓ 正确! 输出特征数量 ({len(filtered_features_3)}) 等于选中特征数量 ({len(selected_features_3)})")
    else:
        print(f"✗ 错误! 输出特征数量 ({len(filtered_features_3)}) 不等于选中特征数量 ({len(selected_features_3)})")
    
    print("\n" + "="*60)
    print("测试完成!")
    print("="*60)
    
    return True

if __name__ == "__main__":
    test_feature_selection()
