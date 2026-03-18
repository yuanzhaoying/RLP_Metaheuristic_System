#!/bin/bash
# 运行自定义实例数量的示例脚本

echo "============================================================"
echo "运行自定义实例数量示例"
echo "============================================================"
echo ""

# 示例1: 运行40个j30、30个j60、20个j90、10个j120
echo "示例1: 运行 40个j30 + 30个j60 + 20个j90 + 10个j120"
echo "命令: python main.py --subset custom --j30-count 40 --j60-count 30 --j90-count 20 --j120-count 10"
echo ""

# 示例2: 只运行j30和j60
echo "示例2: 只运行 50个j30 + 50个j60"
echo "命令: python main.py --subset custom --j30-count 50 --j60-count 50"
echo ""

# 示例3: 运行所有实例
echo "示例3: 运行所有2040个实例"
echo "命令: python main.py --subset all"
echo ""

# 示例4: 运行单个子集
echo "示例4: 只运行j30的所有实例"
echo "命令: python main.py --subset j30"
echo ""

echo "============================================================"
echo "计算运行时间"
echo "============================================================"

# 计算示例1的运行时间
j30_count=40
j60_count=30
j90_count=20
j120_count=10
total_instances=$((j30_count + j60_count + j90_count + j120_count))
algorithms=3
seeds=2
time_per_run=60
total_runs=$((total_instances * algorithms * seeds))
total_time_hours=$(echo "scale=1; $total_runs * $time_per_run / 3600" | bc)

echo ""
echo "示例1的运行配置:"
echo "  - j30实例数: $j30_count"
echo "  - j60实例数: $j60_count"
echo "  - j90实例数: $j90_count"
echo "  - j120实例数: $j120_count"
echo "  - 总实例数: $total_instances"
echo "  - 算法数: $algorithms"
echo "  - 种子数: $seeds"
echo "  - 总运行次数: $total_runs"
echo "  - 每次运行时间: ${time_per_run}秒"
echo "  - 预计总时间: ${total_time_hours}小时"
echo ""

echo "开始运行示例1..."
python main.py --subset custom --j30-count 40 --j60-count 30 --j90-count 20 --j120-count 10
