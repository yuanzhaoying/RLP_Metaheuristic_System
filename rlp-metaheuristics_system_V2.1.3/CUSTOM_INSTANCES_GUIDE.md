# 运行自定义实例数量指南

## 📋 概述

现在支持手动控制每个子集运行的实例数量，可以灵活配置实验规模。

---

## 🎯 使用方式

### 方式1: 命令行参数

```bash
python main.py --subset custom --j30-count 40 --j60-count 30 --j90-count 20 --j120-count 10
```

### 方式2: 使用脚本

```bash
./run_custom_instances.sh
```

---

## 📊 参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--subset` | 子集选择 | j30 |
| `--j30-count` | j30实例数量 | 0 (全部) |
| `--j60-count` | j60实例数量 | 0 (全部) |
| `--j90-count` | j90实例数量 | 0 (全部) |
| `--j120-count` | j120实例数量 | 0 (全部) |

---

## 💡 使用示例

### 示例1: 运行自定义数量
```bash
# 运行 40个j30 + 30个j60 + 20个j90 + 10个j120
python main.py --subset custom --j30-count 40 --j60-count 30 --j90-count 20 --j120-count 10
```

**运行配置**:
- 实例数: 100
- 算法数: 3 (ga, sa, ils)
- 种子数: 2
- 总运行次数: 600
- 预计时间: 10小时

### 示例2: 只运行部分子集
```bash
# 只运行 50个j30 + 50个j60
python main.py --subset custom --j30-count 50 --j60-count 50
```

**运行配置**:
- 实例数: 100
- 总运行次数: 600
- 预计时间: 10小时

### 示例3: 运行所有实例
```bash
# 运行所有2040个实例
python main.py --subset all
```

**运行配置**:
- 实例数: 2040
- 总运行次数: 12240
- 预计时间: 204小时

### 示例4: 运行单个子集
```bash
# 只运行j30的所有实例
python main.py --subset j30

# 只运行j60的所有实例
python main.py --subset j60
```

---

## ⏱️ 时间计算

### 计算公式
```
总运行次数 = 实例数 × 算法数 × 种子数
总时间(小时) = 总运行次数 × 每次运行时间(秒) / 3600
```

### 示例计算

**配置**: 40个j30 + 30个j60 + 20个j90 + 10个j120
```
实例数 = 40 + 30 + 20 + 10 = 100
算法数 = 3 (ga, sa, ils)
种子数 = 2
每次运行时间 = 60秒

总运行次数 = 100 × 3 × 2 = 600
总时间 = 600 × 60 / 3600 = 10小时
```

---

## 🔧 高级选项

### 跳过实验，只运行统计分析
```bash
python main.py --subset custom --j30-count 40 --skip-experiments
```

### 跳过ML训练
```bash
python main.py --subset custom --j30-count 40 --skip-ml
```

### 使用延迟因子
```bash
python main.py --subset custom --j30-count 40 --use-delay-factors
```

---

## 📁 输出文件

实验结果保存在:
```
results/raw/
├── runs.csv          # 所有运行结果
├── statistics.csv    # 统计分析结果
└── selector_model.pkl # 算法选择器模型
```

---

## ⚠️ 注意事项

1. **时间估算**: 实际运行时间可能因硬件性能而异
2. **内存需求**: j120实例较大，需要更多内存
3. **并行运行**: 可以同时运行多个实验，但注意资源限制
4. **数据完整性**: 确保PSPLIB数据集完整

---

## 📊 PSPLIB数据集统计

| 子集 | 实例数 | 活动数 | 说明 |
|------|--------|--------|------|
| j30 | 480 | 30 | 小规模 |
| j60 | 480 | 60 | 中规模 |
| j90 | 480 | 90 | 中大规模 |
| j120 | 600 | 120 | 大规模 |
| **总计** | **2040** | - | - |

---

## 🚀 快速开始

```bash
# 1. 测试运行 (10个实例)
python main.py --subset custom --j30-count 10

# 2. 小规模实验 (100个实例)
python main.py --subset custom --j30-count 40 --j60-count 30 --j90-count 20 --j120-count 10

# 3. 完整实验 (所有实例)
python main.py --subset all
```
