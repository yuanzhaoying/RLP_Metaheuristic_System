"""
自定义算子组合配置示例

这个文件展示了如何自定义算子组合配置
"""

def generate_custom_algorithm_configs():
    """生成自定义的算子组合配置"""
    configs = []
    
    # ========================================
    # 示例1：只运行DE算法的特定变异策略
    # ========================================
    de_mutation_strategies = [
        ("rand/1", True),   # rand/1 with adaptive F
        ("best/1", True),   # best/1 with adaptive F
    ]
    
    de_crossover_strategies = [
        ("bin", True),      # bin with adaptive CR
        ("exp", False),     # exp with fixed CR
    ]
    
    for mut_strat, use_adapt_F in de_mutation_strategies:
        for cross_strat, use_adapt_CR in de_crossover_strategies:
            config_name = f"DE_{mut_strat}_{'adaptiveF' if use_adapt_F else 'fixedF'}_{cross_strat}_{'adaptiveCR' if use_adapt_CR else 'fixedCR'}"
            configs.append((config_name, "DE", {
                "mutation_strategy": mut_strat,
                "use_adaptive_F": use_adapt_F,
                "crossover_strategy": cross_strat,
                "use_adaptive_CR": use_adapt_CR,
                "use_local_search": False,
            }))
    
    # ========================================
    # 示例2：只运行GA算法的特定配置
    # ========================================
    ga_configs = [
        # 配置1：轮盘赌 + 单点交叉 + 随机变异
        {
            "selection_strategy": "roulette",
            "crossover_strategy": "single_point",
            "mutation_strategy": "random",
            "initialization_strategy": "random",
            "use_local_search": True,
            "neighborhood_size": 2,
        },
        # 配置2：锦标赛 + 两点交叉 + 自适应变异
        {
            "selection_strategy": "tournament",
            "crossover_strategy": "two_point",
            "mutation_strategy": "adaptive",
            "initialization_strategy": "heuristic",
            "use_local_search": False,
            "neighborhood_size": 0,
        },
    ]
    
    for i, params in enumerate(ga_configs, 1):
        config_name = f"GA_custom_{i}"
        configs.append((config_name, "GA", params))
    
    # ========================================
    # 示例3：运行所有算法的默认配置
    # ========================================
    configs.append(("DE_default", "DE", {
        "mutation_strategy": "rand/1",
        "use_adaptive_F": False,
        "crossover_strategy": "bin",
        "use_adaptive_CR": False,
        "use_local_search": False,
    }))
    
    configs.append(("GA_default", "GA", {
        "selection_strategy": "roulette",
        "crossover_strategy": "single_point",
        "mutation_strategy": "random",
        "initialization_strategy": "random",
        "use_local_search": False,
        "neighborhood_size": 0,
    }))
    
    configs.append(("SA_default", "SA", {}))
    configs.append(("ILS_default", "ILS", {"perturbation_strength": 5}))
    configs.append(("TS_default", "TS", {"tabu_strategy": "static"}))
    configs.append(("PR_default", "PR", {
        "path_strategy": "forward",
        "selection_strategy": "best",
        "use_local_search": False,
    }))
    configs.append(("GSA_default", "GSA", {}))
    
    return configs


# ========================================
# 使用示例
# ========================================

if __name__ == "__main__":
    configs = generate_custom_algorithm_configs()
    
    print("=" * 80)
    print("自定义算子组合配置")
    print("=" * 80)
    print(f"\n总组合数: {len(configs)}")
    print("\n配置详情:")
    for i, (config_name, algo_type, params) in enumerate(configs, 1):
        print(f"\n{i}. {config_name} ({algo_type})")
        for key, value in params.items():
            print(f"   - {key}: {value}")
