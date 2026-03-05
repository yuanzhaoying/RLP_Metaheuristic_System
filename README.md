# RLP_Metaheuristic_System
RLP Meta heuristic Algorithm Selection System based on PSPLIB  
## V1:完成了手动输入实例数量，基础逻辑正确
1.元启发算法：sa、ils、ga  
2.编码：开始时间编码  
3.GA算子
- 选择：轮盘赌
- 交叉：单点交叉（随机选择交叉点，交换两个父代的后半部分）
- 变异：单点变异（对每个活动，以概率mutation_rate变异，在ES-LS范围内随机选择新的开始时间）

4.SA算子
- 邻域算子：单点变异（随机选择一个活动，在ES-LS范围内随机选择新的开始时间）

5.ils算子：
- 局部搜索算子：逐活动优化（对每个活动，在ES-LS范围内搜索最优开始时间，使用first-improvement策略）
- 扰动算子：多点变异（随机选择perturbation_strength个活动，对每个选中的活动，在ES-LS范围内随机选择新的开始时间）




