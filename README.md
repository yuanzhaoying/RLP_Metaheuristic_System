# RLP_Metaheuristic_System
RLP Meta heuristic Algorithm Selection System based on PSPLIB  
/Users/yuanzhaoying/Documents/GitHub/  
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
## V1.1.2:添加了禁忌搜索（TS）、路径重连（PR）算法及算子  

### TS可选算子

1.禁忌表长度（2种）  
- 静态禁忌列表  
描述：静态禁忌列表  
代码：strategy = ‘static’ 
- 动态禁忌列表  
参考文献：Li, H., Xu, Z., & Demeulemeester, E. (2015). Scheduling Policies for the Stochastic Resource Leveling Problem. JOURNAL OF CONSTRUCTION ENGINEERING AND MANAGEMENT, 141(2). https://doi.org/10.1061/(ASCE)CO.1943-7862.0000936  
描述：初始为固定值n（活动数量），当 “无改进迭代次数” 超过阈值（nr_noimprove==10）时，动态调整长度：从均匀分布(√n ,4√n)中随机选取新长度，增强搜索多样性，避免陷入局部最优。  
代码：strategy = ‘dynamic’

### PR可选算子

参考文献：Ranjbar, M. (2013). A path-relinking metaheuristic for the resource levelling problem. JOURNAL OF THE OPERATIONAL RESEARCH SOCIETY, 64(7), 1071–1078. https://doi.org/10.1057/jors.2012.119.  

1.局部搜索 

代码：use_local_search: bool = False or True  

2.解选择策略（2种）  
- 评估所有解   
描述：选择路径上的最优解    
代码：selection_strategy: str = "best"   
- 随机选择两个解  
描述：从路径上 随机选择两个解（Su, Sv），然后选优加入解集   
代码：selection_strategy: str = "random_two"  

3.路径方向（4种）
- 正向探索  
描述：从初始解到目标解  
代码：path_strategy: str = "forward"
- 反向探索  
描述：从目标解到初始解  
代码：path_strategy: str = "backward"
- 随机探索  
描述：随机顺序调整活动  
代码：path_strategy: str = "random"
- 双向探索  
描述：随机选择两个解，然后选优（先S0→S00，再S00→S0）  
代码：path_strategy: str = "bidirectional"



