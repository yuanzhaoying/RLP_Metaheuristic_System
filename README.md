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
## V1.1.2:添加了禁忌搜索（TS）、路径重连（PR）算法、遗传（GA）算法及算子  

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


### 遗传算法（GA）可选算子

1. 选择算子（2种）  
- 轮盘赌  
代码：selection_strategy="roulette"  
- 锦标赛选择  
描述：随机选择k个个体，选择其中适应度最高的个体  
代码：selection_strategy="tournament"  

2. 交叉算子（4种）  
- 单点交叉  
代码：crossover_strategy="single_point"  
- 双点交叉  
描述：随机选择两个交叉点，交换两个父代在交叉点之间的部分  
代码：crossover_strategy= "two_point"  
- 资源约束交叉  
参考文献：Li, H., & Demeulemeester, E. (2016). A genetic algorithm for the robust resource leveling problem. JOURNAL OF SCHEDULING, 19(1), 43–60. https://doi.org/10.1007/s10951-015-0457-6.  
描述：寻找资源使用成本最低的时间窗口，交换在该窗口内有活动的个体  
代码：crossover_strategy="rcx"  
-	混合交叉  
参考文献：Li, H., & Demeulemeester, E. (2016). A genetic algorithm for the robust resource leveling problem. JOURNAL OF SCHEDULING, 19(1), 43–60. https://doi.org/10.1007/s10951-015-0457-6.  
描述：将父代种群分为两组，组1（前α*POP个）执行资源基交叉（RCX）；组2（后(1-α)*POP个）执行两点交叉  
代码：crossover_strategy="hybrid"; alpha=0.5  # 前50%执行RCX，后50%执行两点交叉  
3. 变异算子  
- 单点变异  
描述：检查每个活动，在ES-LS范围内随机选择新的开始时间  
代码：mutation_strategy="random"  
- 自适应变异  
参考文献：Luis Ponz-Tienda, J., Yepes, V., Pellicer, E., & Moreno-Flores, J. (2013). The Resource Leveling Problem with multiple resources using an adaptive genetic algorithm. AUTOMATION IN CONSTRUCTION, 29, 161–172. https://doi.org/10.1016/j.autcon.2012.10.003.  
描述：根据可行解比例动态调整变异概率。如果可行解比例 < FLR（0.1），变异概率减半；如果可行解比例 > FUR（0.35），变异概率加倍  
代码：mutation_strategy="adaptive", feasible_lower_rate=0.1, feasible_upper_rate=0.35  
- 混合变异算子  
参考文献：Kazemi, S., & Davari-Ardakani, H. (2020). Integrated resource leveling and material procurement with variable execution intensities. COMPUTERS & INDUSTRIAL ENGINEERING, 148. https://doi.org/10.1016/j.cie.2020.106673  
描述：模式0：交换变异-随机交换两个活动的开始时间；模式1：逆转变异-反转一段活动的开始时间；模式2：单点变异-在可行范围内随机改变活动的开始时间。每次变异时随机选择一种模式执行 
代码：mutation_strategy="hybrid"  
- 邻域变异算子  
参考文献：Kyriklidis, C., & Dounias, G. (2016). Evolutionary computation for resource leveling optimization in project management. INTEGRATED COMPUTER-AIDED ENGINEERING, 23(2), 173–184. https://doi.org/10.3233/ICA-150508  
描述：在当前值的±SP范围内随机选择  
代码：mutation_strategy="neighborhood",neighborhood_size=2  # 在当前值±2范围内变异  

4.是否启用精英保留  
描述：将父代和子代合并，选择最优的POP个个体  
代码：elitism=True or False  
5.局部搜索   
- 双遍局部搜索  
参考文献：Li, H., Xiong, L., Liu, Y., & Li, H. (2018). An effective genetic algorithm for the resource levelling problem with generalised precedence relations. INTERNATIONAL JOURNAL OF PRODUCTION RESEARCH, 56(5), 2054–2075. https://doi.org/10.1080/00207543.2017.1355120.   
描述：先升序再降序遍历，确保充分搜索。每隔N代执行一次，平衡计算成本和解质量。算法结束时对最优解再执行一次局部搜索  
代码：use_local_search=True/False, local_search_interval=10  

