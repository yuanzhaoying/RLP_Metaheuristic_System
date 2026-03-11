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

### 差分进化（DE）
1. 变异算子（6种）  
- rand/1  
描述：随机选择3个个体，v = x_r1 + F * (x_r2 - x_r3)  
代码：mutation_strategy="rand/1"  
- best/1  
描述：最优个体x_best，随机选择2个个体，v = x_best + F * (x_r1 - x_r2)  
代码：mutation_strategy=" best /1"  
- rand/2  
描述：随机选择5个个体，v = x_r1 + F * (x_r2 - x_r3) + F * (x_r4 - x_r5)  
代码：mutation_strategy="rand/2"  
- best/2  
描述：最优个体x_best，随机选择4个个体，v = x_best + F * (x_r1 - x_r2) + F * (x_r3 - x_r4)  
代码：mutation_strategy=" best /2"  
- adaptive  
参考文献：Li, H., Zheng, L., Chen, R., & Zhang, X. (2024). Stochastic resource leveling in projects with flexible structures. COMPUTERS & OPERATIONS RESEARCH, 169. https://doi.org/10.1016/j.cor.2024.106753  
描述：自适应混合变异策略：以概率L选择rand/1策略，以概率1-L选择best/1 + 差分向量策略，L随迭代次数指数衰减  
代码：mutation_strategy="adaptive"  
- current-to-rand/2  
参考文献：庞南生, 纪昌明, & 乞建勋. (2009). 基于MDE资源分时段与活动平移并行的均衡优化. 中国管理科学, 17(06), 130–138. https://doi.org/10.16381/j.cnki.issn1003-207x.2009.06.005  
描述：当前个体x_current，随机选择4个个体，v = x_current + K * (x_r1 - x_r2 + x_r3 - x_r4)，K为自适应参数，K = K0 * (2 ** np.exp(1 - t_max/(t_max+1-t)))  
代码：mutation_strategy="current-to-rand/2"  
2. 交叉算子（2种）  
- 二项交叉  
描述：对每个位置，以概率CR选择变异向量的值  
代码：crossover_strategy="bin"  
- 指数交叉  
描述：从随机位置开始，连续选择变异向量的值，直到随机数>=CR   
代码：crossover_strategy="exp"  

3.局部搜索  
- use_local_search  
参考文献：Li, H., Zheng, L., Chen, R., & Zhang, X. (2024). Stochastic resource leveling in projects with flexible structures. COMPUTERS & OPERATIONS RESEARCH, 169. https://doi.org/10.1016/j.cor.2024.106753  
描述：对前local_search_top个最优个体执行交换操作，随机选择两个位置进行交换，修复约束违反  
代码：use_local_search=False or True  

4.自适应参数（2种）  
参考文献：Li, H., Zheng, L., Chen, R., & Zhang, X. (2024). Stochastic resource leveling in projects with flexible structures. COMPUTERS & OPERATIONS RESEARCH, 169. https://doi.org/10.1016/j.cor.2024.106753  
- 变异算子自适应参数  
描述：针对== rand/1、rand/2、best/1和best/2 ==生效。F = F_max * exp(iteration * log(F_min/F_max) / max_iterations)  
代码：use_adaptive_F: bool = False or True  
- 交叉算子自适应参数  
描述：针对所有交叉算子生效。CR = CR_min * exp(iteration * log(CR_max/CR_min) / max_iterations)  
代码：use_adaptive_CR: bool = False or True  



