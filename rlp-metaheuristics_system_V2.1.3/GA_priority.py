import numpy as np
import copy
import random
from collections import defaultdict
import matplotlib.pyplot as plt
import time
import math
from datetime import datetime

class Activity(object):
    def __init__(self, id, duration, duration_min, duration_max, resourceRequest, successor):
        self.id = id
        self.duration = duration
        self.duration_min = duration_min
        self.duration_max = duration_max
        self.resourceRequest = np.array(resourceRequest)
        self.predecessor = None
        self.successor = successor
        self.es = 0
        self.ef = 0
        self.ls = 0
        self.lf = 0
        self.visited = False
def readData(fileName):
    f = open(fileName)
    taskAndResourceType = f.readline().split()  # 第一行数据包含活动数和资源数
    taskSum = int(taskAndResourceType[0])  # 得到活动数
    resourceType = int(taskAndResourceType[1])  # 得到资源数
    resourceAvail = np.array([int(value) for value in f.readline().split()])  # 获取资源限量
    # 将每个活动的所有信息存入到对应的Activity对象中去
    allTasks = {}
    preActDict = defaultdict(lambda: [])
    for i in range(taskSum):
        nextLine = [int(value) for value in f.readline().split()]
        task = Activity(i + 1, nextLine[0], int(nextLine[0]-math.sqrt(nextLine[0])), int(nextLine[0]+math.sqrt(nextLine[0])), nextLine[1:5], nextLine[6:])
        allTasks[task.id] = task
        for act in nextLine[6:]:
            preActDict[act].append(i + 1)
    f.close()
    # 给每个活动加上紧前活动信息
    for actKey in allTasks.keys():
        allTasks[actKey].predecessor = preActDict[allTasks[actKey].id].copy()
    return  taskSum, allTasks # 活动数int，  所有活动集合dic{活动代号：活动对象}

# 优先级列表随机-种群初始化
def generate_unique_list(m):
    first = 1
    last = m
    available_middle = list(range(2, m))
    need_middle = m - 2
    middle = random.sample(available_middle, need_middle) if need_middle > 0 else []
    return [first] + middle + [last]
# 活动优先级调度
def activity_scheduling(activities, priority_list):
    total_activities = len(priority_list)
    activity_priority = {
        act: priority_list[act - 1] for act in range(1, total_activities + 1)
    }
    completed = []        # 已完成的活动列表
    scheduled = []        # 最终调度列表
    remaining = list(range(1, total_activities + 1))  # 未调度的活动
    while len(scheduled) < total_activities:
        candidate_activities = []
        for act in remaining:
            all_pre_done = all(pre in completed for pre in activities[act].predecessor)
            if all_pre_done:
                candidate_activities.append(act)
        if not candidate_activities:
            raise ValueError(
                f"无可用的可调度活动，剩余未调度：{remaining}，已完成：{completed}\n"
                "可能原因：1. 存在循环依赖 2. 紧前关系配置错误"
            )

        candidate_activities.sort(key=lambda x: activity_priority[x])
        selected = candidate_activities[0]
        # 更新状态
        scheduled.append(selected)
        completed.append(selected)
        remaining.remove(selected)
    return scheduled
# 生成set_num组活动时间-蒙特卡洛模拟
def generate_activity_times(set_num):
    activity_times = []
    time_group = []
    for i in range(set_num):
        time_group = []
        for j in range(num_activities):
            group = random.randint(activities[j+1].duration_min, activities[j+1].duration_max)
            time_group.append(group)
        activity_times.append(time_group)
    return activity_times


# 计算项目工期
def calculate_activity_time(schedule_order, activities, durations):
    total_activities = len(durations)
    # 计算最早开始(ES)、最早完成(EF)
    activity_time = {}
    for act in schedule_order:
        if not activities[act].predecessor:
            es = 0
        else:
            es = max(activity_time[pre]["EF"] for pre in activities[act].predecessor)
        ef = es + durations[act - 1]
        activity_time[act] = {"ES": es, "EF": ef}
    project_duration = max([activity_time[act]["EF"] for act in activity_time])

    return activity_time, project_duration


# 计算资源消耗
def calculate_resource_consumption(activity_time, activities):
    project_duration = max([activity_time[act]["EF"] for act in activity_time])
    time_resource = {t: 0 for t in range(project_duration)}

    # 遍历每个活动，累加持续期内的资源消耗
    for act in activity_time:
        act_es = activity_time[act]["ES"]
        act_ef = activity_time[act]["EF"]
        act_resource = activities[act].resourceRequest
        for t in range(act_es, act_ef):
            if t in time_resource:
                time_resource[t] += act_resource
    # 总资源消耗
    total_resource = np.sum(sum(time_resource.values()))
    time_resource_list = [np.sum(time_resource[t]) for t in sorted(time_resource.keys())]
    # 总资源消耗平均值
    mean_total_resource = total_resource / len(time_resource_list)
    # 目标函数
    function = round(sum([(x - mean_total_resource) ** 2 for x in time_resource_list]) / len(time_resource_list), 6)
    return function
# 计算优先级的适应度
def calculate_priority_fitness(activities,pri,simulation_times):
    scheduled = activity_scheduling(activities, pri)
    # 蒙特卡洛模拟
    function_list = []
    fitness_list = []
    activity_groups =generate_activity_times(simulation_times)
    for i in range(len(activity_groups)):
        activity_time,project_duration = calculate_activity_time(scheduled, activities, activity_groups[i])
        function = calculate_resource_consumption(activity_time, activities)
        function_list.append(function)
        fitness = round(1/function,6)
        fitness_list.append(fitness)
    function_priority = round(sum(function_list)/len(function_list),6)
    fitness_priority = round(sum(fitness_list)/len(fitness_list),6)
    return fitness_priority
# 计算种群适应度
def calculate_population_fitness_list(population,activities,simulation_times):
    pop_fitness_list = []
    for pop in population:
        fitness_priority = calculate_priority_fitness(activities,pop,simulation_times)
        pop_fitness_list.append(fitness_priority)
    return pop_fitness_list
# 选择-染色体被选择的概率
def calculate_pop_selection_probability(population,pop_fitness_list):
    pop_fitness_sum = sum(pop_fitness_list)
    probabilities = [x / pop_fitness_sum for x in pop_fitness_list]
    probabilities = np.array(probabilities)
    probabilities /= probabilities.sum()  # 确保总和为1
    pop_selection_probability_list = probabilities.tolist()  # 转换为Python列表
    return pop_selection_probability_list
# 选择 轮盘赌 返回新的种群
def select(population, fitness,fitness_p):
    population_select_id = np.random.choice(np.arange(len(population)), len(population), p = fitness_p)
    population_select = []
    for pop_id in population_select_id:
        population_select.append(population[pop_id])
    return population_select
# 交叉
def crossover_and_mutation(num_activities,population, CROSSOVER_RATE=0.9):  # 单点交叉
    population_cross_mutation = []
    for father in population:  # 遍历种群中的每一个个体，将该个体作为父亲
        child = father
        if np.random.rand() < CROSSOVER_RATE:
            mother = population[np.random.randint(len(population))]
            cross_points = np.random.randint(low = 0, high = num_activities)
            child[cross_points:] = mother[cross_points:]
        mutation(child)  # 每个后代有一定的机率发生变异
        population_cross_mutation.append(child)
    return population_cross_mutation
# 变异
def mutation(child, MUTATION_RATE=0.05):
    for i in range(1,len(child) - 1):
        if np.random.rand() < MUTATION_RATE:  # 以MUTATION_RATE的概率进行变异
            child[i] = random.randint(2,len(child) - 1)
    return child
def GA_pri(num_activities, population_num, activities, simulation_times, best_pop_simulation_times):
    population = [generate_unique_list(num_activities) for j in range(population_num)]
    pop_fitness_list = calculate_population_fitness_list(population,activities,simulation_times)
    pop_best_pop_list = []
    pop_best_fitness_list = []
    ga_n = 0
    iter_start = time.time()  # 迭代阶段的开始时间（排除数据读取耗时）
    while True:
        # 检查是否超过时间上限
        elapsed_time = time.time() - iter_start
        if elapsed_time >= t:
            break
        pop_selection_p_list = calculate_pop_selection_probability(population,pop_fitness_list)
        population = select(population, pop_fitness_list, pop_selection_p_list)
        population = crossover_and_mutation(num_activities,population, CROSSOVER_RATE=0.9)
        pop_fitness_list = calculate_population_fitness_list(population,activities,simulation_times)
        pop_best_fitness = max(pop_fitness_list)
        best_pop_index = pop_fitness_list.index(pop_best_fitness)
        best_pop = population[best_pop_index]
        pop_best_fitness_list.append(pop_best_fitness)
        pop_best_pop_list.append(best_pop)
        ga_n += 1
    if not pop_best_fitness_list:
        print("警告：无有效迭代结果！")
        best_function = 1/max(pop_fitness_list)
    else:
        best_fitness_index = pop_best_fitness_list.index(max(pop_best_fitness_list))
        best_pop = pop_best_pop_list[best_fitness_index]
        be_f = calculate_priority_fitness(activities,best_pop,best_pop_simulation_times)
        best_function = 1 / max(be_f,max(pop_best_fitness_list))
    return best_function, ga_n
# 写入数据
import pandas as pd
def write_result(data,file_name):
    file_name = f"{file_name}.xlsx"
    df = pd.DataFrame({f'第{i+1}次训练': col for i, col in enumerate(data)})
    df.to_excel(file_name, index=False)
    return (f"数据已写入文件：{file_name}")

if __name__ == "__main__":
    now = datetime.now()
    formatted_time = now.strftime("%m%d_%H%M")
    psplib_project_num = [30,60,90,120]
    psplib_act_num = [480,480,480,600]
    # psplib_act_num = [4,3,2,1]
    # psplib_project_num = [60]
    # psplib_act_num = [1]
    TIME_LIMIT = [5.1, 9, 13.7, 18.9]  # 时间限制（秒）
    # TIME_LIMIT = [1,2,3,4]  # 迭代时间上限（秒）
    simulation_times = 10 #模拟次数
    best_pop_simulation_times = 100 #最佳种群模拟次数
    population_num = 10 #种群数量
    epetitions_n = 5 #重复次数
    best_function_list = [[] for i in range(epetitions_n)]
    cpu_time = [[] for i in range(epetitions_n)]
    for n in range(epetitions_n):
        for p, a, t in zip(psplib_project_num, psplib_act_num, TIME_LIMIT):
            for i in range(a):
                total_start = time.time()
                fileName = '/Users/yuanzhaoying/Desktop/硕士研究/code/不确定性/PSPLIB/j{0}/J{0}_{1}.RCP'.format(str(p),str(i + 1))
                num_activities,activities= readData(fileName)
                best_function, ga_n = GA_pri(num_activities, population_num, activities, simulation_times, best_pop_simulation_times)
                best_function_list[n].append(round(best_function,6))
                total_elapsed = time.time() - total_start
                cpu_time[n].append(round(total_elapsed, 6))
                print(f"第{n+1}次求解 | 实例J{p}_{i + 1} | 总耗时：{cpu_time[n][-1]:.2f}s | 迭代次数：{ga_n} | 最优方差：{best_function_list[n][-1]}")

    write_result(best_function_list,"GA_priority_result" + formatted_time)
    write_result(cpu_time,"GA_priority_time" + formatted_time)

