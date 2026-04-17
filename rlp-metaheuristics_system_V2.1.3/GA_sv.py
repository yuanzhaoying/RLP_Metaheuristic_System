import numpy as np
import copy
import random
from collections import defaultdict
import time
from datetime import datetime
class Activity(object):
    def __init__(self, id, duration, resourceRequest, successor):
        self.id = id
        self.duration = duration
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
    resourceAvail = np.array([int(value) for value in f.readline().split()])
    allTasks = {}
    preActDict = defaultdict(lambda: [])
    for i in range(taskSum):
        nextLine = [int(value) for value in f.readline().split()]
        task = Activity(i + 1, nextLine[0], nextLine[1:5], nextLine[6:])
        allTasks[task.id] = task
        for act in nextLine[6:]:
            preActDict[act].append(i + 1)
    f.close()
    for actKey in allTasks.keys():
        allTasks[actKey].predecessor = preActDict[allTasks[actKey].id].copy()
    allTasks[1].es = 0
    for i in range(2, taskSum+1):
        predecessors = allTasks[i].predecessor
        early_start_time = max([allTasks[p].es + allTasks[p].duration for p in predecessors])
        allTasks[i].es = early_start_time
    max_time = max([allTasks[j+1].es for j in range(taskSum)])
    allTasks[taskSum].ls = max_time
    for m in range(taskSum - 1, 0, -1):
        successors = allTasks[m].successor
        late_start_time = min([allTasks[n].ls - allTasks[m].duration for n in successors])
        allTasks[m].ls = late_start_time
    total_during_time = allTasks[taskSum].ls
    return  taskSum, resourceType, resourceAvail, allTasks, total_during_time  # 活动数int， 资源数int， 资源限量np.array， 所有活动集合dic{活动代号：活动对象}
def initialize_population_displacement(num_population,taskSum:int,activities_es,activities_ls):
    population=[]
    for i in range(num_population):
        activity = []
        for activity_num in range(0,taskSum ):
            displacement_time =round(random.uniform(0, 1) * (activities_ls[activity_num] - activities_es[activity_num]))
            activity.append(displacement_time)
        population.append(activity)
    return population

# 适应度函数
def fitness(population, allTasks, total_during_time, total_resource):
    ga_fitness_list = []
    ga_fitness_p_list = []
    for pop in population:
        displacement_time = pop
        total_resource_sum = 0
        time_total_resource = []
        for time in range(0, total_during_time):
            resourceSum = np.zeros(len(total_resource), dtype=int)
            for j in allTasks.keys():
                if displacement_time[j - 1] + allTasks[j].es <= time < displacement_time[j - 1] + allTasks[j].duration + \
                        allTasks[j].es:
                    resourceSum += allTasks[j].resourceRequest
            time_total_resource.append(sum(resourceSum))  # R(t)
        ga_fitness = 1 / np.var(time_total_resource)
        ga_fitness_list.append(ga_fitness)
    ga_fitness_p = sum(ga_fitness_list)
    fitness_best = max(ga_fitness_list)
    for p in ga_fitness_list:
        ga_fitness_p = p / sum(ga_fitness_list)
        ga_fitness_p_list.append(ga_fitness_p)
    return ga_fitness_list, ga_fitness_p_list, fitness_best

def select(population, fitness,fitness_p):
    population_select_id = np.random.choice(np.arange(len(population)), len(population), p = fitness_p)
    population_select = []
    for pop_id in population_select_id:
        population_select.append(population[pop_id])
    return population_select,population_select_id

def mutation(activities_es,activities_ls,child, MUTATION_RATE=0.01):
    if np.random.rand() < MUTATION_RATE:
        mutate_point = np.random.randint(0, len(child))
        child[mutate_point] = random.randint(activities_es[mutate_point],activities_ls[mutate_point])
    return child

def crossover_and_mutation(activities_es,activities_ls, num_activities,population, CROSSOVER_RATE=0.9):  # 单点交叉
    population_cross_mutation = []
    for father in population:
        child = father
        if np.random.rand() < CROSSOVER_RATE:
            mother = population[np.random.randint(len(population))]
            cross_points = np.random.randint(low = 0, high = num_activities)
            child[cross_points:] = mother[cross_points:]
        mutation(activities_es,activities_ls,child)  # 每个后代有一定的机率发生变异
        population_cross_mutation.append(child)
    return population_cross_mutation
def GA_sv(num_population,num_activities, num_resource_type, total_resource, activities, total_during_time,t):
    activities_es = []
    activities_ls = []
    for i in activities.keys():
        activities_es.append(activities[i].es)
        activities_ls.append(activities[i].ls)
    population = initialize_population_displacement(num_population,num_activities,activities_es,activities_ls)
    best_fit = []
    fit,fit_p,fitness_best = fitness(population,activities,total_during_time,total_resource)
    ga_n = 0
    iter_start = time.time()  # 迭代阶段的开始时间（排除数据读取耗时）
    while True:
        # 检查是否超过时间上限
        elapsed_time = time.time() - iter_start
        if elapsed_time >= t:
            break
        pop_select,pop_id = select(population,fit,fit_p)
        pop_corss_mation = crossover_and_mutation(activities_es,activities_ls, num_activities,pop_select, CROSSOVER_RATE=0.9)
        fit,fit_p, fitness_best = fitness(population,activities,total_during_time,total_resource)
        best_fit.append(fitness_best)
        ga_n += 1

    if not best_fit:
        print("警告：无有效迭代结果！")
        best_func = 1 / fitness_best
    else:
        best_func = 1 / max(best_fit)
    return best_func,ga_n
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
    # psplib_act_num = [2,2,1,1]
    # psplib_project_num = [60]
    # psplib_act_num = [1]
    TIME_LIMIT = [0.1,0.1,0.1,0.1]  # 迭代时间上限（秒）
    epetitions_n = 5 #重复次数
    num_population = 10
    best_function_list = [[] for i in range(epetitions_n)]
    cpu_time = [[] for i in range(epetitions_n)]
    for n in range(epetitions_n):
        for p, a, t in zip(psplib_project_num, psplib_act_num, TIME_LIMIT):
            for i in range(a):
                # print(t)
                total_start = time.time()
                fileName = '/Users/yuanzhaoying/Desktop/硕士研究/code/确定性/PSPLIB/j{0}/J{0}_{1}.RCP'.format(str(p),str(i + 1))
                num_activities, num_resource_type, total_resource, activities, total_during_time= readData(fileName)
                best_func,ga_n = GA_sv(num_population,num_activities, num_resource_type, total_resource, activities, total_during_time,t)
                interival = time.time() - total_start
                best_function_list[n].append(best_func)
                cpu_time[n].append(interival)
                print(f"第{n+1}次求解 | 实例J{p}_{i + 1} | 总耗时：{cpu_time[n][-1]:.2f}s | 迭代次数：{ga_n} | 最优方差：{best_function_list[n][-1]}")
    write_result(best_function_list,"GA_sv_result" + formatted_time)
    write_result(cpu_time,"GA_sv_time" + formatted_time)
