import sys
if __name__ == "__main__":
    # 读取第一行的n
    n = int(sys.stdin.readline().strip())
    m = []
    for i in range(2 * n):
        # 读取每一行
        line = sys.stdin.readline().strip()
        # 把每一行的数字分隔后转化成int列表
        values = list(map(int, line.split()))
        m.append(values)
n = 3
m = [
    [3, 1, 1, 3],
    [5, 1, 7],
    [1, 1, 1, 5],
    [9],
    [1, 0, 1, 10],
    [3]
]
for i in range(n):
    # print(max_m)
    result = []
    a = m[2*i][1]
    b = m[2*i][2]
    k = m[2*i][3]
    # print(a, b, k)
    
    sum_r = sum(m[2*i +1])
    while a != 0:

        max_m = max(m[2*i +1])
        m[2*i +1].pop()
        max_s = int(max_m/2)
        m[2*i +1].append(max_s)
        print(m[2*i +1])
        sum_r = sum_r  - max_m+ max_s
        a =a -1
    while b != 0:
        sum_r = sum_r -k
        b-=1
    print(sum_r)



x = [2,3,4,5]
x.pop()
print(x)