# 实现了 y = 2x 的 机器学习
import random

def loss(train :list,k :float) -> float :
    sum :float = 0
    for trainItem in train:
        y = k * trainItem[0]
        dist = y - trainItem[1]
        sum += dist ** 2
    averge = sum/len(train)
    return averge

train :list = [[0,0],[1,2],[2,4],[3,6],[4,8]]
# 假设 y = k * x
k :float = random.uniform(1.0,10.0)
average = loss(train,k)
print(f"平均差距为{average:.2f},k为{k:.2f}")

episo :float = 0.001
rate :float = 0.001
for i in range(1000):
    distLoss :float = ( loss(train,k + episo) - loss(train,k) ) / episo
    k -= distLoss * rate
    average = loss(train,k)
    print(f"平均差距为{average:.2f},k为{k:.2f}.distLoss为:{distLoss:.2f}")

