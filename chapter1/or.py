import random 
import math
# 让我们先实现一个 或门

#  或门的真值表
train :list[list[int]] = [
    [0,0,0],
    [0,1,1],
    [1,0,1],
    [1,1,1]
]

# 设或门关系式为 y = sigmoid(w1*x1+w2*x2 + b)

# 定义sigmoid函数
def sigmoid(x :float) -> float:
    return 1 / (1 + math.exp(-x))

# 定义一个关系式函数，在w1，w2，b已知的情况下，根据输入，得到输出
def forward(w1 :float,w2 :float,b :float,x1 :float,x2 :float) -> float:
    return sigmoid(w1 * x1 + w2 * x2 + b)

# 定义一个损失函数 因为 train 是 全局变量 所以 可以 不设置 train 参数 返回值为平均残差方
def loss(w1 :float,w2 :float,b :float) -> float:
    sum :float = 0
    average :float = 0
    for i in range(len(train)):
        y = forward(w1,w2,b,train[i][0],train[i][1])
        sum += (y-train[i][2]) ** 2
    average = sum / len(train)
    return average 


# 随机生成数字
w1:float
w2:float
b:float

w1,w2,b = [random.uniform(1,10) for i in range(3)]

# 输出还未训练的参数值和损失值
print(f"batch: {0},w1: {w1:.4f},w2: {w2:.4f},b: {b:.4f},loss :{loss(w1,w2,b):.4f}")

# 通过有限差分求loss函数极小值

# 每次差分的步长
eps :float = 1e-3

# 每次的移动的比率
rate :float = 1

# 训练次数
batch :int = 100000

for i in range(batch):
    DlossDw1 = (loss(w1+eps,w2,b) - loss(w1,w2,b)) / eps
    w1 -= DlossDw1 * rate
    DlossDw2 = (loss(w1,w2+eps,b) - loss(w1,w2,b)) / eps
    w2 -= DlossDw2 * rate
    DlossDb = (loss(w1,w2,b+eps) - loss(w1,w2,b)) / eps
    b -= DlossDb * rate
    print(f"batch: {i},w1: {w1:.7f},w2: {w2:.7f},b: {b:.7f},loss :{loss(w1,w2,b):.7f}")

# 输出真值表输入在模型下的输出
for trainItem in train:
    print(f"x1: {trainItem[0]},x2: {trainItem[1]},forward: {forward(w1,w2,b,trainItem[0],trainItem[1])},y: {trainItem[2]}")
