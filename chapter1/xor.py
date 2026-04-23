# 前面我们都是一个神经元，现在要实现通过三个神经元建模 xor
import math
import random
# 异或门的真值表
train: list[list[int]] = [
    [0,0,0],
    [0,1,1],
    [1,0,1],
    [1,1,0]
]

# 假设函数为 sigmoid(w5 * sigmoid(w1*x1+w2*x2 + b1) + w6 * sigmoid(w3*x1+w4*x2 + b2) + b3)
# 定义一个 xor 类

class Xor:
    def __init__(self,w1:float,w2:float,b1:float,w3:float,w4:float,b2:float,w5 :float,w6 :float,b3 :float):
        self.w1:float = w1
        self.w2:float = w2
        self.b1:float = b1
        self.w3:float = w3
        self.w4:float = w4
        self.b2:float = b2
        self.w5 :float = w5
        self.w6 :float = w6
        self.b3 :float = b3
    def __sigmoid(self,x:float) -> float:
        return 1 / (1 + math.exp(-x))
    def forward(self,x1 :float,x2 :float) -> float:
        return self.__sigmoid(self.w5 * self.__sigmoid(self.w1 * x1 + self.w2 * x2 + self.b1) 
                              + self.w6 * self.__sigmoid(self.w3 * x1 + self.w4 * x2 + self.b2) 
                              + self.b3)
    def loss(self) -> float:
        sum :float = 0
        for trainItem in train:
            y = self.forward(trainItem[0],trainItem[1])
            sum += (y-trainItem[2]) ** 2
        return sum/len(train)
    def train(self,batch :int,rate :float,eps :float) -> None:
        for i in range(batch):
            constLoss = self.loss()
            
            temp = self.w1
            self.w1 += eps
            dlossdw1 = (self.loss() - constLoss) / eps
            self.w1 = temp

            temp = self.w2
            self.w2 += eps
            dlossdw2 = (self.loss() - constLoss) / eps
            self.w2 = temp

            temp = self.w3
            self.w3 += eps
            dlossdw3 = (self.loss() - constLoss) / eps
            self.w3 = temp

            temp = self.w4
            self.w4 += eps
            dlossdw4 = (self.loss() - constLoss) / eps
            self.w4 = temp

            temp = self.w5
            self.w5 += eps
            dlossdw5 = (self.loss() - constLoss) / eps
            self.w5 = temp

            temp = self.w6
            self.w6 += eps
            dlossdw6 = (self.loss() - constLoss) / eps
            self.w6 = temp

            temp = self.b1
            self.b1 += eps
            dlossdb1 = (self.loss() - constLoss) / eps
            self.b1 = temp

            temp = self.b2
            self.b2 += eps
            dlossdb2 = (self.loss() - constLoss) / eps
            self.b2 = temp

            temp = self.b3
            self.b3 += eps
            dlossdb3 = (self.loss() - constLoss) / eps
            self.b3 = temp

            self.w1 -= dlossdw1 * rate
            self.w2 -= dlossdw2 * rate
            self.w3 -= dlossdw3 * rate
            self.w4 -= dlossdw4 * rate
            self.w5 -= dlossdw5 * rate
            self.w6 -= dlossdw6 * rate
            self.b1 -= dlossdb1 * rate
            self.b2 -= dlossdb2 * rate
            self.b3 -= dlossdb3 * rate

            print(f"batch: {i}, loss: {self.loss():.7f}")
            

xor :Xor = Xor(random.uniform(-1,1),random.uniform(-1,1),random.uniform(-1,1),random.uniform(-1,1),random.uniform(-1,1),random.uniform(-1,1),random.uniform(-1,1),random.uniform(-1,1),random.uniform(-1,1))
xor.train(10000,1e-1,1e-4)


