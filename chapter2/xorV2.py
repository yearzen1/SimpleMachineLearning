# 用自己实现的矩阵库，重新实现xor模型
from matrix import Matrix
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
# xor模型 可以看成 2 2 1 的 神经网络

class Xor:
    def __init__(self):
        self.a0 :Matrix = None
        self.a1 :Matrix = None
        self.a2 :Matrix = None
        self.w1 :Matrix = Matrix(2,2,"w1")
        self.w2 :Matrix = Matrix(2,1,"w2")
        self.b1 :Matrix = Matrix(1,2,"b1")
        self.b2 :Matrix = Matrix(1,1,"b2")
        self.w1.Fulfill(random.uniform(1,1))
        self.w2.Fulfill(random.uniform(1,1))
        self.b1.Fulfill(random.uniform(1,1))
        self.b2.Fulfill(random.uniform(1,1))

    def __sigmoid(self,x:float) -> float:
        return 1 / (1 + math.exp(-x))
    
    def forward(self,trainInputItem :Matrix) -> Matrix:
        self.a0 = trainInputItem
        self.a1 = self.a0 * self.w1 + self.b1
        for i in range(self.a1.rows):
            for j in range(self.a1.columns):
                self.a1.SetAt(i,j,self.__sigmoid(self.a1.GetAt(i,j)))
        self.a2 = self.a1 * self.w2 + self.b2
        for i in range(self.a2.rows):
            for j in range(self.a2.columns):
                self.a2.SetAt(i,j,self.__sigmoid(self.a2.GetAt(i,j)))
        return self.a2
        
    def loss(self,trainInput :Matrix,trainOutput :Matrix) -> float:
        sum :Matrix = Matrix(1,trainOutput.columns,"sum")
        for i in range(trainInput.rows):
            y :Matrix = self.forward(trainInput.getSubMatrixByRow(i))
            diff :Matrix = y - trainOutput.getSubMatrixByRow(i)
            for i in range(diff.columns):
                diff.SetAt(0,i,diff.GetAt(0,i) ** 2)
            sum += diff
        result :float = 0
        for i in range(sum.columns):
            result += sum.GetAt(0,i)
        return result / trainInput.rows
    @staticmethod
    def finiteDiff(lossaddedEps :float,loss :float,eps :float) -> float:
        return (lossaddedEps - loss) / eps
        
    def train(self,rate :float,eps :float,trainInput :Matrix,trainOutput :Matrix) -> None:
        constLoss :float = self.loss(trainInput,trainOutput)
        Saved :float = None

        gw1 :Matrix = Matrix(self.w1.rows,self.w1.columns,"gw1")
        for i in range(self.w1.rows):
            for j in range(self.w1.columns):
                Saved = self.w1.GetAt(i,j)
                self.w1.SetAt(i,j,self.w1.GetAt(i,j)+eps)
                gw1.SetAt(i,j,self.finiteDiff(self.loss(trainInput,trainOutput),constLoss,eps))
                self.w1.SetAt(i,j,Saved)
        
        gw2 :Matrix = Matrix(self.w2.rows,self.w2.columns,"gw2")
        for i in range(self.w2.rows):
            for j in range(self.w2.columns):
                Saved = self.w2.GetAt(i,j)
                self.w2.SetAt(i,j,self.w2.GetAt(i,j)+eps)
                gw2.SetAt(i,j,self.finiteDiff(self.loss(trainInput,trainOutput),constLoss,eps))
                self.w2.SetAt(i,j,Saved)

        gb1 :Matrix = Matrix(self.b1.rows,self.b1.columns,"gb1")
        for i in range(self.b1.rows):
            for j in range(self.b1.columns):
                Saved = self.b1.GetAt(i,j)
                self.b1.SetAt(i,j,self.b1.GetAt(i,j)+eps)
                gb1.SetAt(i,j,self.finiteDiff(self.loss(trainInput,trainOutput),constLoss,eps))
                self.b1.SetAt(i,j,Saved)

        gb2 :Matrix = Matrix(self.b2.rows,self.b2.columns,"gb2")
        for i in range(self.b2.rows):
            for j in range(self.b2.columns):
                Saved = self.b2.GetAt(i,j)
                self.b2.SetAt(i,j,self.b2.GetAt(i,j)+eps)
                gb2.SetAt(i,j,self.finiteDiff(self.loss(trainInput,trainOutput),constLoss,eps))
                self.b2.SetAt(i,j,Saved)
        self.w1 = self.w1 - (gw1 * rate)
        self.w2 = self.w2 - (gw2 * rate)
        self.b1 = self.b1 - (gb1 * rate)
        self.b2 = self.b2 - (gb2 * rate)
        print(f"loss: {constLoss:.7f}")    

train1 :Matrix = Matrix(4,3,"train1")
train1.matrix = train
traininput :Matrix = train1.getSubMatrix(0,0,3,1)
trainoutput :Matrix = train1.getSubMatrix(0,2,3,2)
# traininput.print()
# trainoutput.print()
xor :Xor = Xor()
for i in range(10000):
    print(f"batch:{i}",end = " ")
    xor.train(1e-1,1e-4,traininput,trainoutput)
# xor.w1.print()
# xor.w2.print()
# xor.b1.print()
# xor.b2.print()
    