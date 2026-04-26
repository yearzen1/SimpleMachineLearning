# 实现一个神经网络库
# 本质是可以由前面的矩阵运算库组成
from matrix import Matrix
from typing import Callable
import random
import math
class NerualNetwork:
    # 输入为 [1,2,3] 代表神经网络的第一层1个神经元，第二层2个神经元，第三层3个神经元
    def __init__(self,arch :list[int],name :str,activateFunc :Callable[[float],float]):
        layers :int = len(arch)
        self.arch :list[int] = arch
        self.a :list[Matrix] = [None for i in range(layers)]
        self.w :list[Matrix] = [None for i in range(layers)]
        self.b :list[Matrix] = [None for i in range(layers)]
        self.name :str = name
        self.activateFunc :Callable[[float],float] = activateFunc
        for i in range(1,layers):
            self.w[i] = Matrix(arch[i-1],arch[i],f"w[{i}]")
            self.b[i] = Matrix(1,arch[i],f"b[{i}]")
                
    def assignRandom(self,value :float) -> None:
        for i in range(1,len(self.w)):
                self.w[i].Fulfill(value)
                self.b[i].Fulfill(value)

    def print(self) -> None:
        print(f"{self.name}:")
        print("{")
        for i in range(len(self.w)):
            if i != 0 :
                self.w[i].print()
                self.b[i].print()
        print("}\n")
    
    def forward(self,trainInputItem:Matrix) -> Matrix:
        self.a[0] = trainInputItem
        n = len(self.a)
        for i in range(1,n):
            self.a[i] = self.a[i-1] * self.w[i] + self.b[i]
            self.a[i].everySetByFunc(self.activateFunc)
        return self.a[n-1]
    
    def loss(self,trainInput:Matrix,trainOutput:Matrix) -> float:
        sum :Matrix = Matrix(1,trainOutput.columns,"temp")
        result :float = 0
        temp :Matrix = None
        for i in range(trainInput.rows):
            y :Matrix = self.forward(trainInput.getSubMatrixByRow(i))
            temp = y - trainOutput.getSubMatrixByRow(i)
            temp.everySetByFunc(lambda x:x**2)
            sum += temp
        for i in range(sum.columns):
            result += sum.GetAt(0,i)
        return result/trainInput.rows
    
    @staticmethod
    def finiteDiff(lossAddeps :float,loss :float,eps :float) -> float:
        return (lossAddeps - loss) / eps
    
    def train(self,eps :float,rate :float,trainInput:Matrix,trainOutput:Matrix) -> float:
        constloss :float = self.loss(trainInput,trainOutput)
        gw :list[Matrix] = [None for _ in range(len(self.w))]
        gb :list[Matrix] = [None for _ in range(len(self.b))]
        for i in range(1,len(self.w)):
            gw[i] = Matrix(self.w[i].rows,self.w[i].columns,"gw{i}")
            for j in range(self.w[i].rows):
                for k in range(self.w[i].columns):
                    saved = self.w[i].GetAt(j,k)
                    self.w[i].SetAt(j,k,saved + eps)
                    gw[i].SetAt(j,k,self.finiteDiff(self.loss(trainInput,trainOutput),constloss,eps))
                    self.w[i].SetAt(j,k,saved)
        for i in range(1,len(self.b)):
            gb[i] = Matrix(self.b[i].rows,self.b[i].columns,"gb{i}")
            for j in range(self.b[i].rows):
                for k in range(self.b[i].columns):
                    saved = self.b[i].GetAt(j,k)
                    self.b[i].SetAt(j,k,saved + eps)
                    gb[i].SetAt(j,k,self.finiteDiff(self.loss(trainInput,trainOutput),constloss,eps))
                    self.b[i].SetAt(j,k,saved)
        for i in range(1,len(self.w)):
            self.w[i] -=  gw[i] * rate 
            self.b[i] -=  gb[i] * rate
        
        return self.loss(trainInput,trainOutput)

            
if __name__ == "__main__" :
    train: list[list[int]] = [
        [0,0,0],
        [0,1,1],
        [1,0,1],
        [1,1,0]
    ]
    train1 :Matrix = Matrix(4,3,"train1")
    train1.matrix = train
    traininput :Matrix = train1.getSubMatrix(0,0,3,1)
    trainoutput :Matrix = train1.getSubMatrix(0,2,3,2)
    def sigmoid(x:float) -> float:
        return 1 / (1 + math.exp(-x))
    nn :NerualNetwork = NerualNetwork([2,2,1],"test",sigmoid)
    # nn.print()
    nn.assignRandom(random.uniform(1,1))
    # nn.forward(traininput.getSubMatrixByRow(0)).print()
    # print(nn.loss(traininput,trainoutput))
    for _ in range(1000):
        if _ == 0 :
            print(f"batch: {_} loss: {nn.loss(traininput,trainoutput):.4f}")
        else:
            print(f"batch: {_} loss: {nn.train(1e-4,1e-1,traininput,trainoutput):.4f}")
    