# 自己实现一个矩阵加法，点乘的库
from __future__ import annotations
class Matrix:
    def __init__(self,rows :int,columns :int,name :str):
        self.name :str = name
        self.rows :int = rows
        self.columns :int = columns
        self.matrix:list[list[float]] = [[0 for j in range(self.columns)] for i in range(self.rows)]
    
    def AssignAt(self,row :int,column :int,value :float) -> bool:
        if row < 0 or row > self.rows - 1:
            return False
        elif column < 0 or column > self.columns -1:
            return False
        else:
            self.matrix[row][column] = value
            return True
    def Fulfill(self,value :int) -> None:
        for i in range(self.rows):
            for j in range(self.columns):
                self.AssignAt(i,j,value)     
    def print(self) -> None:
        print(f"{self.name}:\n[")
        for i in range(self.rows):
            for j in range(self.columns):
                print(self.matrix[i][j],end = " ")
            print("")
        print("]")
    
    def __add__(self,other :Matrix) -> Matrix:
        if self.rows != other.rows or self.columns != other.columns:
            return None
        result :Matrix = Matrix(self.rows,self.columns,f"{self.name}+{other.name}")
        for i in range(result.rows):
            for j in range(result.columns):
                result.matrix[i][j] = self.matrix[i][j] + other.matrix[i][j]
        return result
    def __mul__(self,other :Matrix) -> Matrix:
        if self.columns != other.rows:
            return None
        n :int = self.columns
        result :Matrix = Matrix(self.rows,self.columns,f"{self.name} * {other.name}")
        for i in range(result.rows):
            for j in range(result.columns):
                for k in range(n):
                    result.matrix[i][j] += self.matrix[i][k] * other.matrix[k][j] 
        return result

if __name__ == "__main__":
    a :Matrix = Matrix(4,4,"a")
    b :Matrix = Matrix(4,4,"b")
    a.print()
    b.Fulfill(1)
    (a+b).print()
    (a*b).print()