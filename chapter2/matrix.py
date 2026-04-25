# 自己实现一个矩阵加法，点乘的库
from __future__ import annotations
class Matrix:
    def __init__(self,rows :int,columns :int,name :str):
        self.name :str = name
        self.rows :int = rows
        self.columns :int = columns
        self.matrix:list[list[float]] = [[0 for j in range(self.columns)] for i in range(self.rows)]
    
    def SetAt(self,row :int,column :int,value :float) -> bool:
        if row < 0 or row > self.rows - 1:
            return False
        elif column < 0 or column > self.columns -1:
            return False
        else:
            self.matrix[row][column] = value
            return True
    def GetAt(self,row :int,column :int) -> float:
        if row < 0 or row > self.rows - 1:
            return None
        elif column < 0 or column > self.columns -1:
            return None
        else:
            return self.matrix[row][column] 
    
    def Fulfill(self,value :int) -> None:
        for i in range(self.rows):
            for j in range(self.columns):
                self.SetAt(i,j,value)     
    def print(self) -> None:
        print(f"{self.name}:\n[")
        for i in range(self.rows):
            for j in range(self.columns):
                print(self.matrix[i][j],end = " ")
            print("")
        print("]")
    def getSubMatrixByRow(self,row) -> Matrix:
        if row < 0 or row > self.rows - 1:
            return None
        result :Matrix = Matrix(1,self.columns,f"{self.name}-{row}-row")
        for i in range(result.columns):
            result.SetAt(0,i,self.GetAt(row,i))
        return result
    
    def getSubMatrix(self,rowStart,columnStart,rowEnd,columnEnd) -> Matrix:
        if rowStart < 0 or rowStart > self.rows - 1:
            return None
        elif columnStart < 0 or columnStart > self.columns -1:
            return None
        elif rowEnd < 0 or rowEnd > self.rows - 1:
            return None
        elif columnEnd < 0 or columnEnd > self.columns -1:
            return None
        else:
            result :Matrix = Matrix(rowEnd - rowStart + 1,columnEnd - columnStart + 1,f"{self.name}-SubMatrix")
            for i in range(result.rows):
                for j in range(result.columns):
                    result.SetAt(i,j,self.GetAt(rowStart+i,columnStart+j))
        return result

    def __add__(self,other :Matrix) -> Matrix:
        if self.rows != other.rows or self.columns != other.columns:
            return None
        result :Matrix = Matrix(self.rows,self.columns,f"{self.name}+{other.name}")
        for i in range(result.rows):
            for j in range(result.columns):
                result.matrix[i][j] = self.matrix[i][j] + other.matrix[i][j]
        return result
    
    def __sub__(self,other :Matrix) -> Matrix:
        if self.rows != other.rows or self.columns != other.columns:
            return None
        result :Matrix = Matrix(self.rows,self.columns,f"{self.name}+{other.name}")
        for i in range(result.rows):
            for j in range(result.columns):
                result.matrix[i][j] = self.matrix[i][j] - other.matrix[i][j]
        return result
    
    def __mul__(self, other :float | Matrix) -> float | Matrix:
        # 点乘
        if isinstance(other, Matrix):
            if self.columns != other.rows:
                return None
            result :Matrix = Matrix(self.rows, other.columns, f"{self.name} * {other.name}")
            for i in range(result.rows):
                for j in range(result.columns):
                    for k in range(self.columns):
                        result.matrix[i][j] += self.matrix[i][k] * other.matrix[k][j]
            return result
        
        # 数乘
        if isinstance(other, (int, float)):
            result :Matrix = Matrix(self.rows,self.columns,f"{self.name} * {other}")
            for i in range(result.rows):
                for j in range(result.columns):
                    result.matrix[i][j] = self.matrix[i][j] * other
            return result
        


if __name__ == "__main__":
    a :Matrix = Matrix(4,4,"a")
    b :Matrix = Matrix(4,4,"b")
    a.print()
    b.Fulfill(1)
    (a+b).print()
    (a*b).print()