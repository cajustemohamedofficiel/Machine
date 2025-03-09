import numpy as np

class Matrix:
    def __init__(self, data):
        self.data = np.array(data)

    def __add__(self, other):
        """Element-wise addition"""
        return Matrix(self.data + other.data)

    def __sub__(self, other):
        """Element-wise subtraction"""
        return Matrix(self.data - other.data)

    def __mul__(self, other):
        """Matrix multiplication (dot product)"""
        if isinstance(other, Matrix):
            return Matrix(np.dot(self.data, other.data))
        return Matrix(self.data * other)
    
    def transpose(self):
        """Transpose the tensor"""
        return Matrix(self.data.T)

    def reshape(self, shape):
        """Reshape the tensor"""
        return Matrix(self.data.reshape(shape))

    def __repr__(self):
        return str(self.data)
