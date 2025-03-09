import unittest
import numpy as np
from core.Matrix import Matrix  # Assuming your Matrix class is in a file named your_module.py

class TestMatrixOperations(unittest.TestCase):

    def setUp(self):
        # Initialize test matrices
        self.matrix_a = Matrix([[1, 2], [3, 4]])
        self.matrix_b = Matrix([[5, 6], [7, 8]])

    def test_add(self):
        result = self.matrix_a + self.matrix_b
        expected = [[6, 8], [10, 12]]
        self.assertEqual(result.data.tolist(), expected)

    def test_sub(self):
        result = self.matrix_a - self.matrix_b
        expected = [[-4, -4], [-4, -4]]
        self.assertEqual(result.data.tolist(), expected)

    def test_scalar_multiplication(self):
        result = self.matrix_a * 3  # Scalar multiplication
        expected = [[3, 6], [9, 12]]
        self.assertEqual(result.data.tolist(), expected)

    def test_matrix_multiplication(self):
        result = self.matrix_a * self.matrix_b  # Matrix multiplication (dot product)
        expected = [[19, 22], [43, 50]]
        self.assertEqual(result.data.tolist(), expected)

    def test_transpose(self):
        result = self.matrix_a.transpose()
        expected = [[1, 3], [2, 4]]
        self.assertEqual(result.data.tolist(), expected)

    def test_reshape(self):
        result = self.matrix_a.reshape((4,))
        expected = [1, 2, 3, 4]
        self.assertEqual(result.data.tolist(), expected)

if __name__ == '__main__':
    unittest.main()
