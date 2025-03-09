import numpy as np

def relu(x):
    """ReLU activation function"""
    return np.maximum(0, x)

def sigmoid(x):
    """Sigmoid activation function"""
    return 1 / (1 + np.exp(-x))

def tanh(x):
    """Tanh activation function"""
    return np.tanh(x)

def leaky_relu(x, alpha=0.01):
    return np.where(x > 0, x, alpha * x)

# ELU Activation Function
def elu(x, alpha=1.0):
    return np.where(x > 0, x, alpha * (np.exp(x) - 1))

# Softmax Activation Function
def softmax(x):
    e_x = np.exp(x - np.max(x))  # Stability trick to prevent overflow
    return e_x / np.sum(e_x, axis=0, keepdims=True)

# Swish Activation Function
def swish(x):
    return x * sigmoid(x)

# GELU Activation Function
def gelu(x):
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))