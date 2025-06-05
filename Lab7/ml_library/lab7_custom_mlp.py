import random 
import numpy as np

class Affine:
    def __init__(self, input_shape):
        self.W = np.random.randn(input_shape[0], input_shape[1])
        self.b = np.zeros(input_shape[1])
        self.x = None
        self.dL_dW = None
        self.dL_db = None
    def forward_propagation(self, x):
        self.x = np.array(x)
        return np.dot(x, self.W)+ self.b
    def back_propagation(self,dL_dz):
        dL_dx = np.dot(dL_dz, self.W.T)
        self.dL_dW = np.dot(self.x.T, dL_dz)
        self.dL_db = np.sum(dL_dz, axis=0)
        return dL_dx


class ReLU:
    def __init__(self):
        self.x = None

    def forward_propagation(self, x):
        self.x = x

        out = np.maximum(x, 0) 
        return out

    def back_propagation(self, dout):

        dx = dout.copy()
        dx[self.x <= 0] = 0 
        return dx
    
class Sigmoid:
    def __init__(self):
        self.output = None

    def forward_propagation(self, x):
        self.output = 1 / (1 + np.exp(-x))
        return self.output

    def back_propagation(self, dout):
        return dout * self.output * (1 - self.output)
    

class BinaryCrossEntropyLoss:
    def __init__(self):
        self.y_pred = None
        self.y_true = None

    def func(self, y_pred, y_true):
        self.y_pred = np.array(y_pred)
        self.y_true = np.array(y_true).reshape(-1, 1)
        epsilon = 1e-15
        self.y_pred = np.clip(self.y_pred, epsilon, 1 - epsilon)
        loss = -np.mean(self.y_true * np.log(self.y_pred) + (1 - self.y_true) * np.log(1 - self.y_pred))
        return loss

    def div_func(self):
        return (self.y_pred - self.y_true) / (self.y_pred * (1 - self.y_pred) * self.y_true.size)
    
class MSELoss:
    def __init__(self):
        self.y_pred = None
        self.y_true = None
    
    def func(self,y_pred, y_true):
        self.y_pred = np.array(y_pred) 
        
        self.y_true = np.array(y_true).reshape(-1, 1) 
        loss = np.mean((self.y_pred-self.y_true)**2)
        return loss

    def div_func(self):
        return 2*(self.y_pred - self.y_true)/self.y_true.size
    
class SGD:
    def __init__(self,learning_rate = 0.001):
        self.learning_rate = learning_rate

    def update_weight(self,layers):
        for layer in layers:
            if hasattr(layer, 'W'):
                layer.W -= self.learning_rate*layer.dL_dW
                layer.b -= self.learning_rate*layer.dL_db

class NeuralNetwork:
    def __init__(self, layers, loss_func, optimizer):
        self.layers = layers  
        self.loss_func = loss_func
        self.optimizer = optimizer
    def forward_propagation(self, x):
        for layer in self.layers:
            x = layer.forward_propagation(x)
        return x

    def back_propagation(self, dout):
        for layer in reversed(self.layers):
            dout = layer.back_propagation(dout)
        return dout
    
    def fit(self, x,y_true, epochs):
        for epoch in range(epochs):
            y = self.forward_propagation(x)

            loss = self.loss_func.func(y, y_true)
            print(f"эпоха {epoch} потеря {loss:.4f}")

            back = self.loss_func.div_func()

            self.back_propagation(back)

            self.optimizer.update_weight(self.layers)
    def predict(self, x):
        return self.forward_propagation(x)
    

