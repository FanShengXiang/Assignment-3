import numpy as np

class Relu():
    def forward(self, x):
        self.x = x
        return np.maximum(self.x, 0)

    def backward(self, eta):
        eta[self.x <= 0] = 0
        return eta


class sigmoid():
    def forward(self, x):
        self.out = 1 / (1 + np.exp(-x))
        return self.out

    def backward(self, eta):
        result = eta * (self.out * (1-self.out))
        return result

