import numpy as np
import random
import matplotlib.pyplot as plt

class Perception:
    # y = sign(wx + b)
    # x: [x0, x1]
    # y: [y0]
    def __init__(self) -> None:
        self.w = np.array([0.0, 0.0])
        self.b = np.array([0.0])
        
        pass
    
    def _sign(self, x):
        return 1 if x >= 0 else -1
        
    
    def train(self, x_nparray, y_nparray, lr):
        epochs = 100000
        
        for epoch in range(epochs):            
            flag = False
            
            s = list(range(len(x_nparray)))
            random.shuffle(s)
            for i in s:
                res = y_nparray[i] * self._sign(np.dot(self.w, x_nparray[i]) + self.b)
                if res < 0:
                    if epoch % 1000 == 0:
                        print(f"epoch: {epoch}, i: {i}, err: {abs(res)}")
                    
                    self.w += lr * y_nparray[i] * x_nparray[i]
                    self.b += lr * y_nparray[i]
                    flag = True

                    break
                
            if flag == False:
                break
            
        print(f"w:{self.w}, b:{self.b}")
            
    def infer(self, x_nparray, y_nparray):
        # y = sign(w0x0+w1x1+b)
        # w0x0+w1x1+b=0
        x0 = np.linspace(0, 1, 100)
        x1 = (-self.w[0] * x0 - self.b) / self.w[1]
        
        plt.plot(x0, x1)
        
        y_pred = np.array([0 for i in range(len(x_nparray))])
        
        for i in range(len(x_nparray)):
            y_pred[i] = self._sign(np.dot(self.w, x_nparray[i]) + self.b)
            
            # plt.scatter(x_nparray[i][0], x_nparray[i][1], c="yellow" if y_pred[i] == 1 else "blue")
            plt.scatter(x_nparray[i][0], x_nparray[i][1], c="yellow" if y_nparray[i] == 1 else "blue")
    
                