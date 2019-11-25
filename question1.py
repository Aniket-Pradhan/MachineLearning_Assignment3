import sys
import numpy as np

from ML_HW3 import datasets

class NN:
    def __init__(self, x, y, iter=1000, lr=0.1):
        self.input = x
        self.output = y
        self.lr = lr       # user defined learning rate
        self.iter = iter
        # 3 hidden layers with 100, 50, 50 neurons
        neurons_1 = 100       # neurons for hidden layers
        neurons_2 = 50       # neurons for hidden layers
        neurons_3 = 50       # neurons for hidden layers
        ip_dim = x.shape[0] # input layer size 64
        op_dim = y.shape[0] # output layer size 10

        self.w1 = np.random.randn(ip_dim, neurons_1) # weights
        self.b1 = np.zeros((1, neurons_1))           # biases
        self.w2 = np.random.randn(neurons_1, neurons_2)
        self.b2 = np.zeros((1, neurons_2))
        self.w3 = np.random.randn(neurons_2, neurons_3)
        self.b3 = np.zeros((1, neurons_3))
        self.w4 = np.random.randn(neurons_3, op_dim)
        self.b4 = np.zeros((1, op_dim))
    
    def train(self):
        for _ in range(iter):
            print(_)
            self.feedforward()
            self.backprop()

    def sigmoid(self, s):
        return 1/(1 + np.exp(-1 * s))

    def softmax(self, s):
        exps = np.exp(s - np.max(s, axis=1, keepdims=True))
        return exps/np.sum(exps, axis=1, keepdims=True)

    def feedforward(self):
        z1 = np.dot(self.input, self.w1) + self.b1
        self.a1 = self.sigmoid(z1)
        z2 = np.dot(self.a1, self.w2) + self.b2
        self.a2 = self.sigmoid(z2)
        z3 = np.dot(self.a2, self.w3) + self.b3
        self.a3 = self.sigmoid(z3)
        z4 = np.dot(self.a3, self.w4) + self.b4
        self.a4 = self.softmax(z4)
    
    def sigmoid_derv(self, s):
        return s * (1 - s)

    def cross_entropy(self, pred, real):
        n_samples = real.shape[0]
        res = pred - real
        return res/n_samples

    def error(self, pred, real):
        n_samples = real.shape[0]
        logp = - np.log(pred[np.arange(n_samples), real.argmax(axis=1)])
        loss = np.sum(logp)/n_samples
        return loss

    def backprop(self):
        loss = self.error(self.a4, self.output)
        print('Error :', loss)
        a4_delta = self.cross_entropy(self.a4, self.output) # w4
        z3_delta = np.dot(a4_delta, self.w4.T)
        a3_delta = z3_delta * self.sigmoid_derv(self.a3) # w3
        z2_delta = np.dot(a3_delta, self.w3.T)
        a2_delta = z2_delta * self.sigmoid_derv(self.a2) # w2
        z1_delta = np.dot(a2_delta, self.w2.T)
        a1_delta = z1_delta * self.sigmoid_derv(self.a1) # w1

        self.w4 -= self.lr * np.dot(self.a3.T, a4_delta)
        self.b4 -= self.lr * np.sum(a4_delta, axis=0, keepdims=True)
        self.w3 -= self.lr * np.dot(self.a2.T, a3_delta)
        self.b3 -= self.lr * np.sum(a3_delta, axis=0)
        self.w2 -= self.lr * np.dot(self.a1.T, a2_delta)
        self.b2 -= self.lr * np.sum(a2_delta, axis=0)
        self.w1 -= self.lr * np.dot(self.input.T, a1_delta)
        self.b1 -= self.lr * np.sum(a1_delta, axis=0)

train_dataset = datasets.dataset(sys.argv[1])
data = train_dataset.loadMNIST()
# print(data['X'][0].shape)
x_train = data["X"]
y_train = data["Y"]

nn = NN(x_train.ravel(), y_train)
# nn.train()

