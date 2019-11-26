import numpy as np
import progressbar

class NN:
    def __init__(self, x, y, lr=0.1, num_layers = 3, num_neurons = [100, 50, 50], epochs=1):
        assert num_layers >= 1
        assert num_layers == len(num_neurons)

        self.input = x
        self.output = y
        self.lr = lr       # user defined learning rate
        self.epochs = epochs
        # 3 hidden layers with 100, 50, 50 neurons
        neurons_1 = 100       # neurons for hidden layers
        neurons_2 = 50       # neurons for hidden layers
        neurons_3 = 50       # neurons for hidden layers
        ip_dim = self.input.shape[1]*self.input.shape[2] # input layer size 64
        op_dim = 2 # output layer size 10

        self.W = [np.random.randn(ip_dim, num_neurons[0])]
        self.b = [np.zeros((1, num_neurons[0]))]
        for layer_ind in range(num_layers-1):
            self.W.append(np.random.randn(num_neurons[layer_ind], num_neurons[layer_ind+1]))
            self.b.append(np.zeros((1, num_neurons[layer_ind+1])))
        self.W.append(np.random.randn(num_neurons[-1], op_dim))
        self.b.append(np.zeros((1, op_dim)))

        self.a = [0] * len(self.W)
        self.w1 = np.random.randn(ip_dim, neurons_1) # weights
        self.b1 = np.zeros((1, neurons_1))           # biases
        self.w2 = np.random.randn(neurons_1, neurons_2)
        self.b2 = np.zeros((1, neurons_2))
        self.w3 = np.random.randn(neurons_2, neurons_3)
        self.b3 = np.zeros((1, neurons_3))
        self.w4 = np.random.randn(neurons_3, op_dim)
        self.b4 = np.zeros((1, op_dim))
    
    def one_hot_encoded(self, x):
        # only 9 and 7... therefore encode only two bits
        if x == 7:
            return np.array([[0, 1]])
        else:
            return np.array([[1, 0]])
        pass
    
    def train(self):
        for epoch in range(self.epochs):
            print("epoch:", epoch+1)
            for x, y in progressbar.progressbar(zip(self.input, self.output)):
                self.x = np.array([x.ravel()])
                self.y = self.one_hot_encoded(y)
                self.feedforward()
                self.backprop()

    def sigmoid(self, s):
        s = np.round(s, decimals=5)
        exp = 1/(1 + np.exp(-1 * s))
        return exp

    def softmax(self, s):
        exps = np.exp(s - np.max(s, axis=1, keepdims=True))
        return exps/np.sum(exps, axis=1, keepdims=True)

    def feedforward(self):
        for layer_ind, (w, b) in enumerate(zip(self.W, self.b)):
            # print(self.x.shape, w.shape)
            if layer_ind == 0:
                z = np.dot(self.x, w) + b
                self.a[layer_ind] = self.sigmoid(z)
            else:
                z = np.dot(self.a[layer_ind-1], w)
                self.a[layer_ind] = self.sigmoid(z)

        # z1 = np.dot(self.x, self.w1) + self.b1
        # self.a1 = self.sigmoid(z1)
        # z2 = np.dot(self.a1, self.w2) + self.b2
        # self.a2 = self.sigmoid(z2)
        # z3 = np.dot(self.a2, self.w3) + self.b3
        # self.a3 = self.sigmoid(z3)
        # z4 = np.dot(self.a3, self.w4) + self.b4
        # self.a4 = self.softmax(z4)
    
    def sigmoid_derv(self, s):
        return s * (1 - s)

    def cross_entropy(self, pred, real):
        n_samples = real.shape[0]
        res = pred - real
        return res/n_samples

    def error(self, pred, real):
        n_samples = real.shape[0]
        # real = np.array([0, 1])
        arg_max = real.argmax(axis=0)
        arr = np.arange(n_samples)
        logp = - np.log(pred[arr, arg_max])
        loss = np.sum(logp)/n_samples
        return loss

    def backprop(self):
        loss = self.error(self.a[-1], self.y)
        # print('Error :', loss)
        delta_a = [0] * len(self.a)
        delta_z = [0] * (len(self.a)-1)
        
        delta_a[0] = self.cross_entropy(self.a[-1], self.y)
        delta_z[0] = np.dot(delta_a[0], self.W[-1].T)

        for layer_ind in range(1, len(self.a)):
            delta_a[layer_ind] = delta_z[layer_ind-1] * self.sigmoid_derv(self.a[-1 - layer_ind])
            if layer_ind < (len(self.a)-1):
                delta_z[layer_ind] = np.dot(delta_a[layer_ind], self.W[-1-layer_ind].T)

        # a4_delta = self.cross_entropy(self.a4, self.y) # w4
        # z3_delta = np.dot(a4_delta, self.w4.T)
        # a3_delta = z3_delta * self.sigmoid_derv(self.a3) # w3
        # z2_delta = np.dot(a3_delta, self.w3.T)
        # a2_delta = z2_delta * self.sigmoid_derv(self.a2) # w2
        # z1_delta = np.dot(a2_delta, self.w2.T)
        # a1_delta = z1_delta * self.sigmoid_derv(self.a1) # w1

        self.W[-1] -= self.lr * np.dot(self.a[-2].T, delta_a[0])
        self.b[-1] -= self.lr * np.sum(delta_a[0], axis=0, keepdims=True)
        for layer_ind in range(len(self.W), 1, -1):
            layer_ind -= 2
            if layer_ind > 0:
                # print(layer_ind)
                # print(self.W[layer_ind+1].shape, self.W[layer_ind].shape, self.a[layer_ind-1].T.shape, delta_a[-layer_ind-1].shape, "LOL")
                self.W[layer_ind] -= self.lr * np.dot(self.a[layer_ind-1].T, delta_a[-layer_ind-1])
                self.b[layer_ind] -= self.lr * np.sum(delta_a[-layer_ind-1])
            else:
                self.W[layer_ind] -= self.lr * np.dot(self.x.T, delta_a[-layer_ind-1])
                self.b[layer_ind] -= self.lr * np.sum(delta_a[-layer_ind-1])

        # self.w4 -= self.lr * np.dot(self.a3.T, a4_delta)
        # self.b4 -= self.lr * np.sum(a4_delta, axis=0, keepdims=True)
        # self.w3 -= self.lr * np.dot(self.a2.T, a3_delta)
        # self.b3 -= self.lr * np.sum(a3_delta, axis=0)
        # self.w2 -= self.lr * np.dot(self.a1.T, a2_delta)
        # self.b2 -= self.lr * np.sum(a2_delta, axis=0)
        # self.w1 -= self.lr * np.dot(self.x.T, a1_delta)
        # self.b1 -= self.lr * np.sum(a1_delta, axis=0)
    
    def predict(self, data):
        self.x = np.array([data.ravel()])
        self.feedforward()
        if self.a[-1].argmax() == 0:
            return 9
        else:
            return 7
