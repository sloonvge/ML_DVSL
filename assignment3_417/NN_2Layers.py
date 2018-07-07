import numpy as np
class NN(object):
    
    def __init__(self, input_dim, L1, L2, num_labels):
        self.input_dim = input_dim
        self.num_labels = num_labels
        self.L1 = L1
        self.L2 = L2
        self.train_loss = []

    def train(self, xs, ys, learning_rate=1e-2, reg=0.0):
        self.N = xs.shape[0]
        #fc1 layer
        a_fc1 = np.sum((np.dot(xs, self.W_fc1), self.b_fc1))
        h_fc1 = sigmoid(a_fc1)
        #h_fc1 = relu(a_fc1)
        #fc2 layer
        a_fc2 = np.sum((np.dot(h_fc1, self.W_fc2), self.b_fc2))
        h_fc2 = sigmoid(a_fc2)
        #h_fc2 = relu(a_fc2)
        #output layer
        output_ys = np.sum((np.dot(h_fc2, self.W_ys), self.b_ys))
        prediction = softmax(output_ys)
        #loss
        self.loss = np.mean(np.sum(- ys * np.log(prediction), 1, keepdims=True)) + \
        0.5 * reg / self.N * (np.sum(self.W_ys ** 2) + np.sum(self.W_fc2 ** 2) + np.sum(self.W_fc1 ** 2))
        self.train_loss.append(self.loss)
        #accuracy
        self.acc = np.mean(np.equal(np.argmax(prediction, 1), np.argmax(ys, 1)))

        W_ys_error = output_ys - ys
        W_ys_grad = h_fc2.T.dot(W_ys_error)
        W_fc2_error = W_ys_error.dot(self.W_ys.T) * (h_fc2 * (1 - h_fc2))
        #W_fc2_error = W_ys_error.dot(self.W_ys.T) * drelu(h_fc2)
        W_fc2_grad = h_fc1.T.dot(W_fc2_error)
        W_fc1_error = W_fc2_error.dot(self.W_fc2.T) * (h_fc1 * (1- h_fc1))
        #W_fc1_error = W_fc2_error.dot(self.W_fc2.T) * drelu(h_fc1)
        W_fc1_grad = xs.T.dot(W_fc1_error)
        W_ys_temp = self.W_ys
        W_fc2_temp = self.W_fc2
        W_fc1_temp = self.W_fc1
        self.W_ys -= learning_rate * (W_ys_grad / self.N + reg * W_ys_temp / self.N)
        self.W_fc2 -= learning_rate * (W_fc2_grad / self.N + reg * W_fc2_temp / self.N)
        self.W_fc1 -= learning_rate * (W_fc1_grad / self.N + reg * W_fc1_temp / self.N)
        return self.train_loss
    
    def test(self, xs, ys):
        a_fc1 = np.sum((np.dot(xs, self.W_fc1), self.b_fc1))
        h_fc1 = sigmoid(a_fc1)
        #fc2 layer

        a_fc2 = np.sum((np.dot(h_fc1, self.W_fc2), self.b_fc2))
        h_fc2 = sigmoid(a_fc2)
        #output layer

        output_ys = np.sum((np.dot(h_fc2, self.W_ys), self.b_ys))
        prediction = softmax(output_ys)
        loss = np.mean(np.sum(- ys * np.log(prediction), 1, keepdims=True))
        acc = np.mean(np.equal(np.argmax(prediction, 1), np.argmax(ys, 1)))
        return acc, loss
    
    def _check_grad(self, xs, ys, h=1e-6):
        W = np.r_[self.W_fc1.flatten(), self.W_fc2.flatten(), self.W_ys.flatten()]
        w = W.shape[0]
        dW = np.zeros(w)
        for i in range(w):
            loss1 = self.test(xs, ys)[1]
            print(loss1)
            W[i] += h
            loss2 = self.test(xs, ys)[1]
            print(loss2)
            dW[i] = (loss2 - loss1) /  h
        return dW

def softmax(s):
    output = np.exp(s) / np.reshape(np.sum(np.exp(s), axis=1), (-1, 1))
    return output
def sigmoid(x):
    x = np.exp(x) / (1 + np.exp(x))
    return x
def relu(x):
    x[x < 0] = 0
    return x
def drelu(x):
    x[x > 0] = 1
    return x