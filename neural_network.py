import numpy as np

def sigmoid(x):
    return 1/(1+np.exp(-x))


#1. define the structure
#2. given training datas and training labels, train the neural_network.
#3. given test datas, predict the output
class neural_network:

    def __init__(self):
        self.structure = [] #store the structure of the nn
        self.weights = [] #store the weights of each layer. Note: weights of each layer is a matrix.
        self.biases = [] # store the bias of each layer
        self.outputs = [] # store each layer's output, the first layer is the input.
        self.gradients_w = [] # store the gradient of weights of each layer. it should has the same dimension as self.weights
        self.gradients_b = [] # store the gradient of bias of each layer. it should has the smae dimension as self.bias
        self.alpha = 0.1 # the learning rate

    #using a list structure to define the structure of the network
    #e.g [2,3,2] means the input layer has two nodes, the hidden layer has three nodes and the output layer has two nodes.
    def def_structure(self, structure):
        self.structure = structure
        self.outputs.append(np.zeros([structure[0],1]))
        for i in xrange(0 , len(structure) - 1):
            self.weights.append(np.random.randn(structure[i + 1], structure[i]))
            self.gradients_w.append(np.zeros([structure[i + 1], structure[i]]))
            self.biases.append(np.random.randn(structure[i + 1],1))
            self.gradients_b.append(np.zeros([structure[i + 1],1]))
            self.outputs.append(np.zeros([structure[i + 1],1]))
        self.weights = np.array(self.weights)
        self.biases = np.array(self.biases)
        self.gradients_w = np.array(self.gradients_w)
        self.gradients_b = np.array(self.gradients_b)
        self.outputs = np.array(self.outputs)
    
    
    # compute the forward result of the nn and record output of each layer
    def forward(self, data):
        data = data.reshape(-1, 1)
        sum_product = data
        self.outputs[0] = data
        index = 1
        for (w , b) in zip(self.weights, self.biases):
            sum_product = np.dot(w, sum_product) + b
            sum_product = sigmoid(sum_product)
            self.outputs[index] = sum_product
            index += 1
        return sum_product
    
    # compute gradient of weights and biases of each layer
    def backpropogate(self, labels):
        #output layer error:
        labels = labels.reshape(-1,1)
        sum_grad = self.outputs[-1] - labels 
        grad = np.dot(sum_grad, self.outputs[-2].T)
        self.gradients_w[-1] += grad
        self.gradients_b[-1] += sum_grad

        #layer error of the rest layers.
        layer_num = len(self.structure)
        for i in xrange(layer_num - 2, 0, -1):
            node_grad = np.dot(sum_grad.T, self.weights[i]).T
            sum_grad = node_grad * self.outputs[i] * (1 - self.outputs[i])
            grad = np.dot(sum_grad, self.outputs[i - 1].T)
            self.gradients_w[i - 1] += grad
            self.gradients_b[i - 1] += sum_grad

    #train network on mini_batch
    #first, compute the forward result and record
    #second, compute the gradient of each weights and biases
    #thrid, updata the network
    def train_mini_batch(self, batch_datas, batch_labels):
        self.gradients_w.fill(0)
        self.gradients_b.fill(0)
        for data, label in zip(batch_datas, batch_labels):
            self.forward(data)
            self.backpropogate(label)
        self.weights -= self.alpha * self.gradients_w / len(batch_datas)
        self.biases -= self.alpha * self.gradients_b / len(batch_datas)
        #print self.weights

    #train_data: numpy_array, n*m, n is the number of items, m is the number of features.
    #train_labels: numpy_array, n*1, n is the number of items.
    #first, shuffle and split the train_data and train_label
    #second, train network on each mini_batch
    def fit(self, train_datas, train_labels, epochs = 10, batch_size = 64, learning_rate = 0.1):
        self.alpha = learning_rate
        data_size = len(train_data)
        for epoch in xrange(0, epochs):
            index = np.arange(0, data_size)
            np.random.shuffle(index)
            start_data = 0
            print("epoch : {}\n".format(epoch))
            while (start_data < data_size):
                end_data = min(start_data + batch_size, data_size)
                batch_datas = train_datas[index[start_data: end_data]]
                batch_labels = train_labels[index[start_data: end_data]]
                self.train_mini_batch(batch_datas, batch_labels)
                start_data += batch_size
    
    def predict(self, data):
        return self.forward(data)
            

nn = neural_network()
nn.def_structure([2,30,30,30,30,2])
train_data = np.array([[-1,-1],[-0.1,-0.3],[0.2,0.4],[2,2]])
train_label = np.array([[1,0],[1,0],[0,1],[0,1]])
nn.fit(train_data,train_label,200,2,0.1)
print nn.predict(np.array([-1,-1]))
print nn.predict(np.array([2,2]))
a = 1
