import numpy


# 神经网络
class NeuralNetwork:
    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
        self.i_nodes = input_nodes
        self.h_nodes = hidden_nodes
        self.o_nodes = output_nodes
        self.lr = learning_rate

    def train(self):
        pass

    def query(self):
        w = numpy.random.rand(3, 3)
        print(w)
