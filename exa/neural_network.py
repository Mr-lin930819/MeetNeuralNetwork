import numpy
import scipy.special


# 神经网络
class NeuralNetwork:
    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
        self.i_nodes = input_nodes
        self.h_nodes = hidden_nodes
        self.o_nodes = output_nodes
        self.lr = learning_rate
        # 输入层与隐藏层初始权重矩阵
        self.wih = numpy.random.rand(self.h_nodes, self.i_nodes) - 0.5
        # 隐藏层与输出层初始权重矩阵
        self.who = numpy.random.rand(self.o_nodes, self.h_nodes) - 0.5
        self.activation_func = lambda x: scipy.special.expit(x)

    def train(self):
        # 更新节点j与下一层节点k间的链接权重公式为：△W(j,k) = α * E(k) * sigmoid(O(k)) * (1 - sigmoid(O(k))) · O(j).T
        # E(k)为下一节点误差：E(k) = target - sigmoid(O(k))
        # E(j)为更新节点j的误差，可以由E(k)及所连接权重分隔误差，计算出：E(j) = W(j,k).T · E(k)
        pass

    def query(self, inputs_list):
        # 转置输入2维矩阵，转换为numpy的矩阵数组
        inputs = numpy.array(inputs_list, ndmin=2).T
        # 权重点乘输入，得到隐藏层输入
        hidden_inputs = numpy.dot(self.wih, inputs)
        # 运用激活函数，或得隐藏层输出
        hidden_output = self.activation_func(hidden_inputs)

        # 隐藏层到最终输出层同理
        final_inputs = numpy.dot(self.who, hidden_output)
        final_outputs = self.activation_func(final_inputs)
        return final_outputs
