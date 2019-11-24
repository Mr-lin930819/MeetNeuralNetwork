from exa.neural_network import NeuralNetwork
import exa.mnist_exam
import numpy

def main():
    input_nodes = 784
    hidden_nodes = 100
    output_nodes = 10

    network = NeuralNetwork(input_nodes, hidden_nodes, output_nodes, 0.3)
    # 获取训练数据集
    train_records = exa.mnist_exam.read_train_records()
    # 训练网络
    for record in train_records:
        inputs, targets = exa.mnist_exam.read_train_record_data(record, output_nodes)
        network.train(inputs, targets)
    # 测试网络
    test_records = exa.mnist_exam.read_test_records()
    for record in test_records:
        expect_number, query_inputs = exa.mnist_exam.read_test_record_data(record)
        _, _, final_outputs = network.query(query_inputs)
        query_result = numpy.argmax(final_outputs)
        print("正确数值：", expect_number)
        print("测试网络数值：", query_result)


if __name__ == '__main__':
    main()
