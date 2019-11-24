import numpy
import matplotlib.pyplot


def read_train_records():
    # 获取文件的训练集，数据格式为每行的第一项为数字数值，之后的28x28=784项为手写数字的像素数据
    data_file = open("exa/mnist_dataset/mnist_train_100.csv")
    data_list = data_file.readlines()
    data_file.close()
    return data_list


def read_test_records():
    test_data_file = open("exa/mnist_dataset/mnist_test_10.csv")
    test_data_list = test_data_file.readlines()
    test_data_file.close()
    return test_data_list


def read_train_record_data(record, output_nodes):
    all_values = record.split(',')
    # 将训练数据转换为神经网络合适的值区间：0.01~1.00
    scaled_train_inputs = scale_to_network_values(all_values)
    # 转换目标期望输出：全部初始化为最小值，all_values[0]表示的是目标数字数值，所以期望输出为最大值0.99
    targets = numpy.zeros(output_nodes) + 0.01
    targets[int(all_values[0])] = 0.99
    return scaled_train_inputs, targets


def read_test_record_data(record):
    all_values = record.split(',')
    return all_values[0], scale_to_network_values(all_values)


def scale_to_network_values(raw_values):
    return numpy.asfarray(raw_values[1:]) / 255.0 * 0.99 + 0.01


def draw_data(data):
    all_values = data.split(',')
    image_array = numpy.asfarray(all_values[1:]).reshape((28, 28))
    matplotlib.pyplot.imshow(image_array, cmap='Greys', interpolation='None')
    matplotlib.pyplot.show()
