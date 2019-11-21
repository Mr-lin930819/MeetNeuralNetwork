from exa.neural_network import NeuralNetwork


def main():
    input_nodes = 3
    hidden_nodes = 3
    output_nodes = 3

    network = NeuralNetwork(input_nodes, hidden_nodes, output_nodes, 0.3)
    print(network.query([1.0, 0.5, -1.5]))
    for i in range(0, 30):
        network.train([1.0, 0.5, -1.5], [0.7, 0.8, 0.9])
    print(network.query([1.0, 0.5, -1.5]))


if __name__ == '__main__':
    main()
