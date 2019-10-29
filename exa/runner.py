from exa.neural_network import NeuralNetwork


def main():
    input_nodes = 3
    hidden_nodes = 3
    output_nodes = 3

    network = NeuralNetwork(input_nodes, hidden_nodes, output_nodes, 0.3)
    network.query()


if __name__ == '__main__':
    main()
