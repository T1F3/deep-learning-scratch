from naive_mlp import (
    Network,
    Layer
)

inputs = [
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1],
]
outputs = [0, 1, 1, 0]

xor_network = Network()
xor_network = (
    xor_network
    .add_layer(Layer(2))
    .add_layer(Layer(2))
    .add_layer(Layer(1))
)
xor_network.forward_pass(inputs[:])
