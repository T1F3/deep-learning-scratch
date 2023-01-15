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
input_layer = Layer(2, activations=inputs[3])
output_layer = Layer(1)
xor_network = (
    xor_network
    .add_layer(input_layer)
    .add_layer(Layer(2))
    .add_layer(output_layer)
)

xor_network.forward_pass()
