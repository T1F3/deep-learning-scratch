import random
import logging

logging.basicConfig(level=logging.DEBUG)

def relu_activation(x):
    """Only return either positive inputs or 0"""
    return max(0, x)

def identity_activation(x):
    """A pass-through activation for input layers"""
    return x

class Neuron:
    def __init__(
        self,
        activation_function,
        node_idx=None,
        bias=random.uniform(-2, 2),
        activation = None,
    ):
        self.bias = bias
        self.node_idx = node_idx
        self.activation_function = activation_function
        self.input_value = None
        self.activation = activation
    def __str__(self) -> str:
        return f"Neuron #{self.node_idx}"
    def set_activation(self):
        if self.input_value is not None:
            self.activation =  self.activation_function(self.input_value + self.bias)

class Layer:
    def __init__(self, num_neurons: int, activation_function=relu_activation, activations=None):
        self.num_neurons = num_neurons
        self.activations = activations
        self.layer_idx = None
        self.neurons = [
            Neuron(
                node_idx=node_idx,
                activation_function=activation_function,
                activation=self._get_default_neuron_activation(node_idx)
            )
            for node_idx in range(self.num_neurons)
        ]
    def _get_default_neuron_activation(self, idx):
        if self.activations is None:
            return None
        return self.activations[idx]

class DenseLayerWeights:
    """Created nested weight array for pair of layers"""
    def __init__(self, from_layer: Layer, to_layer: Layer):
        self.weights = [
            [random.uniform(-2, 2) for _ in range(from_layer.num_neurons)]
            for _ in range(to_layer.num_neurons)
        ]
        self.from_layer = from_layer
        self.to_layer = to_layer
    def get_weight(self, from_neuron_idx: int, to_neuron_idx: int):
        return self.weights[to_neuron_idx][from_neuron_idx]

class Network:
    def __init__(self):
        self.layers = []
        self.layer_weights = []
    def add_layer(self, layer: Layer):
        self.layers.append(layer)
        layer.layer_idx = len(self.layers) - 1
        # skip if network only has input layer
        if len(self.layers) != 1:
            previous_layer: Layer = self.layers[-2]
            layer_weights = DenseLayerWeights(previous_layer, layer)
            self.layer_weights.append(layer_weights)
        return self
    def get_weight(self, from_neuron_idx: int, to_neuron_idx: int, from_layer: Layer):
        layer_weights: DenseLayerWeights = self.layer_weights[from_layer.layer_idx]
        return layer_weights.get_weight(from_neuron_idx, to_neuron_idx)
    def layer_forward_pass(self, from_layer: Layer):
        to_layer: Layer = self.layers[from_layer.layer_idx + 1]
        logging.debug("To Layer: %d", to_layer.layer_idx)
        for to_neuron_idx in range(to_layer.num_neurons):
            to_neuron = to_layer.neurons[to_neuron_idx]
            to_neuron_input = 0
            logging.debug("\tTo-Neuron Idx: %d", to_neuron_idx)
            for from_neuron_idx in range(from_layer.num_neurons):
                from_neuron = from_layer.neurons[from_neuron_idx]
                weight = self.get_weight(from_neuron_idx, to_neuron_idx, from_layer)
                to_neuron_input += weight * from_neuron.activation
                logging.debug("\t\tFrom-Neuron Idx: %d", from_neuron_idx)
                logging.debug("\t\t\tFrom-Neuron Weight: %f", weight)
                logging.debug("\t\t\tFrom-Neuron Activation: %f", from_neuron.activation)
            to_neuron.input_value = to_neuron_input
            to_neuron.set_activation()
            logging.debug("\t\tTo-Neuron Input: %f", to_neuron.input_value)
            logging.debug("\t\tTo-Neuron Bias: %f", to_neuron.bias)
            logging.debug("\t\tTo-Neuron Activation: %f", to_neuron.activation)
    def forward_pass(self):
        for from_layer in self.layers[:-1]:
            self.layer_forward_pass(from_layer)
        final_layer: Layer = self.layers[-1]
        return [neuron.activation for neuron in final_layer.neurons]
