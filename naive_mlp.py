"""Naive element-wise MLP Implementation"""

import random
import logging
from typing import Union

logging.basicConfig(level=logging.DEBUG)
# pylint: disable=logging-fstring-interpolation

def relu_activation(x):
    """Only return either positive inputs or 0"""
    return max(0, x)

def l2_loss(output, target):
    error = 0
    for output_value, target_value in zip(output, target):
        error += ((output_value - target_value) ** 2) / 2
    return error

def relu_derivative(x):
    if x > 0:
        return 1
    return 0

def l2_loss_derivative(neuron_activation, target):
    return neuron_activation - target

class Bias:
    def __init__(self, value, error_gradient=None):
        self.value = value
        self.error_gradient = error_gradient

# Primitive Classes

class Neuron:
    def __init__(
        self,
        activation_function,
        activation_derivative_func,
        node_idx=None,
        bias_value=random.uniform(0, 0),
        activation = None,
    ):
        self.bias = Bias(bias_value)
        self.node_idx = node_idx
        self.activation_function = activation_function
        self.activation_derivative_func = activation_derivative_func
        self.input_value = None
        self.activation = activation
        self.error = None
    def __str__(self) -> str:
        return f"Neuron #{self.node_idx}"
    def set_activation(self):
        if self.input_value is not None:
            self.activation =  self.activation_function(self.input_value + self.bias.value)
    def set_neuron_bias_gradient(self):
        self.bias.error_gradient = self.error

class Layer:
    def __init__(
        self,
        num_neurons: int,
        activation_function=relu_activation,
        activation_derivative_func=relu_derivative
):
        self.num_neurons = num_neurons
        self.layer_idx = None
        self.neurons = [
            Neuron(
                node_idx=node_idx,
                activation_function=activation_function,
                activation_derivative_func=activation_derivative_func,
            )
            for node_idx in range(self.num_neurons)
        ]
    def set_neuron_activations(self, activations: list):
        for neuron, activation in zip(self.neurons, activations):
            neuron.activation = activation

class Weight:
    def __init__(self, value, error_gradient=None):
        self.value = value
        self.error_gradient = error_gradient

class DenseLayerWeights:
    """Created nested weight array for pair of layers"""
    def __init__(self, from_layer: Layer, to_layer: Layer):
        self.weights = [
            [Weight(random.uniform(-0.5, 0.5)) for _ in range(from_layer.num_neurons)]
            for _ in range(to_layer.num_neurons)
        ]
        self.from_layer = from_layer
        self.to_layer = to_layer
    def get_weight(self, from_neuron_idx: int, to_neuron_idx: int):
        return self.weights[to_neuron_idx][from_neuron_idx]

def update_item_value_w_gradient_descent_step(
    item: Union[Bias, Weight],
    learning_rate: float=1
):
    item.value -= (item.error_gradient * learning_rate)


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
    def _get_weight(self, from_neuron_idx: int, to_neuron_idx: int, from_layer: Layer):
        layer_weights: DenseLayerWeights = self.layer_weights[from_layer.layer_idx]
        return layer_weights.get_weight(from_neuron_idx, to_neuron_idx)
    def get_output_neurons(self):
        output_layer: Layer = self.layers[-1]
        return output_layer.neurons
    def set_output_neuron_error(self, target: list[float], loss_derivative_func=l2_loss_derivative):
        output_neurons = self.get_output_neurons()
        for output_neuron, target_value in zip(output_neurons, target):
            d_activation_d_input = output_neuron.activation_derivative_func(output_neuron.input_value)
            d_loss_d_activation = loss_derivative_func(output_neuron.activation, target_value)
            neuron_error = d_loss_d_activation * d_activation_d_input
            output_neuron.error = neuron_error
            output_neuron.set_neuron_bias_gradient()
            update_item_value_w_gradient_descent_step(output_neuron.bias)

    def learn_step_for_layer_from_neurons(self, layer_weights: DenseLayerWeights):
        logging.debug(f"Layer: from: {layer_weights.from_layer.layer_idx} to {layer_weights.to_layer.layer_idx}")
        if layer_weights.from_layer.layer_idx == 0:
            return
        for from_neuron in layer_weights.from_layer.neurons:
            to_layer_weighted_errors = 0
            for to_neuron in layer_weights.to_layer.neurons:
                weight = layer_weights.get_weight(from_neuron.node_idx, to_neuron.node_idx).value
                to_layer_weighted_errors += to_neuron.error * weight
            logging.debug(f"from neuron idx: {from_neuron.node_idx}")
            neuron_error = (
                to_layer_weighted_errors * from_neuron.activation_derivative_func(from_neuron.input_value)
            )
            from_neuron.error = neuron_error
            from_neuron.set_neuron_bias_gradient()
            update_item_value_w_gradient_descent_step(from_neuron.bias)

    def learn_step_for_layer_weight_gradients(self, layer_weights: DenseLayerWeights):
        for from_neuron in layer_weights.from_layer.neurons:
            for to_neuron in layer_weights.to_layer.neurons:
                d_error_d_weight = from_neuron.activation * to_neuron.error
                weight = layer_weights.get_weight(from_neuron.node_idx, to_neuron.node_idx)
                weight.error_gradient = d_error_d_weight
                update_item_value_w_gradient_descent_step(weight)
    def _layer_forward_pass(self, from_layer: Layer):
        to_layer: Layer = self.layers[from_layer.layer_idx + 1]
        logging.debug("\tTo Layer: %d", to_layer.layer_idx)
        for to_neuron_idx in range(to_layer.num_neurons):
            to_neuron = to_layer.neurons[to_neuron_idx]
            to_neuron_input = 0
            logging.debug("\t\tTo-Neuron Idx: %d", to_neuron_idx)
            for from_neuron_idx in range(from_layer.num_neurons):
                weight = self._get_weight(from_neuron_idx, to_neuron_idx, from_layer).value
                from_neuron = from_layer.neurons[from_neuron_idx]
                logging.debug("\t\t\tFrom-Neuron Idx: %d", from_neuron_idx)
                logging.debug("\t\t\t\tFrom-Neuron Weight: %f", weight)
                logging.debug("\t\t\t\tFrom-Neuron Activation: %f", from_neuron.activation)
                to_neuron_input += weight * from_neuron.activation
            to_neuron.input_value = to_neuron_input
            to_neuron.set_activation()
            logging.debug("\t\t\tTo-Neuron Input: %f", to_neuron.input_value)
            logging.debug("\t\t\tTo-Neuron Bias: %f", to_neuron.bias.value)
            logging.debug("\t\t\tTo-Neuron Activation: %f", to_neuron.activation)
    def _single_forward_pass(self, network_input):
        input_layer: Layer = self.layers[0]
        input_layer.set_neuron_activations(network_input)
        for from_layer in self.layers[:-1]:
            self._layer_forward_pass(from_layer)
    def _single_backprop_pass(self, target):
        logging.debug("\t------Starting BackProp------")
        self.set_output_neuron_error(target)
        for layer_weight in self.layer_weights[::-1]:
            self.learn_step_for_layer_from_neurons(layer_weight)
            self.learn_step_for_layer_weight_gradients(layer_weight)

    def _epoch_pass(self, network_inputs, targets):
        epoch_loss = 0
        for idx, input_target_pair in enumerate(zip(network_inputs, targets)):
            network_input, target = input_target_pair
            # forward pass
            logging.debug(f"\t------Input idx: {idx}, Input: {network_input}, Target: {target}------")
            self._single_forward_pass(network_input)
            output_neurons = self.get_output_neurons()
            network_output: list[float] = [neuron.activation for neuron in output_neurons]
            logging.debug(f"\t------Network Output: {network_output}------")
            epoch_loss += l2_loss(network_output, target)
            # backward pass
            self._single_backprop_pass(target)
        network_loss = epoch_loss / len(targets)
        logging.debug(f"------Avg Epoch Loss: {network_loss}------")
        return network_loss
    def fit(self, inputs, targets, epochs=3):
        logging.debug(f"{[layer.layer_idx for layer in self.layers]}")
        epoch_losses = []
        for epoch in range(epochs):
            logging.debug(f"------Epoch #{epoch}------")
            epoch_network_loss = self._epoch_pass(inputs, targets)
            epoch_losses.append(epoch_network_loss)
        logging.warn(f"------Epoch Losses: {epoch_losses}------")
