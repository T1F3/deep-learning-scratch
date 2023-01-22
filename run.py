import random
import statistics

from naive_mlp import (
    Network,
    Layer
)


def true_func(X):
    return (2.35 * X[0]) + (1.6 * X[1]) + (4.3 * X[2])

NUM_SAMPLES = 5
inputs = [[random.randint(1, 9) for __ in range(3)] for _ in range(NUM_SAMPLES)]
targets = [[true_func(input) + random.random()] for input in inputs]


def norm(X):
    x_stdev = statistics.stdev(X)
    x_mean = statistics.mean(X)
    return [(x - x_mean) / x_stdev for x in X]

stripped_targets = [target[0] for target in targets]
norm_targets = norm(stripped_targets)
padded_norm_targets = [[target] for target in norm_targets]

TRAIN_RATIO = 0.7
NUM_TRAIN_SAMPLES = int(NUM_SAMPLES * TRAIN_RATIO)
train_inputs, test_inputs = inputs[:NUM_TRAIN_SAMPLES], inputs[NUM_TRAIN_SAMPLES:]
train_targets, test_targets = targets[:NUM_TRAIN_SAMPLES], targets[NUM_TRAIN_SAMPLES:]

xor_network = Network()
xor_network = (
    xor_network
    .add_layer(Layer(3))
    .add_layer(Layer(6))
    .add_layer(Layer(6))
    .add_layer(Layer(1))
)
xor_network.fit(train_inputs, train_targets)
