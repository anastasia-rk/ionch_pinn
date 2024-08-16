import torch as pt
import torch.nn as nn
from time import perf_counter
from functools import partial
import numpy as np
import os
import pickle as pkl
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (20,10)
plt.rcParams['figure.dpi'] = 400
plt.style.use("ggplot")

# define nn class for testing
class FCN(nn.Module):
    """
    Defines a standard fully-connected network in PyTorch. this is a simple feedforward neural network
    with a sigmoid activation function.
    """
    def __init__(self, N_INPUT, N_OUTPUT, N_HIDDEN, N_LAYERS):
        super().__init__()
        activation = nn.Sigmoid
        self.first_layer = nn.Sequential(*[
            nn.Linear(N_INPUT, N_HIDDEN),
            activation()])
        self.hidden_layers = nn.Sequential(*[
            nn.Sequential(*[
                nn.Linear(N_HIDDEN, N_HIDDEN),
                activation()]) for _ in range(N_LAYERS - 1)])
        self.output_layer = nn.Linear(N_HIDDEN, N_OUTPUT)

    def forward(self, x):
        x = self.first_layer(x)
        x = self.hidden_layers(x)
        x = self.output_layer(x)
        return x

class FCN_multi_input(nn.Module):
    """
    Defines a FCN with multiple inputs. This is a simple feedforward neural network
    N_INPUTS - the number of input layers (one per time & each parameter range)
    LEN_INPUTS - the length of inputs
    N_OUTPUT - the number of outputs
    N_HIDDEN - the lenght of hidden layers
    N_LAYERS - the number of hidden layers - this excludes the first hidden layer that concatenates the hidden layers
    """
    def __init__(self, N_INPUTS, LEN_INPUTS, N_OUTPUT, N_HIDDEN, N_LAYERS):
        super().__init__()
        activation = nn.Sigmoid
        # dims_concatenated = sum(DIM_INPUTS)
        # we are just defining a separate first layer for each input, no need to connect them in initialisation
        for iInput in range(N_INPUTS):
            setattr(self, f'first_layer_{iInput}', nn.Sequential(*[
                nn.Linear(LEN_INPUTS[iInput], N_HIDDEN),
                activation()]))
        # we expect to concatenate the input layers along the dimension that corresponds to the input length, so the the
        # concatenated output will be sum(dimensions of all inputs)xN_hidden, and we probably don't need a concat_hidden

        # # add the hidden layer that that takes the concatenated output of the first layers as input
        # self.conctat_hidden = nn.Sequential(*[
        #     nn.Sequential(*[
        #         nn.Linear(dims_concatenated, N_HIDDEN),
        #         activation()])])
        # then add fully connected layers as hidden layers
        self.hidden_layers = nn.Sequential(*[
            nn.Sequential(*[
                nn.Linear(N_HIDDEN, N_HIDDEN),
                activation()]) for _ in range(N_LAYERS - 1)])
        self.output_layer = nn.Linear(N_HIDDEN, N_OUTPUT)

    def forward(self, xs):
        # this should take in the list of inputs and pass them through the network
        first_layer_outputs = []
        # pass each input through its own first layer
        for ix, x in enumerate(xs):
            x = getattr(self, f'first_layer_{ix}')(x)
            first_layer_outputs.append(x)
        # compute a geometric product of all first layer outputs to generate the tensor output
        for i in range(len(first_layer_outputs)):
            if i == 0:
                x = first_layer_outputs[i]
            else:
                x = pt.tensordot(x,first_layer_outputs[i])
        # concatenate the outputs of the first layers
        x = pt.cat(first_layer_outputs, dim=0)
        # # pass the concatenated output through the first hidden layer
        # x = self.conctat_hidden(x)
        # pass the output through the rest of the hidden layers
        x = self.hidden_layers(x)
        # pass the output through the output layer
        x = self.output_layer(x)
        return x


def pretty_axis(ax, legendFlag=True):
    # set axis labels and grid
    ax.set_facecolor('white')
    ax.grid(which='major', color='grey', linestyle='solid', alpha=0.2, linewidth=1)
    if legendFlag:
        ax.legend(loc='best', fontsize=12,framealpha=0.5)
    return ax


# from generate_data import *
if __name__ == '__main__':
    # # load the protocols
    # load_protocols
    # # generate the segments with B-spline knots and intialise the betas for splines
    # jump_indeces, times_roi, voltage_roi, knots_roi, collocation_roi, spline_order = generate_knots(times)
    # jumps_odd = jump_indeces[0::2]
    # jumps_even = jump_indeces[1::2]
    # nSegments = len(jump_indeces[:-1])
    # print('Inner optimisation is split into ' + str(nSegments) + ' segments based on protocol steps.')

    device = pt.device("cpu")

    # create neural network
    N = nn.Sequential(nn.Linear(1, 30), nn.Sigmoid(), nn.Linear(30,1, bias=False))

    # Initial condition
    A = 0.

    # The Psi_t function
    Psi_t = lambda x: A + x * N(x)

    # The right hand side function
    f = lambda x, Psi: pt.exp(-x / 5.0) * pt.cos(x) - Psi / 5.0

    # The loss function
    def loss(x):
        x.requires_grad = True
        outputs = Psi_t(x)
        Psi_t_x = pt.autograd.grad(outputs, x, grad_outputs=pt.ones_like(outputs),
                                      create_graph=True)[0]
        return pt.mean((Psi_t_x - f(x, outputs)) ** 2)


    # Optimize (same algorithm as in Lagaris)
    optimizer = pt.optim.LBFGS(N.parameters())

    # The collocation points used by Lagaris
    x = pt.Tensor(np.linspace(0, 2, 200)[:, None])


    # Run the optimizer
    def closure():
        optimizer.zero_grad()
        l = loss(x)
        l.backward()
        return l


    for i in range(10):
        optimizer.step(closure)

    # compare the result to the true solution
    xx = np.linspace(0, 2, 200)[:, None]
    with pt.no_grad():
        yy = Psi_t(pt.Tensor(xx)).numpy()
    ## this is true solution
    yt = np.exp(-xx / 5.0) * np.sin(xx)

    fig, ax = plt.subplots(dpi=100)
    ax.plot(xx, yt, label='True')
    ax.plot(xx, yy, '--', label='Neural network approximation')
    ax.set_xlabel('$x$')
    ax.set_ylabel('$\Psi(x)$')
    ax = pretty_axis(ax)
    plt.legend(loc='best')
    fig.savefig('Figures/neural_network_approximation.png')
