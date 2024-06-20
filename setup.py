import torch as pt
import torch.nn as nn
from time import perf_counter
from functools import partial
import numpy as np
import os
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (20,10)
plt.rcParams['figure.dpi'] = 400
plt.style.use("ggplot")


def pretty_axis(ax, legendFlag=True):
    # set axis labels and grid
    ax.set_facecolor('white')
    ax.grid(which='major', color='grey', linestyle='solid', alpha=0.2, linewidth=1)
    if legendFlag:
        ax.legend(loc='best', fontsize=12)
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
