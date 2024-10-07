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

########################################################################################################################


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

    def train(self, data_loader, N_EPOCHS, validation_data=None):
        all_cost_names = ['IC', 'RHS', 'L1', 'Data', 'Penalty', 'Total']
        stored_costs = {name: [] for name in all_cost_names}
        for epoch in range(N_EPOCHS):
            running_losses = self._train_iteration(data_loader)
            for iLoss in range(len(all_cost_names)):
                stored_costs[all_cost_names[iLoss]].append(running_losses[iLoss])
            running_loss = running_losses[-1]
            # validation placeholder
            val_loss = None
            if validation_data is not None:
                y_hat = self(validation_data['X'])
                val_loss = self.lossFct(input=y_hat, target=validation_data['y']).detach().numpy()
                print('[%d] loss: %.3f | validation loss: %.3f' %
                      (epoch + 1, running_loss, val_loss))
            else:
                print('[%d] loss: %.3f' %
                      (epoch + 1, running_loss))

    def _train_iteration(self, data_loader, lambdas, time_of_domain, IC_stacked_domain, IC, scaling_args):
        rhs_error_state_weights, t_scaling_coeff, param_scaling_coeff = scaling_args
        lambda_ic, lambda_rhs, lambda_data, lambda_l1, lambda_penalty = lambdas
        # prepare losses for cumulation
        running_loss = 0.0
        running_IC_loss = 0.0
        running_RHS_loss = 0.0
        running_data_loss = 0.0
        running_L1_loss = 0.0
        running_penalty_loss = 0.0
        for i_batch, (input_batch, *precomputed_RHS_batch, target_batch) in enumerate(data_loader):
            # if we sent all the parts of the dataset to device, we do not need to pass them individually
            # # extract time from the input batch - in the tensor dataset, the time is the first element of the input batch
            # time_of_batch = input_batch[0,:,0]
            # time_of_batch = time_of_batch.cpu().detach().numpy()
            # time_of_batch =time_of_batch/t_scaling_coeff
            time_of_batch = time_of_domain
            # compute the gradient loss
            state_batch = self(input_batch)
            # use custom detivarive function to compute the derivatives of outputs, because grad assumed that the output is a scalar
            dxdt = derivative_multi_input(state_batch, input_batch)
            dxdt = dxdt * t_scaling_coeff  # restore to the original scale
            # compute the RHS from the precomputed parameters - note that this is in the original scale too
            rhs_pinn = RHS_from_precomputed(state_batch, precomputed_RHS_batch)
            # compute the current tensor to compare with the measured current
            current_pinn = observation_tensors(time_of_batch, state_batch, input_batch)
            current_pinn = current_pinn / param_scaling_coeff
            ################################################################################################################
            # compute the loss function
            ################################################################################################################
            # compute the RHS loss
            if lambda_rhs != 0:
                error_rhs = (rhs_pinn - dxdt) ** 2
                # aplify the error along the second state dimension by multiplying it by 10
                for iState in range(error_rhs.shape[-1]):
                    error_rhs[..., iState] = error_rhs[..., iState] * rhs_error_state_weights[iState]
                # simple trapezoidal rule to compute the integral
                sumerror = error_rhs[:, 1:, ...] + error_rhs[:, :-1, ...]
                dt = input_batch[:, 1:, 0] - input_batch[:, :-1, 0]
                # this should only expand it once I think
                for iDim in range(len(sumerror.shape) - len(dt.shape)):
                    dt = dt.unsqueeze(-1)
                dt = dt.expand(sumerror.shape)
                loss_rhs = pt.sum(sumerror * dt / 2)
                del sumerror
            else:
                loss_rhs = pt.tensor(0, dtype=pt.float32).to(device)
            ################################################################################################################
            # compute the IC loss
            if lambda_ic != 0:
                # state_ic = state_domain[0,...] # if we are starting from 0 at the time domain
                state_ic = self(IC_stacked_domain)  # or we can put the IC domain through the pinn
                loss_ic = pt.sum((state_ic - IC) ** 2)
            else:
                loss_ic = pt.tensor(0, dtype=pt.float32).to(device)
            ################################################################################################################
            # commpute the data loss w.r.t. the current
            if lambda_data != 0:
                residuals_data = current_pinn - target_batch
                loss_data = pt.sum((residuals_data) ** 2)  # by default, pytorch sum goes over all dimensions
            else:
                loss_data = pt.tensor(0, dtype=pt.float32).to(device)
            ################################################################################################################
            # compute the L1 norm
            if lambda_l1 != 0:
                par_pinn = list(self.parameters())
                L1 = pt.tensor([par_pinn[l].abs().sum() for l in range(len(par_pinn))]).sum()
            else:
                L1 = pt.tensor(0, dtype=pt.float32).to(device)
            ################################################################################################################
            #  compute network output penalty (we know that each output must be between 0 and 1)
            target_penalty = pt.tensor(0, dtype=pt.float32).to(device)
            if lambda_penalty != 0:
                for iOutput in range(state_batch.shape[-1]):
                    lower_bound = pt.zeros_like(state_batch[..., iOutput]).to(device)
                    upper_bound = pt.ones_like(state_batch[..., iOutput]).to(device)
                    target_penalty += pt.sum(pt.relu(lower_bound - state_batch[..., iOutput])) + pt.sum(
                        pt.relu(state_batch[..., iOutput] - upper_bound))
            ################################################################################################################
            def closure():
                # zero the gradients
                if pt.is_grad_enabled():
                    self.optim.zero_grad()
                # compute the total loss
                loss = lambda_ic * loss_ic + lambda_rhs * loss_rhs + lambda_data * loss_data + lambda_l1 * L1 + lambda_penalty * target_penalty
                # the backward pass computes the gradient of the loss with respect to the parameters
                loss.backward(retain_graph=True)
                return loss

            # make a step in the parameter space, then combine the batch losses
            self.optim.step(closure)
            loss = closure()
            running_loss += loss.item()
            running_IC_loss += loss_ic.item()
            running_RHS_loss += loss_rhs.item()
            running_data_loss += loss_data.item()
            running_L1_loss += L1.item()
            running_penalty_loss += target_penalty.item()
        return running_IC_loss, running_RHS_loss, running_data_loss, running_L1_loss, running_penalty_loss, running_loss


def derivative_multi_input(outputs,inputs):
    """
    This function computes the derivative of the outputs w.r.t. the inputs and returns only derivatives w.r.t to time
    as a stacked tensor
    :param outputs: outputs of the NN
    :param inputs: inputs of the NN, the first one being time
    :return: the derivatives of all outputs w.r.t. time of size [nSamples x nTimes x nOutputs]
    """
    list_of_grads = []
    # the number of outputs is the last element of the shape of the tensor - this will hold becaus of the way we stack inputs
    nOutputs = outputs.shape[-1]
    # we want to iterate over the outputs, not their shape
    for iOutput in range(nOutputs):
        output = outputs[...,iOutput]
        ones = pt.ones_like(output)
        grad = pt.autograd.grad(output, inputs, grad_outputs=ones, create_graph=True)[0] # this will compute the gradient of the output w.r.t. all inputs!
        # the time is alway the first in the list of inputs, so we need to only store the first element of the last dimension of grad
        # I think with the way we now create the input tensor this line is wrong - check this!
        grad_wrt_time_only = grad[...,0].unsqueeze(-1) # need to unsquze to make sure we can stack them along the state dimension
        list_of_grads.append(grad_wrt_time_only)
    #  create a tensor from the list of tensor by stacking them along the last dimension
    return pt.cat(list_of_grads,dim=-1)

# in the function above, the only thing that will change from epoch to epoch is the state tensor x, so we should be able
# to pre-compute everything up until tau_a, a_inf, tau_r, r_inf, and then just compute the RHS from those tensors
# this will save us some time in the computation
def RHS_tensors_precompute(t, x, theta):
    """
    This function procudes a part of the right hand side of the ODE that is independent of the state tensor x.
    We can compute this becasue we know all parameter points for taining the PINN
    Only need to run this once before entering the training loop
    :param t: time vector - has to be numpy to use interpolate from scipy
    :param x: state tensor of size [nSamples x nTimes x nStates]
    :param theta: parameter tensor of size [nSampes x nTimes x nParams]
    """
    # extract all params that are not time
    p = theta[:,:,1:]
    # extract a and r from the x tensor - this will be the first and second column of the output tensor
    a_state = x[..., 0] # we only need the state here to get its shape
    # get the shape of the state
    state_shape = a_state.shape
    # in this case, we are assuming that the first dimension of the tensor is the time dimension,
    # and all other dimensions correspond to possible parameter values.
    # So, when we create a voltage tensor, that is the same for all parameter values
    v = pt.tensor(V(t), dtype=pt.float32)
    # add as many dimensions to the tensor as there are elements in the shape of the state tensor
    for idim in range(len(state_shape) - len(v.shape)):
        v = v.unsqueeze(0)
    #  now we should be able to expand the original tensor along those dimensions
    v = v.expand(state_shape).to(device) # note that this does not allocate new memony, but rather creates a view of the original tensor
    # then we work with the tensors as if they were scalars (I think)
    k1 = p[...,0] * pt.exp(p[...,1] * v)
    k2 = p[...,2] * pt.exp(-p[...,3] * v)
    k3 = p[...,4] * pt.exp(p[...,5] * v)
    k4 = p[...,6] * pt.exp(-p[...,7] * v)
    # these tensors will have the same shape as the state tensor
    tau_a = 1 / (k1 + k2)
    a_inf = tau_a * k1
    tau_r = 1 / (k3 + k4)
    r_inf = tau_r * k4
    return tau_a, a_inf, tau_r, r_inf

# then define a model that will take the precomputed tensors and compute the RHS
def RHS_from_precomputed(x, precomputed_params):
    """
    This function computes the RHS of the ODE using the precomputed parameters and the state tensor x
    :param x: tensor of states
    :param precomputed_params: tensor of precomputed parameters
    :return: right hand side of the ODE evaluated at x
    """
    a = x[..., 0]
    r = x[..., 1]
    tau_a, a_inf, tau_r, r_inf = precomputed_params
    dx = pt.stack([(a_inf - a) / tau_a, (r_inf - r) / tau_r], dim=len(a.shape))
    return dx

def observation_tensors(t, x, theta):
    """
        This function produces the current tensor from the state tensor x and the parameter tensor theta
    :param t: time vector - has to be numpy to use interpolate from scipy
    :param x: state tensor of size [nSamples x nTimes x nStates]
    :param theta: parameter tensor of size [nSampes x nTimes x nParams]
    :return: the current tensor of size [nSamples x nTimes x 1]
    """
    a = x[..., 0]
    r = x[..., 1]
    g = theta[:,:,-1]
    g = g.to(device)
    state_shape = a.shape
    v = pt.tensor(V(t), dtype=pt.float32)
    # add as many dimensions to the tensor as there are elements in the shape of the state tensor
    for idim in range(len(state_shape) - len(v.shape)):
        v = v.unsqueeze(0)
    #  now we should be able to expand the original tensor along those dimensions
    v = v.expand(state_shape).to(device)
    EK = -80
    EK_tensor = pt.tensor(EK, dtype=pt.float32).expand(state_shape).to(device)
    # the only input in this the conductance that has been broadcasted to the shape of a single state
    current =  g * a * r * (v - EK_tensor)
    return current

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
