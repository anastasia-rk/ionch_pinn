import matplotlib.pyplot as plt
import numpy as np
from preliminaries import *
from generate_data import *
from tqdm import tqdm
from plot_figures import *
from torch.utils.data import DataLoader, TensorDataset
import math

from save_to_gif import RHS_name


# definitions
def generate_parameter_sample(nSamples, nParams, lower, upper, rateConstraint=False):
    """
    This function generates a sample of parameters for the training of the PINN.
    The samples are generated from a lognormal distribution and then bounded by the lower and upper bounds.
    If the rateConstraint is set to True, the function will also check if the rate constants are within the bounds
    Outputs a 2D tensor of size nParams x nSamples.
    :param nSamples: number of samples to generate
    :param nParams: number of parameters to generate
    :param lower: lower bounds for the parameters
    :param upper: upper bounds for the parameters
    :param rateConstraint: if True, the function will also check if the rate constants are within the bounds
    :return: randomly generated paramer values, a tensor of size nParams x nSamples
    """
    # check if nParams matches the length of the lower and upper bounds
    if len(lower) != nParams or len(upper) != nParams:
        raise ValueError('The length of the lower and upper bounds should match the number of parameters.')
    ###############################################################################################################
    # generate the tensor
    mean = -4
    std_dev = 5.0
    tensor_sample = pt.distributions.LogNormal(mean, std_dev).sample((nSamples, nParams))
    tensor_bounded_list = []
    for iParam in range(tensor_sample.shape[-1]): # iterate over the parameters
        tensor_i = tensor_sample[..., iParam] # sample of values per parameter
        # lower bound
        below_min_mask = tensor_i < lower[iParam]
        tensor_i[below_min_mask] = pt.rand_like(tensor_i[below_min_mask]) * (upper[iParam] - lower[iParam]) + lower[iParam]
        # upper bound
        above_max_mask = tensor_i > upper[iParam]
        tensor_i[above_max_mask] = pt.rand_like(tensor_i[above_max_mask]) * (upper[iParam] - lower[iParam]) + lower[iParam]
        # add to list for stacking
        tensor_bounded_list.append(tensor_i)
    # stack along a new dimension
    bounded_tensor = pt.stack(tensor_bounded_list, dim=len(tensor_sample.shape) - 1)
    # transpose the tensor to make the the n_samples the last dimension and return it
    bounded_tensor = bounded_tensor.T
    ###############################################################################################################
    # Check the rates of the bounded tensor
    if rateConstraint:
        V_min = -120
        V_max = 60
        rr_min = 1.67e-5
        rr_max = 1000
        V_bound_per_rate = pt.tensor([V_max, V_min, V_max, V_min]) # positive and negative rates alternate
        sign_per_rate = pt.tensor([1, -1, 1, -1]) # positive and negative rates alternate
        for iSample in range(bounded_tensor.shape[-1]): # now we iterate over the samples
            tensor_i = bounded_tensor[..., iSample] # get the sample
            for iPair in range(4): # for each pair of parameters, compute the rate and check if it is within the bounds
                #  note that the last parameter is the conductance
                r = tensor_i[2*iPair] * pt.exp(sign_per_rate[iPair] * tensor_i[2*iPair+1] * V_bound_per_rate[iPair])
                if r <= rr_min:
                    # compute tensor_i[1] to be within the constraint (rr_min + 20%) and send it to the original tensor
                    bounded_tensor[2*iPair+1,iSample] = sign_per_rate[iPair]*(pt.log(pt.tensor(rr_min*1.5)) - pt.log(tensor_i[2*iPair])) / V_bound_per_rate[iPair]
                elif r >= rr_max:
                    # compute tensor_i[1] to be within the constraint (rr_max - 20%) and send it to the original tensor
                    bounded_tensor[2*iPair+1,iSample] = sign_per_rate[iPair]*(pt.log(pt.tensor(rr_max*0.5)) - pt.log(tensor_i[2*iPair])) / V_bound_per_rate[iPair]
    return bounded_tensor

def stack_inputs(time_input,generated_param_samples):
    """
    This function generates a combined input tensor from an ordered time_input tensor of size nTimesx1
     and a parameter sample tensor of size nParams x nSamples
    outputs the stacked tensor of size nSampes x nTimes x nInputs, where nInputs is nParams+1
    """
    # make sure that the time input is stackable
    if len(time_input.shape) == 1:
        time_input = time_input.unsqueeze(-1)
    # create the combined input tensor
    tensors_to_stack = []
    for iSample in range(generated_param_samples.shape[-1]):
        single_param_sample = generated_param_samples[..., iSample]
        param_sample_expanded = single_param_sample.expand(time_input.shape[0], single_param_sample.shape[0])
        concat_tensor = pt.cat([time_input, param_sample_expanded], dim=1)
        tensors_to_stack.append(concat_tensor)
    # now stack all tensors along the first dimension
    stacked_input = pt.stack(tensors_to_stack, dim=0)
    return stacked_input

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
    EK_tensor = pt.tensor(EK, dtype=pt.float32).expand(state_shape).to(device)
    # the only input in this the conductance that has been broadcasted to the shape of a single state
    current =  g * a * r * (v - EK_tensor)
    return current

########################################################################################################################
# start the main script
# load the protocols
EK = -80
V_min = -120
V_max = 60
tlim = [0, 14899]
times = np.linspace(*tlim, tlim[-1] - tlim[0], endpoint=False)
load_protocols
# generate the segments with B-spline knots and intialise the betas for splines
jump_indeces, times_roi, voltage_roi, knots_roi, collocation_roi, spline_order = generate_knots(times)
jumps_odd = jump_indeces[0::2]
jumps_even = jump_indeces[1::2]
nSegments = len(jump_indeces[:-1])
# use collocation points as an array to get the training times
unique_times = np.unique(np.hstack(knots_roi))
####################################################################################################################
# run the generative model - to generate current and two true states
model_name = 'HH'
if model_name.lower() not in available_models:
    raise ValueError(f'Unknown model name: {model_name}. Available models are: {available_models}.')
elif model_name.lower() == 'hh':
    thetas_true = thetas_hh_baseline
elif model_name.lower() == 'kemp':
    thetas_true = thetas_kemp
elif model_name.lower() == 'wang':
    thetas_true = thetas_wang
solution, current_model = generate_synthetic_data(model_name, thetas_true, times)
true_states = solution.sol(times)
# set signal to noise ratio in decibels
snr_db = 30
snr = 10 ** (snr_db / 10)
current_true = current_model(times, solution, thetas_true, snr=snr)
####################################################################################################################
#set up the model for the PINN training
IsArchTest = False # if ture, this will fit the PINN output directly to true state, if false, it will fit to the RHS
InLogScale = False # if true, the conductances will be in log scale
rhs_name = 'hh_all_inputs_model'
available_rhs = ['test_all_inputs_model', 'hh_all_inputs_model']
####################################################################################################################
# set up the domain for the PINN
IC = [0, 1]
scaled_domain_size = 10 # this is max value of the domain - we are dealing only with positive inputs so do not need to worry about the minimum
# create time domain from knots
t_scaling_coeff = scaled_domain_size / unique_times[-1]
t_domain_unscaled = pt.tensor(unique_times, dtype=pt.float32).requires_grad_(True)
t_domain = t_domain_unscaled * t_scaling_coeff
if rhs_name.lower() == 'test_all_inputs_model':
    nSamples = 1
    # in this case, we just want to generate the sample of of true parameters
    param_sample_unscaled = pt.tensor(thetas_true).unsqueeze(-1).requires_grad_(True)
    # colour for plottin network output
    colours = ['darkorange']  # for the test, we only have one instance
elif rhs_name.lower() == 'hh_all_inputs_model':
    # generate sample for all HH parameters within given bounds
    min_val = pt.tensor([1e-7, 1e-7, 1e-7, 1e-7, 1e-7, 1e-7, 1e-7, 1e-7, 1e-5])
    max_val = pt.tensor([1e3, 0.4, 1e3, 0.4, 1e3, 0.4, 1e3, 0.4, 10])
    nSamples = 19 # number of parameter samples
    param_sample_unscaled = generate_parameter_sample(nSamples, len(max_val), min_val, max_val, rateConstraint=True)
    # I want to add the true data point to the sample to see if the PINN is able to fit that even when it struggles with others
    true_params_unscaled = pt.tensor(thetas_true).unsqueeze(-1)
    param_sample_unscaled = pt.cat([param_sample_unscaled, true_params_unscaled], dim=-1)
    nSamples = param_sample_unscaled.shape[-1]
    nPerBatch = 6  # number of samples per batch
    # if nSamples % nPerBatch != 0:
    #     raise ValueError('The number of samples should be divisible by the number of samples per batch.')
    # plot histograms of smapled values for each parameter
    param_names = ['p1', 'p2', 'p3', 'p4', 'p5', 'p6', 'p7', 'p8','g']
    fig, axs = plt.subplots(3, 3, figsize=(15, 15))
    axs = axs.ravel()
    for i in range(param_sample_unscaled.shape[0]):
        axs[i].hist(param_sample_unscaled[i, :].detach().numpy(),density=True, bins=20, alpha=0.3, color='darkorange')
        # axs[i].set_title(f'Parameter {i + 1} Histogram')
        axs[i].set_xlabel(param_names[i])
        axs[i] = pretty_axis(axs[i],legendFlag=False)
        # axs[i].set_xscale('log')
        # axs[i].set_ylabel('Frequency')
    # save the figure
    fig.tight_layout()
    fig.savefig('Figures/'+RHS_name.lower()+'_parameter_histograms.png', dpi=400)
    # set up the colour wheel for plotting output at different training samples
    colours = plt.cm.PuOr(np.linspace(0, 1, nSamples))
    # make a throwaway counntour plot to generate a heatmap of conductances
    fig, ax = plt.subplots(1, 1, figsize=(10, 6), dpi=400)
    Z = [[0,0],[0,0]]
    levels = np.linspace(0, nSamples-1, nSamples) # in this case we want to iterate over samples rather than values of parameters
    cond_heatmap = plt.contourf(Z, levels, cmap='PuOr')
    plt.clf()
    rr_min = pt.tensor(1.67e-5)
    rr_max = pt.tensor(1000)
    V_bound_per_rate = pt.tensor([V_max, V_min, V_max, V_min])  # positive and negative rates alternate
    sign_per_rate = pt.tensor([1, -1, 1, -1])
    rr_limit = pt.tensor([rr_min, rr_max])
    # plot parameter samples on p1 vs p2, p3 vs p4, p5 vs p6, p7 vs p8, and g axes and colour code then according to the colour wheel
    fig, axs = plt.subplots(2, 2, figsize=(10.5, 10))
    axs = axs.ravel()
    for i in range(4):
        # add bounaries of the parameter space as dashed lines
        axs[i].axvline(min_val[2*i].detach().numpy(), linestyle='--', alpha=0.7, color='grey')
        axs[i].axvline(max_val[2*i].detach().numpy(), linestyle='--', alpha=0.7, color='grey')
        axs[i].axhline(min_val[2*i+1].detach().numpy(), linestyle='--', alpha=0.7, color='grey')
        axs[i].axhline(max_val[2*i+1].detach().numpy(), linestyle='--', alpha=0.7,  color='grey',label='bounds')
        # add minimal and maximal rate lines
        p1_values = pt.logspace(pt.log(min_val[2*i]), pt.log(max_val[2*i]), base=math.e, steps=100)
        p2_values_min = (pt.log(rr_min) - pt.log(p1_values)) / (sign_per_rate[i]*V_bound_per_rate[i])
        p2_values_min = pt.max(p2_values_min, min_val[2*i+1]*pt.ones_like(p2_values_min))
        p2_values_max = (pt.log(rr_max) - pt.log(p1_values)) / (sign_per_rate[i]*V_bound_per_rate[i])
        axs[i].plot(p1_values.detach().numpy(), p2_values_min.detach().numpy(), linestyle='--', alpha=0.7, color='black',label=r'$r_{min}$')
        axs[i].plot(p1_values.detach().numpy(), p2_values_max.detach().numpy(), linestyle='-', alpha=0.7, color='black', label=r'$r_{max}$')
        axs[i].axvline(thetas_true[2 * i], linestyle='--', alpha=0.3,  color='magenta')
        axs[i].axhline(thetas_true[2 * i + 1], linestyle='--', alpha=0.3, color='magenta', label='truth')
        axs[i].scatter(param_sample_unscaled[2*i, :].detach().numpy(), param_sample_unscaled[2*i+1, :].detach().numpy(), c=colours)
        axs[i].set_xlabel(param_names[2*i])
        axs[i].set_ylabel(param_names[2*i+1])
        axs[i].set_xscale('log')
        axs[i].set_yscale('log')
        axs[i] = pretty_axis(axs[i],legendFlag=False)
    axs[-1].legend(loc='lower right')
    fig.tight_layout() #, w_pad=0.3, h_pad=0.3)
    cbar = fig.colorbar(cond_heatmap, ax=axs.tolist(), location='right' , aspect=50) #ticks=levels
    # cbar.ax.set_yticklabels(["{:.2f}".format(j + 1) for j in levels])
    cbar.ax.set_xlabel('j')
    # cbar.ax.yaxis.label.set_rotation(90)
    # save the figure
    # fig.tight_layout()
    fig.savefig('Figures/'+RHS_name.lower()+'_parameter_samples.png', dpi=400)
    # plot the conductances
    fig, ax = plt.subplots(1, 1, figsize=(5.5, 5), dpi=400)
    ax.axhline(min_val[-1].detach().numpy(), linestyle='--', alpha=0.7, color='grey')
    ax.axhline(max_val[-1].detach().numpy(), linestyle='--', alpha=0.7, color='grey',label='bounds')
    ax.axhline(thetas_true[-1], linestyle='--', alpha=0.3, color='magenta',label='truth')
    ax.scatter(np.arange(0,nSamples), param_sample_unscaled[-1, :].detach().numpy(), c=colours)
    ax = pretty_axis(ax,legendFlag=True)
    ax.legend(loc='lower left')
    # fig.tight_layout(pad=0.3) #, w_pad=0.3, h_pad=0.3)
    cbar = fig.colorbar(cond_heatmap, ax=ax, location='right', aspect=50) #ticks=levels
    # cbar.ax.set_yticklabels(["{:.2f}".format(j + 1) for j in levels])
    # cbar.ax.set_xlabel('j')
    # cbar.ax.yaxis.label.set_rotation(90)
    ax.set_xlabel('Sample index')
    ax.set_yscale('log')
    ax.set_ylabel('Conductance')
    fig.tight_layout()
    fig.savefig('Figures/'+RHS_name.lower()+'_conductance_samples.png', dpi=400)

## combine the tensors
####################################################################################################################
# stack inputs to be fed into PINN
# scale the parameter so that the largest value there does not exceed scaled_domain_size - all parameters are above 0
param_scaling_coeff =  scaled_domain_size / pt.max(param_sample_unscaled)
param_sample = param_sample_unscaled * param_scaling_coeff
param_sample = param_sample.requires_grad_(True)
stacked_domain_unscaled = stack_inputs(t_domain_unscaled, param_sample_unscaled)
stacked_domain = stack_inputs(t_domain, param_sample)
# get a separate domain to get the IC, in case we are not starting time at 0
IC_t_domain = pt.tensor([0.], dtype=pt.float32).requires_grad_(True)
IC_stacked_domain = stack_inputs(IC_t_domain, param_sample)
# solve the ODE and get the true state and measured current
time_of_domain = t_domain_unscaled.detach().numpy()
sol_for_x = sp.integrate.solve_ivp(hh_model, [0, t_domain_unscaled[-1]], y0=IC, args=[thetas_true], dense_output=True,
                                   method='LSODA', rtol=1e-8, atol=1e-8)
IC = pt.tensor(sol_for_x.sol(time_of_domain[0]), dtype=pt.float32)  # get the IC from true state
state_true = sol_for_x.sol(time_of_domain)
measured_current = current_model(time_of_domain, sol_for_x, thetas_true, snr=10)
# convert the true state and the measured current into a tensor
state_true_tensor = pt.tensor(state_true.transpose())
measured_current_tensor = pt.tensor(measured_current, dtype=pt.float32)
####################################################################################################################
# set up the neural network
domain_shape = stacked_domain.shape
loss_seq = []
pt.manual_seed(123)
nLayers = 4
nHidden = 800
nOutputs = 2
nInputs = domain_shape[-1]
marks = [int(i) for i in np.linspace(0, nHidden, 3)]
# define a neural network to train
device = pt.device('mps' if pt.backends.mps.is_available() else 'cuda' if pt.cuda.is_available() else 'cpu')
print(f'Using device: {device}')
pinn = FCN(nInputs, nOutputs, nHidden, nLayers).to(device)
########################################################################################################################
# storing parameters for plotting
all_names = [name for _, (name, _) in enumerate(pinn.named_parameters())]
# get unique layer names
first_layer_name = all_names[0].split('.')[0]
last_layer_name = all_names[-1].split('.')[0]
hidden_layer_names = [name.split('.')[0] + '.' + name.split('.')[1]  for name in all_names[2:-2]]
# drip elements of layer list that are duplicates but preserve order - done in weird way from stackoverflow!
layer_names = [first_layer_name] + list(dict.fromkeys(hidden_layer_names)) + [last_layer_name]
# get the biases of the first layer
biases = pinn.first_layer[0].bias.data
# provide larger biases for the first layer as the initialsiation
a = pt.tensor(-10)
b = pt.tensor(10)
biases_new = (b - a) * pt.rand_like(biases) + a
# set the biases of the first layer
pinn.first_layer[0].bias.data = biases_new
########################################################################################################################
# define the optimiser and the loss function
optimiser = pt.optim.Adam(pinn.parameters(),lr=1e-3)
# loss = pt.tensor(100) # if we are using a while loop, we need to initialise the loss
i = 0
lambda_ic = 1e-2 # 1e-2 # weight on the gradient fitting cost
lambda_rhs = 1 # weight on the right hand side fitting cost
lambda_l1 = 0 # weight on the L1 norm of the parameters
lambda_data = 1e-7 # weight on the data fitting cost
lambda_penalty = 1e-3 # weight on the output penalty
lambdas = [lambda_ic, lambda_rhs, lambda_l1, lambda_data,lambda_penalty]
# placeholder for storing the costs
all_cost_names = ['IC', 'RHS', 'L1', 'Data','Penalty']
stored_costs = {name: [] for name in all_cost_names}
########################################################################################################################
# we need to expand the measured current tensor to match the shape of the pinn current tensor
# pass the domain through the PINN to get the output - it does not matter what the output is, we just want to get the dimensions
stacked_domain_unscaled = stacked_domain_unscaled.to(device)
stacked_domain = stacked_domain.to(device)
state_domain = pinn(stacked_domain)
# for the training purposes, we can pre-compute part of the RHS since it will only depend on the domain that does not change
# this will save us some time in the computation - note that this is in the original scale
precomputed_RHS_params = RHS_tensors_precompute(time_of_domain, state_domain, stacked_domain_unscaled)
current_pinn = observation_tensors(time_of_domain, state_domain, stacked_domain)
current_pinn = current_pinn/param_scaling_coeff
# now expand the measured current tensor to match the shape of the current tensor that will be produced by the PINN
current_shape = current_pinn.shape
for iDim in range(len(current_shape)-1):
    measured_current_tensor = measured_current_tensor.unsqueeze(0)
measured_current_tensor = measured_current_tensor.expand(current_shape).to(device)
####################3##################################################################################################
# send thing to device
# create a tensor dataset - we must include parts of RHS parameters that are precomputed to split them into appropriate parts
# note that precomputed_RHS_params is a tuple of tensors - we need to unpack it to send it into the dataloader
dataset = TensorDataset(stacked_domain, *precomputed_RHS_params,  measured_current_tensor)
dataloader = DataLoader(dataset, batch_size=nPerBatch, shuffle=False)
IC_stacked_domain = IC_stacked_domain.to(device)
IC = IC.to(device)
########################################################################################################################
## plots to check the network architecture
# plot the activation functions of the network as a function of domain
# fig, axes = plot_layers_as_bases(pinn, t_domain, t_domain)
# axes[-1].set_xlabel('Input domain at initialisation')
# plt.tight_layout()
# # save the figure
# plt.savefig('Figures/Activators_as_basis.png',dpi=400)
# # plt.show()

# plot the weights and biases of the network to check if everything is plotted correctly
fig, axes = plot_pinn_params_all_inputs(pinn)
# set the suptitle
fig.suptitle('test', fontsize=16)
plt.subplots_adjust(left=0, right=1, wspace=0.1, hspace=1)
# save the figure
fig.savefig('Figures/Weights_and_biases.png', dpi=400)
########################################################################################################################
# loop settings
plotEvery = 5000
maxIter = 250001
rhs_error_state_weights = [1,1]
# start the optimisation loop
for i in tqdm(range(maxIter)):
    for i_batch, (input_batch, *precomputed_RHS_batch, output_batch) in enumerate(dataloader):
        # if we sent all the parts of the dataset to device, we do not need to pass them individually
        # # extract time from the input batch - in the tensor dataset, the time is the first element of the input batch
        # time_of_batch = input_batch[0,:,0]
        # time_of_batch = time_of_batch.cpu().detach().numpy()
        # time_of_batch =time_of_batch/t_scaling_coeff
        time_of_batch = time_of_domain
        # zero the gradients
        optimiser.zero_grad()
        # compute the gradient loss
        state_batch = pinn(input_batch)
        # use custom detivarive function to compute the derivatives of outputs, because grad assumed that the output is a scalar
        dxdt = derivative_multi_input(state_batch, input_batch)
        dxdt = dxdt * t_scaling_coeff  # restore to the original scale
        # compute the RHS from the precomputed parameters - note that this is in the original scale too
        rhs_pinn = RHS_from_precomputed(state_batch, precomputed_RHS_batch)
        # compute the current tensor to compare with the measured current
        current_pinn = observation_tensors(time_of_batch, state_batch, input_batch)
        current_pinn = current_pinn/param_scaling_coeff
        ################################################################################################################
        # compute the loss function
        ################################################################################################################
        # compute the RHS loss
        if lambda_rhs != 0:
            error_rhs = (rhs_pinn - dxdt)**2
            # aplify the error along the second state dimension by multiplying it by 10
            for iState in range(error_rhs.shape[-1]):
                error_rhs[...,iState] = error_rhs[...,iState] * rhs_error_state_weights[iState]
            # simple trapezoidal rule to compute the integral
            sumerror = error_rhs[:,1:, ...] + error_rhs[:,:-1, ...]
            dt = input_batch[:,1:,0] - input_batch[:,:-1,0]
            # this should only expand it once I think
            for iDim in range(len(sumerror.shape) - len(dt.shape)):
                dt = dt.unsqueeze(-1)
            dt = dt.expand(sumerror.shape)
            loss_rhs = pt.sum( sumerror * dt / 2)
            del sumerror
        else:
            loss_rhs = pt.tensor(0,dtype=pt.float32).to(device)
        stored_costs['RHS'].append(loss_rhs.item())
        ################################################################################################################
        # compute the IC loss
        if lambda_ic != 0:
            # state_ic = state_domain[0,...] # if we are starting from 0 at the time domain
            state_ic = pinn(IC_stacked_domain) # or we can put the IC domain through the pinn
            loss_ic = pt.sum((state_ic - IC)**2)
        else:
            loss_ic = pt.tensor(0,dtype=pt.float32).to(device)
        stored_costs['IC'].append(loss_ic.item())
        ################################################################################################################
        # commpute the data loss w.r.t. the current
        if lambda_data != 0:
            residuals_data = current_pinn - output_batch
            loss_data = pt.sum((residuals_data)**2) # by default, pytorch sum goes over all dimensions
        else:
            loss_data = pt.tensor(0,dtype=pt.float32).to(device)
        stored_costs['Data'].append(loss_data.item())
        ################################################################################################################
        # compute the L1 norm
        if lambda_l1 != 0:
            par_pinn = list(pinn.parameters())
            L1 = pt.tensor([par_pinn[l].abs().sum() for l in range(len(par_pinn))]).sum()
        else:
            L1 = pt.tensor(0,dtype=pt.float32).to(device)
        stored_costs['L1'].append(L1.item())
        ################################################################################################################
        #  compute network output penalty (we know that each output must be between 0 and 1)
        output_penalty = pt.tensor(0,dtype=pt.float32).to(device)
        if lambda_penalty != 0:
            for iOutput in range(nOutputs):
                lower_bound = pt.zeros_like(state_batch[...,iOutput]).to(device)
                upper_bound = pt.ones_like(state_batch[...,iOutput]).to(device)
                output_penalty += pt.sum(pt.relu(lower_bound-state_batch[...,iOutput])) + pt.sum(pt.relu(state_batch[...,iOutput]-upper_bound))
        stored_costs['Penalty'].append(output_penalty.item())
        ################################################################################################################
        # compute the total loss
        loss = lambda_ic*loss_ic + lambda_rhs*loss_rhs  + lambda_data*loss_data + lambda_l1*L1 + lambda_penalty*output_penalty
        # store the cost function into the list
        loss_seq.append(loss.item())
        # # retain gradient of the cost to print and save it
        # loss.retain_grad()
        # the backward pass computes the gradient of the loss with respect to the parameters
        loss.backward(retain_graph=True)
        # save the gradient of the loss
        # store_grad = loss.grad
        optimiser.step()
    ####################################################################################################################
    # occasionally plot the output, save the network state and plot the costs
    if i % plotEvery == 0:
        # save  the model andcosts
        # save the model to a pickle file
        pt.save(pinn.state_dict(),
                'Models/' + rhs_name.lower() + '_' + str(nLayers) + '_layers_' + str(nHidden) + '_nodes_' + str(
                    nInputs) + '_ins' + str(nOutputs) + '_outs.pth')
        # save the costs to a pickle file
        with open('Models/' + rhs_name.lower() + '_' + str(nLayers) + '_layers_' + str(nHidden) + '_nodes_' + str(
                nInputs) + '_ins' + str(nOutputs) + '_costs.pkl', 'wb') as f:
            pkl.dump(stored_costs, f)
        # plotting for different samples - we need to call the correct tenso since we have changed how the input tensors are generated
        # in order to plot over the whole interval, we need to produce output
        state_domain = pinn(stacked_domain)
        # use custom detivarive function to compute the derivatives of outputs, because grad assumed that the output is a scalar
        dxdt = derivative_multi_input(state_domain, stacked_domain)
        dxdt = dxdt * t_scaling_coeff # restore to the original scale
        rhs_pinn = RHS_from_precomputed(state_domain, precomputed_RHS_params)
        current_pinn = observation_tensors(time_of_domain, state_domain, stacked_domain)
        current_pinn = current_pinn/param_scaling_coeff
        ################################################################################################################
        # plot network output and errors
        fig, axes = plt.subplots(2,nOutputs +1 , figsize=(10, 6),sharex=True, dpi=400)
        # genreate 2d ndarray that starts at 0 and ends at 2*nOutputs
        axes = axes.ravel()
        for iOutput in range(nOutputs):
            axes[iOutput].plot(time_of_domain, state_true[iOutput,:], label="IVP solution", linewidth=1, color="k", alpha=0.3)
            for iSample in range(0, nSamples):
                axes[iOutput].plot(time_of_domain, state_domain[iSample,...,iOutput].cpu().detach().numpy(),
                                   color=colours[iSample],linewidth=0.5,alpha=0.7)
            axes[iOutput].set_ylabel('State')
            axes[iOutput] = pretty_axis(axes[iOutput], legendFlag=False)
            axes[iOutput].set_ylim([-0.5,1.5])
        # plot the gradient error
        for iOutput in range(nOutputs):
            for iSample in range(0, nSamples):
                if iSample == 0:
                    # give a label
                    axes[nOutputs + iOutput + 1].plot(time_of_domain, rhs_error_state_weights[iOutput] * (
                                dxdt[iSample, ..., iOutput].cpu().detach().numpy() - rhs_pinn[
                            iSample, ..., iOutput].cpu().detach().numpy()),
                                                      linewidth=0.5, color=colours[iSample], alpha=0.7,
                                                      label=f"Error weight: {rhs_error_state_weights[iOutput]}")
                else:
                    # plot without a label
                    axes[nOutputs + iOutput + 1].plot(time_of_domain, rhs_error_state_weights[iOutput] * (
                                dxdt[iSample, ..., iOutput].cpu().detach().numpy() - rhs_pinn[
                            iSample, ..., iOutput].cpu().detach().numpy()),
                                                      linewidth=0.5, color=colours[iSample], alpha=0.7)
            axes[nOutputs+iOutput+1].set_xlabel('Time')
            axes[nOutputs+iOutput+1].set_ylabel('Derivative error')
            axes[nOutputs+iOutput+1] = pretty_axis(axes[nOutputs+iOutput+1], legendFlag=True)
            axes[nOutputs+iOutput+1].set_ylim([-0.2, 0.2])
        # plot the current and current error
        axes[nOutputs].plot(time_of_domain, measured_current, label="Measured current", color="k",  linewidth=1, alpha=0.3)
        # for as many conductances as we put in, plot the current
        for iSample in range(0, nSamples):
            # plot the current
            axes[nOutputs].plot(time_of_domain, current_pinn[iSample,:].cpu().detach().numpy(), color=colours[iSample],linewidth=0.5,alpha=0.7) #label = "PINN current"
            # plot the current error
            axes[-1].plot(time_of_domain, measured_current - current_pinn[iSample,:].cpu().detach().numpy(),
                          color=colours[iSample], linewidth=0.5, alpha=0.7)
        axes[nOutputs].set_ylabel('Current')
        axes[nOutputs] = pretty_axis(axes[nOutputs], legendFlag=False)
        axes[nOutputs].set_ylim([-4, 2])
        # axes[-1].plot(time_of_domain, measured_current - current_pinn.detach().numpy()[0,:], color="k",linewidth=0.5, alpha=0.6)
        axes[-1].set_xlabel('Time')
        axes[-1].set_ylabel('Current error')
        axes[-1] = pretty_axis(axes[-1], legendFlag=False)
        axes[-1].set_ylim([-10, 10])
        fig.tight_layout(pad=0.3, w_pad=0.4, h_pad=0.2)
        if rhs_name.lower() == 'hh_all_inputs_model':
            cbar = fig.colorbar(cond_heatmap, ax=axes.tolist(), location='top', aspect=50) #ticks=levels
            # cbar.ax.set_xticklabels(["{:.2f}".format(j+1) for j in levels])
            cbar.ax.set_ylabel('j')
            cbar.ax.yaxis.label.set_rotation(90)
        # set the suptitle  of the figure
        fig.suptitle(f"i = {i}")
        fig.savefig('Figures/'+rhs_name.lower()+'_NN_approximation_iter_' + str(i) + '.png')
        ################################################################################################################
        fig_costs, axes = plt.subplots(len(all_cost_names) + 1, 1, figsize=(10, 7), sharex=True, dpi=400)
        axes = axes.ravel()
        axes[0].plot(loss_seq)
        axes[0].set_yscale('log')
        axes[0].set_ylabel('Total loss')
        axes[0] = pretty_axis(axes[0], legendFlag=False)
        for iCost, cost_name in enumerate(all_cost_names):
            axes[iCost + 1].plot(stored_costs[cost_name], label=r'$\lambda=$' + '{:.4E}'.format(lambdas[iCost]),
                                 linewidth=1)
            axes[iCost + 1].set_yscale('log')
            axes[iCost + 1].set_ylabel(cost_name)
            axes[iCost + 1] = pretty_axis(axes[iCost + 1], legendFlag=True)
        axes[-1].set_xlabel('Training step')
        fig_costs.tight_layout()
        fig_costs.savefig('Figures/' + rhs_name.lower() + '_costs_iter_' + str(i) + '.png')
        ################################################################################################################
        # # we also want to plot the layers as basis functions
        # fig, axes = plot_layers_as_bases(pinn, domain, domain_scaled)
        # axes[0].set_title(f"i ={i}")
        # fig.tight_layout()
        # # save the figure
        # fig.savefig('Figures/'+rhs_name.lower()+'_layer_outpusts_iter_' + str(i) + '.png', dpi=400)
        # # and parameter values to trace how they are updated
        # fig, axes = plot_pinn_params_all_inputs(pinn)
        # # set the suptitle
        # axes[0].set_ylabel(f"i={i}")
        # plt.subplots_adjust(left=0,right=1,wspace=0.1, hspace=1.3)
        # save the figure
        # fig.savefig('Figures/'+rhs_name.lower()+'_params_iter_' + str(i) + '.png', dpi=400)
        # plt.close('all')
        #  check the convergence of the loss function
        if i > 0:
            derivative_of_cost = np.abs(loss_seq[-1] - loss_seq[-2]) / loss_seq[-1]
            print(derivative_of_cost)
            if derivative_of_cost < 1e-8:
                print('Cost coverged.')
                break
    #  end of plotting condition
# end of training loop

# save the model to a pickle file
pt.save(pinn.state_dict(), 'Models/'+rhs_name.lower()+'_'+str(nLayers)+'_layers_'+str(nHidden)+'_nodes_'+str(nInputs)+'_ins'+str(nOutputs)+'_outs.pth')
# save the costs to a pickle file
with open('Models/'+rhs_name.lower()+'_'+str(nLayers)+'_layers_'+str(nHidden)+'_nodes_'+str(nInputs)+'_ins'+str(nOutputs)+'_costs.pkl', 'wb') as f:
    pkl.dump(stored_costs, f)
########################################################################################################################
# plot the output of the model on the entire time interval
times_scaled = times * t_scaling_coeff
times_all_domain = pt.tensor(times_scaled, dtype=pt.float32)
thetas_true_tensor = pt.tensor(thetas_true).unsqueeze(-1) * param_scaling_coeff
stacked_domain = stack_inputs(times_all_domain, param_sample.detach())
stacked_true = stack_inputs(times_all_domain, thetas_true_tensor)
########################################################################################################################
# send the domain to device
stacked_domain = stacked_domain.to(device)
stacked_true = stacked_true.to(device)
# generate output of the trained PINN and current
pinn_output = pinn(stacked_domain)
pinn_current = observation_tensors(times, pinn_output, stacked_domain)
pinn_current = pinn_current/param_scaling_coeff
pinn_output = pinn_output.cpu().detach().numpy()
pinn_current = pinn_current.cpu().detach().numpy()
# get the true current
pinn_output_at_truth = pinn(stacked_true)
pinn_current_at_truth = observation_tensors(times, pinn_output_at_truth, stacked_true)
pinn_current_at_truth = pinn_current_at_truth/param_scaling_coeff
pinn_output_at_truth = pinn_output_at_truth.cpu().detach().numpy()
pinn_current_at_truth = pinn_current_at_truth.cpu().detach().numpy()
########################################################################################################################
# plot outputs at training points
fig_data, axes = plt.subplots(2+nOutputs, 1, figsize=(10, 7), sharex=True, dpi=400)
axes = axes.ravel()
# plot the solution for all outputs
state_true_all = sol_for_x.sol(times)
for iOutput in range(nOutputs):
    axes[iOutput].plot(times, state_true_all[iOutput,:], label='IVP solution', color='k', alpha=0.3)
    # this part could be wrong because we may have a multi-dim tensor where only the first dimension matches times
    for iSample in range(0, nSamples):
        axes[iOutput].plot(times, pinn_output[iSample,:, iOutput], '--', color=colours[iSample], alpha=0.7, linewidth=0.5)
    # axes[iOutput+1].plot(times, pinn_output[..., iOutput], '--', label='PINN solution')
    # axes[iOutput].set_xlabel('Time')
    axes[iOutput].set_ylabel('State')
    axes[iOutput] = pretty_axis(axes[iOutput], legendFlag=False)
iAxis = nOutputs
# plot current vs PINN current
axes[iAxis].plot(times, current_true, label='True current', color='k', alpha=0.3)
for iSample in range(0, nSamples):
    axes[iAxis].plot(times, pinn_current[iSample,:], '--',color=colours[iSample], alpha=0.7, linewidth=0.5)
# axes[iAxis].plot(times, pinn_current., '--', label='PINN current')
# axes[iAxis].set_xlabel('Time')
axes[iAxis].set_ylabel('Current')
axes[iAxis] = pretty_axis(axes[iAxis], legendFlag=True)
iAxis = nOutputs+1
# plot the voltage
axes[iAxis].plot(times, V(times), color='k', alpha=0.3)
axes[iAxis].set_xlabel('Time')
axes[iAxis].set_ylabel('Input voltage')
axes[iAxis] = pretty_axis(axes[iAxis], legendFlag=False)
fig.tight_layout(pad=0.3, w_pad=0.4, h_pad=0.2)
if rhs_name.lower() == 'hh_all_inputs_model':
    cbar = fig.colorbar(cond_heatmap, ax=axes.tolist(), location='top', aspect=50) # ticks=levels
    # cbar.ax.set_xticklabels(["{:.2f}".format(j+1) for j in levels])
    cbar.ax.set_ylabel('j')
    cbar.ax.yaxis.label.set_rotation(90)
plt.savefig('Figures/'+rhs_name.lower()+'_trained_nn_output_at_training_values.png')
########################################################################################################################
# plot the outputs at the true conductance
fig_data, axes = plt.subplots(2+nOutputs, 1, figsize=(10, 7), sharex=True, dpi=400)
axes = axes.ravel()
for iOutput in range(nOutputs):
    axes[iOutput].plot(times, state_true_all[iOutput,:], label='True state',color='k', alpha=0.3)
    axes[iOutput].plot(times, pinn_output_at_truth[0,..., iOutput], '--', label='PINN solution')
    axes[iOutput].set_ylabel('State')
    axes[iOutput] = pretty_axis(axes[iOutput], legendFlag=True)
iAxis = nOutputs
# plot current vs PINN current
axes[iAxis].plot(times, current_true, label='True current',color='k', alpha=0.3)
axes[iAxis].plot(times, pinn_current_at_truth[0,:], '--', label='PINN current')
axes[iAxis].set_ylabel('Current')
axes[iAxis] = pretty_axis(axes[iAxis], legendFlag=True)
iAxis = nOutputs+1
# plot the voltage
axes[iAxis].plot(times, V(times),color='k', alpha=0.3)
axes[iAxis].set_xlabel('Time')
axes[iAxis].set_ylabel('Input voltage')
axes[iAxis] = pretty_axis(axes[iAxis], legendFlag=False)
plt.tight_layout()
plt.savefig('Figures/'+rhs_name.lower()+'_trained_nn_output_at_truth.png')
########################################################################################################################
# plot all the cost functions and the total cost, all in separate axes with shared xaxis
fig_costs, axes = plt.subplots(len(all_cost_names)+1, 1, figsize=(10, 7), sharex=True, dpi=400)
axes = axes.ravel()
axes[0].plot(loss_seq)
axes[0].set_yscale('log')
axes[0].set_ylabel('Total loss')
axes[0] = pretty_axis(axes[0], legendFlag=False)
for iCost, cost_name in enumerate(all_cost_names):
    axes[iCost+1].plot(stored_costs[cost_name], label=r'$\lambda=$' + '{:.4E}'.format(lambdas[iCost]), linewidth=1)
    axes[iCost+1].set_yscale('log')
    axes[iCost+1].set_ylabel(cost_name)
    axes[iCost+1] = pretty_axis(axes[iCost+1], legendFlag=True)
axes[-1].set_xlabel('Training step')
plt.tight_layout()
plt.savefig('Figures/'+rhs_name.lower()+'_costs.png')