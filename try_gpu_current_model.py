import matplotlib.pyplot as plt
import numpy as np

from preliminaries import *
from generate_data import *
from tqdm import tqdm
from plot_figures import *
from torch.utils.data import DataLoader, TensorDataset

def deriv_multidim(outputs,inputs):
    list_of_grads = []
    # the number of outputs is the last element of the shape of the tensor - this will hold becaus of the way we stack inputs
    nOutputs = outputs.shape[-1]
    # we want to iterate over the outputs, not their shape
    for iOutput in range(nOutputs):
        output = outputs[...,iOutput]
        ones = pt.ones_like(output)
        grad = pt.autograd.grad(output, inputs, grad_outputs=ones, create_graph=True)[0] # this will compute the gradient of the output w.r.t. all inputs!
        # the time is alway the first in the list of inputs, so we need to only store the first element of the last dimension of grad
        grad_wrt_time_only = grad[...,0].unsqueeze(-1) # not sure is we want to unsqueeze this part, but it will essentially keep the number of dimensions the same as outputs
        list_of_grads.append(grad_wrt_time_only)
    #  create a tensor from the list of tensor by stacking
    return pt.cat(list_of_grads,dim=-1)

# this is just a rewriting of the HH model that works with tensors of states
def RHS_tensors(t, x, theta):
    p = theta[:8]
    # extract a and r from the x tensor - this will be the first and second column of the output tensor
    a = x[..., 0]
    r = x[..., 1]
    # get the shape of the state
    state_shape = a.shape
    # in this case, we are assuming that the first dimension of the tensor is the time dimension,
    # and all other dimensions correspond to possible parameter values.
    # So, when we create a voltage tensor, that is the same for all parameter values
    v = pt.tensor(V(t),dtype=pt.float32)
    # add as many dimensions to the tensor as there are elements in the shape of the state tensor
    for idim in range(len(state_shape)-1):
        v = v.unsqueeze(-1)
    #  now we should be able to expand the original tensor along those dimensions
    v = v.expand(state_shape) # note that this does not allocate new memony, but rather creates a view of the original tensor
    # then we work with the tensors as if they were scalars (I think)
    k1 = p[0] * pt.exp(p[1] * v)
    k2 = p[2] * pt.exp(-p[3] * v)
    k3 = p[4] * pt.exp(p[5] * v)
    k4 = p[6] * pt.exp(-p[7] * v)
    # these tensors will have the same shape as the state tensor
    tau_a = 1 / (k1 + k2)
    a_inf = tau_a * k1
    tau_r = 1 / (k3 + k4)
    r_inf = tau_r * k4
    # combine rhs for a and r into one tensor
    a_inf = a_inf.to(device)
    r_inf = r_inf.to(device)
    tau_a = tau_a.to(device)
    tau_r = tau_r.to(device)
    dx = pt.stack([(a_inf - a) / tau_a, (r_inf - r) / tau_r], dim=len(state_shape))
    return dx

# in the function above, the only thing that will change from epoch to epoch is the state tensor x, so we should be able
# to pre-compute everything up until tau_a, a_inf, tau_r, r_inf, and then just compute the RHS from those tensors
# this will save us some time in the computation
def RHS_tensors_precomputed(t, x, theta):
    p = theta[:8]
    # extract a and r from the x tensor - this will be the first and second column of the output tensor
    a_state = x[..., 0] # we only need the stae her to get its shape
    # get the shape of the state
    state_shape = a_state.shape
    # in this case, we are assuming that the first dimension of the tensor is the time dimension,
    # and all other dimensions correspond to possible parameter values.
    # So, when we create a voltage tensor, that is the same for all parameter values
    v = pt.tensor(V(t))
    # add as many dimensions to the tensor as there are elements in the shape of the state tensor
    for idim in range(len(state_shape) - 1):
        v = v.unsqueeze(-1)
    #  now we should be able to expand the original tensor along those dimensions
    v = v.expand(
        state_shape)  # note that this does not allocate new memony, but rather creates a view of the original tensor
    # then we work with the tensors as if they were scalars (I think)
    k1 = p[0] * pt.exp(p[1] * v)
    k2 = p[2] * pt.exp(-p[3] * v)
    k3 = p[4] * pt.exp(p[5] * v)
    k4 = p[6] * pt.exp(-p[7] * v)
    # these tensors will have the same shape as the state tensor
    tau_a = 1 / (k1 + k2)
    a_inf = tau_a * k1
    tau_r = 1 / (k3 + k4)
    r_inf = tau_r * k4
    return tau_a, a_inf, tau_r, r_inf

# then define a model that will take the precomputed tensors and compute the RHS
def RHS_from_precomputed(x, precomputed):
    a = x[..., 0]
    r = x[..., 1]
    tau_a, a_inf, tau_r, r_inf = precomputed
    dx = pt.stack([(a_inf - a) / tau_a, (r_inf - r) / tau_r], dim=len(a.shape))
    return dx


def observation_tensors(t, g, x):
    # this will take the g as the tensor as there is more than one instance of g being passed through the network
    a = x[..., 0]
    r = x[..., 1]
    g = g.to(device)
    state_shape = a.shape
    v = pt.tensor(V(t), dtype=pt.float32)
    # add as many dimensions to the tensor as there are elements in the shape of the state tensor
    for idim in range(len(state_shape) - 1):
        v = v.unsqueeze(-1)
    #  now we should be able to expand the original tensor along those dimensions
    v = v.expand(state_shape)
    v = v.to(device)
    EK_tensor = pt.tensor(EK, dtype=pt.float32).expand(state_shape).to(device)
    # the only input in this the conductance that has been broadcasted to the shape of a single state
    current =  g * a * r * (v - EK_tensor)
    return current

########################################################################################################################
# start the main script
# load the protocols
EK = -80
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
IsArchTest = False # if ture, this will fit the PINN output directly to true state, if false, it will fit to the RHS
InLogScale = False # if true, the conductances will be in log scale
rhs_name = 'hh_current_model'
available_rhs = ['test_current_model', 'hh_current_model']
####################################################################################################################
# set up the domain for the PINN
if rhs_name.lower() == 'test_current_model':
    # in this case, if we want to test teh PINN we just use a single conductance value to check dimensions
    IC = [0, 1]
    end_point = 10
    # t_scaling_coeff = tlim[-1] / end_time
    # nSteps = 1000
    t_scaling_coeff = unique_times[-1] / end_point
    t_domain_unscaled = pt.tensor(unique_times,dtype=pt.float32).requires_grad_(True)
    t_domain = pt.tensor(unique_times/t_scaling_coeff, dtype=pt.float32).requires_grad_(True)
    # t_domain = pt.linspace(0, end_point, nSteps).requires_grad_(True)
    # t_domain_unscaled = pt.linspace(0, end_point * t_scaling_coeff, nSteps)#.requires_grad_(True)
    g_scaling_coeff = 1
    nConductances = 1
    g_domain = pt.tensor(thetas_true[-1]).requires_grad_(True)
    input_mesh = pt.meshgrid(t_domain, g_domain)
    stacked_domain = pt.stack(input_mesh, dim=len(input_mesh))
    # get a separate domain to get the IC, in case we are not starting time at 0
    IC_input_mesh = pt.meshgrid(pt.tensor([0.]), g_domain)
    IC_stacked_domain = pt.stack(IC_input_mesh, dim=len(IC_input_mesh))
    time_of_domain = t_domain_unscaled.detach().numpy()
    sol_for_x = sp.integrate.solve_ivp(hh_model, [0, t_domain_unscaled[-1]], y0=IC, args=[thetas_true], dense_output=True,
                                       method='LSODA', rtol=1e-8, atol=1e-8)
    IC = pt.tensor(sol_for_x.sol(t_domain.detach().numpy()[0]),dtype=pt.float32)  # get the IC from true state
    x_true = sol_for_x.sol(time_of_domain)
    measured_current = current_model(time_of_domain, sol_for_x, thetas_true, snr=10)
elif rhs_name.lower() == 'hh_current_model':
    IC = [0, 1]
    end_point = 10
    # t_scaling_coeff = tlim[-1]/end_point
    # nSteps = 1000
    t_scaling_coeff = unique_times[-1] / end_point
    # t_domain = pt.linspace(0, end_point, nSteps).requires_grad_(True)
    # t_domain_unscaled = pt.linspace(0, end_point * t_scaling_coeff, nSteps).requires_grad_
    t_domain_unscaled = pt.tensor(unique_times,dtype=pt.float32).requires_grad_(True)
    t_domain = pt.tensor(unique_times/t_scaling_coeff, dtype=pt.float32).requires_grad_(True)
    # set the conductance domain
    nConductances = 10
    end_point_log10_g = 2
    start_point_log10_g = -4
    if InLogScale:
        g_domain_logs = pt.linspace(start_point_log10_g, end_point_log10_g, nConductances)#.requires_grad_(True)
        g_domain_unscaled = pt.pow(10, g_domain_logs)
    else:
        g_domain_unscaled = pt.linspace(10**start_point_log10_g, 10**end_point_log10_g, nConductances)
    g_scaling_coeff = 10**end_point_log10_g / end_point
    g_domain = g_domain_unscaled/g_scaling_coeff
    # g_domain_unscaled = pt.linspace(1e-5*g_scaling_coeff, end_point*g_scaling_coeff, nConductances)
    input_mesh = pt.meshgrid(t_domain, g_domain)
    stacked_domain = pt.stack(input_mesh, dim=len(input_mesh))
    # get a separate domain to get the IC, in case we are not starting time at 0
    IC_input_mesh = pt.meshgrid(pt.tensor([0.]), g_domain)
    IC_stacked_domain = pt.stack(IC_input_mesh, dim=len(IC_input_mesh))
    # get the ODE solution and measured output
    time_of_domain = t_domain_unscaled.detach().numpy()
    sol_for_x = sp.integrate.solve_ivp(hh_model, [0, t_domain_unscaled[-1]], y0=IC, args=[thetas_true], dense_output=True,
                                          method='LSODA', rtol=1e-8, atol=1e-8)
    IC = pt.tensor(sol_for_x.sol(t_domain.detach().numpy()[0]), dtype=pt.float32) # get the IC from true state
    x_true = sol_for_x.sol(time_of_domain)
    measured_current = observation_hh(time_of_domain, sol_for_x, thetas_true, snr=10)
# convert the true state and the measured current into a tensor
measured_current_tensor = pt.tensor(measured_current, dtype=pt.float32)
x_true_tensor = pt.tensor(x_true.transpose())
stacked_domain = stacked_domain.to(device)
####################################################################################################################
# set up the neural network
loss_seq = []
pt.manual_seed(123)
nLayers = 3
nHidden = 500
nOutputs = 2
nInputs = 2
marks = [int(i) for i in np.linspace(0, nHidden, 3)]
# define a neural network to train
device = pt.device('mps' if pt.backends.mps.is_available() else 'cuda' if pt.cuda.is_available() else 'cpu')
print(f'Using device: {device}')
pinn = FCN(nInputs, nOutputs, nHidden, nLayers).to(device)
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
loss = pt.tensor(100) # if we are using a while loop, we need to initialise the loss
i = 0
lambda_ic = 1 # 1e-2 # weight on the gradient fitting cost
lambda_rhs = 1 # weight on the right hand side fitting cost
lambda_l1 = 0 # weight on the L1 norm of the parameters
lambda_data = 1e-8 # weight on the data fitting cost
# placeholder for storing the costs
all_cost_names = ['IC', 'RHS', 'L1', 'Data']
stored_costs = {name: [] for name in all_cost_names}
########################################################################################################################
# we need to expand the measured current tensor to match the shape of the current tensor
# pass the domain through the PINN to get the output - it does not matter what the output is, we just want to get the dimensions
x_domain = pinn(stacked_domain)
# for the training purposes, we can pre-compute part of the RHS since it will only depend on the domain that does not change
# this will save us some time in the computation
precomputed_RHS_params = RHS_tensors_precomputed(time_of_domain, x_domain, thetas_true)
current_pinn = observation_tensors(time_of_domain, input_mesh[-1], x_domain) # this has to be a 2d tensor spanning times and conductances
current_pinn = current_pinn*g_scaling_coeff
# now expand the measured current tensor to match the shape of the current tensor that will be produced by the PINN
current_shape = current_pinn.shape
for iDim in range(len(current_shape)-1):
    measured_current_tensor = measured_current_tensor.unsqueeze(-1)
measured_current_tensor = measured_current_tensor.expand(current_shape)
# also want to expans the dt tensor to match the shape of the state tensor
sumerror = (x_domain[1:, ...] + x_domain[:-1, ...]) # we dont care about the values of this, just about the dimension
dt = (input_mesh[0][1:, ...] - input_mesh[0][:-1, ...])/t_scaling_coeff
shape_dt = dt.shape
shape_error = sumerror.shape
# this should only expand it once I think
for iDim in range(len(shape_error)-len(shape_dt)):
    dt = dt.unsqueeze(-1)
dt = dt.expand(shape_error)
####################3##################################################################################################
# send thing to device
# create a tensor dataset
dataset = TensorDataset(stacked_domain, measured_current_tensor)
dataloader = DataLoader(dataset, batch_size=200, shuffle=False)
IC_stacked_domain = IC_stacked_domain.to(device)
IC = IC.to(device)
########################################################################################################################
# prepare generate the color wheel from the plasma heatmap with the number of conductances
colours = plt.cm.PuOr(np.linspace(0, 1, nConductances))
# make a throwaway counntour plot to generate a heatmap of conductances
if rhs_name.lower() == 'hh_current_model':
    fig, ax = plt.subplots(1, 1, figsize=(10, 6), dpi=400)
    Z = [[0,0],[0,0]]
    levels = g_domain_unscaled.detach().numpy()
    cond_heatmap = plt.contourf(Z, levels, cmap='PuOr')
    plt.clf()

## plots to check the network architecture
# plot the activation functions of the network as a function of domain
# fig, axes = plot_layers_as_bases(pinn, t_domain, t_domain)
# axes[-1].set_xlabel('Input domain at initialisation')
# plt.tight_layout()
# # save the figure
# plt.savefig('Figures/Activators_as_basis.png',dpi=400)
# # plt.show()

# plot the weights and biases of the network to check if everything is plotted correctly
fig, axes = plot_pinn_params(pinn)
# set the suptitle
fig.suptitle('test', fontsize=16)
plt.subplots_adjust(left=0, right=1, wspace=0.1, hspace=1)
# save the figure
fig.savefig('Figures/Weights_and_biases.png', dpi=400)

########################################################################################################################
# loop settings
plotEvery = 10000
maxIter = 400001
# start the optimisation loop
for i in tqdm(range(maxIter)):
    for i_batch, (input_batch, output_batch) in enumerate(dataloader):
        input_batch = input_batch.to(device)
        output_batch = output_batch.to(device)
        # get times out of input input_batch
        time_of_batch = input_batch[:,0,0].cpu().detach().numpy()
        time_of_batch =time_of_batch*t_scaling_coeff
        g_of_batch = input_batch[...,-1]
        # zero the gradients
        optimiser.zero_grad()
        # compute the gradient loss
        x_domain = pinn(input_batch)
        # use custom detivarive function to compute the derivatives of outputs, because grad assumed that the output is a scalar
        dxdt = deriv_multidim(x_domain, input_batch)
        dxdt = dxdt / t_scaling_coeff
        rhs_pinn = RHS_tensors(time_of_batch, x_domain, thetas_true)
        current_pinn = observation_tensors(time_of_batch, g_of_batch, x_domain)  # this has to be a 2d tensor spanning times and conductances
        current_pinn = current_pinn*g_scaling_coeff
        ####################################################################################################################
        # compute the loss function
        ################################################################################################################
        # compute the RHS loss
        if lambda_rhs != 0:
            error_rhs = (rhs_pinn - dxdt)**2
            # simple trapezoidal rule to compute the integral
            sumerror = (error_rhs[1:, ...] + error_rhs[:-1, ...])
            dt = pt.tensor(time_of_batch[1:] - time_of_batch[:-1],dtype=pt.float32).to(device)
            shape_dt = dt.shape
            shape_error = sumerror.shape
            # this should only expand it once I think
            for iDim in range(len(shape_error) - len(shape_dt)):
                dt = dt.unsqueeze(-1)
            dt = dt.expand(shape_error)
            loss_rhs = pt.sum( sumerror * dt / 2)
        else:
            loss_rhs = 0
        stored_costs['RHS'].append(loss_rhs.item())
        ################################################################################################################
        # compute the IC loss
        if lambda_ic != 0:
            # x_ic = x_domain[0,...] # if we are starting from 0 at the time domain
            x_ic = pinn(IC_stacked_domain) # or we can put the IC domain through the pinn
            loss_ic = pt.sum((x_ic - IC)**2)
        else:
            loss_ic = pt.tensor(0)
        stored_costs['IC'].append(loss_ic.item())
        ################################################################################################################
        # commpute the data loss w.r.t. the current
        if lambda_data != 0:
            residuals_data = current_pinn - output_batch
            loss_data = pt.sum((residuals_data)**2) # by default, pytorch sum goes over all dimensions
        else:
            loss_data = pt.tensor(0)
        stored_costs['Data'].append(loss_data.item())
        ################################################################################################################
        # compute the L1 norm
        if lambda_l1 != 0:
            par_pinn = list(pinn.parameters())
            L1 = pt.tensor([par_pinn[l].abs().sum() for l in range(len(par_pinn))]).sum()
        else:
            L1 = pt.tensor(0)
        stored_costs['L1'].append(L1.item())
        # compute the total loss
        loss = lambda_ic*loss_ic + lambda_rhs*loss_rhs  + lambda_data*loss_data + lambda_l1*L1
        # store the cost function into the list
        loss_seq.append(loss.item())
        # retain gradient of the cost to print and save it
        loss.retain_grad()
        # the backward pass computes the gradient of the loss with respect to the parameters
        loss.backward(retain_graph=True)
        # save the gradient of the loss
        # store_grad = loss.grad
        optimiser.step()
     # occasionally plot the output
    if i % plotEvery == 0:
        # in order to plot over the whole interval, we need to produce output
        x_domain = pinn(stacked_domain)
        # use custom detivarive function to compute the derivatives of outputs, because grad assumed that the output is a scalar
        dxdt = deriv_multidim(x_domain, stacked_domain)
        dxdt = dxdt / t_scaling_coeff
        rhs_pinn = RHS_tensors(time_of_domain, x_domain, thetas_true)
        current_pinn = observation_tensors(time_of_domain, input_mesh[-1], x_domain)  # this has to be a 2d tensor spanning times and conductances
        current_pinn = current_pinn*g_scaling_coeff
        ################################################################################################################
        fig, axes = plt.subplots(2,nOutputs +1 , figsize=(10, 6),sharex=True, dpi=400)
        # genreate 2d ndarray that starts at 0 and ends at 2*nOutputs
        axes = axes.ravel()
        for iOutput in range(nOutputs):
            axes[iOutput].plot(time_of_domain, x_true[iOutput,:], label="IVP solution", linewidth=1, color="k", alpha=0.3)
            for iConductance in range(0, nConductances):
                axes[iOutput].plot(time_of_domain, x_domain[...,iConductance,iOutput].cpu().detach().numpy(), color=colours[iConductance],linewidth=0.5,alpha=0.7)
            axes[iOutput].set_ylabel('State')
            axes[iOutput] = pretty_axis(axes[iOutput], legendFlag=False)
            axes[iOutput].set_ylim([-1.5,1.5])
        # plot the gradient error
        for iOutput in range(nOutputs):
            for iConductance in range(0, nConductances):
                axes[nOutputs+iOutput+1].plot(time_of_domain, dxdt[...,iConductance,iOutput].cpu().detach().numpy() - rhs_pinn[...,iConductance,iOutput].cpu().detach().numpy(),linewidth=0.5, color=colours[iConductance], alpha=0.7)
            axes[nOutputs+iOutput+1].set_xlabel('Time')
            axes[nOutputs+iOutput+1].set_ylabel('Derivative error')
            axes[nOutputs+iOutput+1] = pretty_axis(axes[nOutputs+iOutput+1], legendFlag=False)
            axes[nOutputs+iOutput+1].set_ylim([-0.2, 0.2])
        # plot the current and current error
        axes[nOutputs].plot(time_of_domain, measured_current, label="Measured current", color="k",  linewidth=1, alpha=0.3)
        # for as many conductances as we put in, plot the current
        for iConductance in range(0, nConductances):
            # plot the current
            axes[nOutputs].plot(time_of_domain, current_pinn[:,iConductance].cpu().detach().numpy(), color=colours[iConductance],linewidth=0.5,alpha=0.7) #label = "PINN current"
            # plot the current error
            axes[-1].plot(time_of_domain, measured_current - current_pinn[:,iConductance].cpu().detach().numpy(),
                          color=colours[iConductance], linewidth=0.5, alpha=0.7)
        axes[nOutputs].set_ylabel('Current')
        axes[nOutputs] = pretty_axis(axes[nOutputs], legendFlag=False)
        axes[nOutputs].set_ylim([-4, 2])
        # axes[-1].plot(time_of_domain, measured_current - current_pinn.detach().numpy()[0,:], color="k",linewidth=0.5, alpha=0.6)
        axes[-1].set_xlabel('Time')
        axes[-1].set_ylabel('Current error')
        axes[-1] = pretty_axis(axes[-1], legendFlag=False)
        axes[-1].set_ylim([-10, 10])
        fig.tight_layout(pad=0.3, w_pad=0.4, h_pad=0.2)
        if rhs_name.lower() == 'hh_current_model':
            cbar = fig.colorbar(cond_heatmap, ax=axes.tolist(), ticks=levels, location='top', aspect=50)
            cbar.ax.set_xticklabels(["{:.2f}".format(np.log(i)) for i in levels])
            cbar.ax.set_ylabel('log(g)')
            cbar.ax.yaxis.label.set_rotation(90)
        # set the suptitle  of the figure
        fig.suptitle(f"i = {i}")
        fig.savefig('Figures/'+rhs_name.lower()+'_NN_approximation_iter_' + str(i) + '.png')
        ################################################################################################################
        # # we also want to plot the layers as basis functions
        # fig, axes = plot_layers_as_bases(pinn, domain, domain_scaled)
        # axes[0].set_title(f"i ={i}")
        # fig.tight_layout()
        # # save the figure
        # fig.savefig('Figures/'+rhs_name.lower()+'_layer_outpusts_iter_' + str(i) + '.png', dpi=400)
        # # and parameter values to trace how they are updated
        # fig, axes = plot_pinn_params(pinn)
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
if rhs_name == 'test_current_model':
    times_scaled = times / t_scaling_coeff
    times_all_domain = pt.tensor(times / t_scaling_coeff, dtype=pt.float32).requires_grad_(True)
    g_domain = pt.tensor(thetas_true[-1])#.requires_grad_(True)
    input_mesh = pt.meshgrid(times_all_domain, g_domain)
    stacked_domain = pt.stack(input_mesh, dim=len(input_mesh))
    # get the solution for the entire domain and true conductance
    g_true = pt.tensor(thetas_true[-1]) / g_scaling_coeff
    input_true = pt.meshgrid(times_all_domain, g_true)
    stacked_true = pt.stack(input_true, dim=len(input_true))
elif rhs_name == 'hh_current_model':
    times_scaled = times / t_scaling_coeff
    times_all_domain = pt.tensor(times_scaled, dtype=pt.float32).requires_grad_(True)
    # g_domain = pt.linspace(1e-5, end_point, nConductances)#.requires_grad_(True)
    g_true = pt.tensor(thetas_true[-1])/g_scaling_coeff # need to make sure it is scaled as the conductances on which the PINN was trained
    # g_domain basically have to stay the same as before
    input_mesh = pt.meshgrid(times_all_domain, g_domain)
    stacked_domain = pt.stack(input_mesh, dim=len(input_mesh))
    # get the solution for the entire domain and true conductance
    input_true = pt.meshgrid(times_all_domain, g_true)
    stacked_true = pt.stack(input_true, dim=len(input_true))
########################################################################################################################
# send the domain to device
stacked_domain = stacked_domain.to(device)
stacked_true = stacked_true.to(device)
# generate output of the trained PINN and current
pinn_output = pinn(stacked_domain)
pinn_current = observation_tensors(times, g_domain, pinn_output)
pinn_output = pinn_output.cpu().detach().numpy()
pinn_current = pinn_current.cpu().detach().numpy()
pinn_current = pinn_current*g_scaling_coeff
# get the true current
pinn_output_at_truth = pinn(stacked_true)
pinn_current_at_truth = observation_tensors(times, g_true, pinn_output_at_truth)
pinn_output_at_truth = pinn_output_at_truth.cpu().detach().numpy()
pinn_current_at_truth = pinn_current_at_truth.cpu().detach().numpy()
pinn_current_at_truth = pinn_current_at_truth*g_scaling_coeff
########################################################################################################################
# plot outputs at training conductances
fig_data, axes = plt.subplots(2+nOutputs, 1, figsize=(10, 7), sharex=True, dpi=400)
axes = axes.ravel()
# plot the solution for all outputs
x_true_all = sol_for_x.sol(times)
for iOutput in range(nOutputs):
    axes[iOutput].plot(times, x_true_all[iOutput,:], label='IVP solution', color='k', alpha=0.3)
    # this part could be wrong because we may have a multi-dim tensor where only the first dimension matches times
    for iConductance in range(0, nConductances):
        axes[iOutput].plot(times, pinn_output[:,iConductance, iOutput], '--', color=colours[iConductance], alpha=0.7, linewidth=0.5)
    # axes[iOutput+1].plot(times, pinn_output[..., iOutput], '--', label='PINN solution')
    # axes[iOutput].set_xlabel('Time')
    axes[iOutput].set_ylabel('State')
    axes[iOutput] = pretty_axis(axes[iOutput], legendFlag=False)
iAxis = nOutputs
# plot current vs PINN current
axes[iAxis].plot(times, current_true, label='True current', color='k', alpha=0.3)
for iConductance in range(0, nConductances):
    axes[iAxis].plot(times, pinn_current[:,iConductance], '--',color=colours[iConductance], alpha=0.7, linewidth=0.5)
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
if rhs_name.lower() == 'hh_current_model':
    fig.tight_layout(pad=0.3, w_pad=0.4, h_pad=0.2)
    cbar = fig.colorbar(cond_heatmap, ax=axes.tolist(), ticks=levels, location='top', aspect=50)
    cbar.ax.set_xticklabels(["{:.2f}".format(np.log(i)) for i in levels])
    cbar.ax.set_ylabel('log(g)')
    cbar.ax.yaxis.label.set_rotation(90)
else:
    fig.tight_layout()
plt.savefig('Figures/'+rhs_name.lower()+'_trained_nn_output_at_training_values.png')
########################################################################################################################
# plot the outputs at the true conductance
fig_data, axes = plt.subplots(2+nOutputs, 1, figsize=(10, 7), sharex=True, dpi=400)
axes = axes.ravel()
for iOutput in range(nOutputs):
    axes[iOutput].plot(times, x_true_all[iOutput,:], label='True state',color='k', alpha=0.3)
    axes[iOutput].plot(times, pinn_output_at_truth[..., iOutput], '--', label='PINN solution')
    axes[iOutput].set_ylabel('State')
    axes[iOutput] = pretty_axis(axes[iOutput], legendFlag=True)
iAxis = nOutputs
# plot current vs PINN current
axes[iAxis].plot(times, current_true, label='True current',color='k', alpha=0.3)
axes[iAxis].plot(times, pinn_current_at_truth, '--', label='PINN current')
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
lambdas = [lambda_ic, lambda_rhs, lambda_l1, lambda_data]
fig_costs, axes = plt.subplots(len(all_cost_names)+1, 1, figsize=(10, 7), sharex=True, dpi=400)
axes = axes.ravel()
axes[0].plot(loss_seq)
axes[0].set_yscale('log')
axes[0].set_ylabel('Total loss')
axes[0] = pretty_axis(axes[0], legendFlag=False)
for iCost, cost_name in enumerate(all_cost_names):
    axes[iCost+1].plot(stored_costs[cost_name], label=r'$\lambda=$' + '{:.4E}'.format(lambdas[iCost]))
    axes[iCost+1].set_yscale('log')
    axes[iCost+1].set_ylabel(cost_name)
    axes[iCost+1] = pretty_axis(axes[iCost+1], legendFlag=True)
axes[-1].set_xlabel('Training step')
plt.tight_layout()
plt.savefig('Figures/'+rhs_name.lower()+'_costs.png')