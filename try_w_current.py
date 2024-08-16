import matplotlib.pyplot as plt
import numpy as np

from setup import *
from generate_data import *
from tqdm import tqdm
from plot_figures import *

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
    v = pt.tensor(V(t))
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
    dx = pt.stack([(a_inf - a) / tau_a, (r_inf - r) / tau_r], dim=len(state_shape))
    return dx

def observation_tensors(t, g, x):
    # this will take the g as the tensor as there is more than one instance of g being passed through the network
    a = x[..., 0]
    r = x[..., 1]
    state_shape = a.shape
    v = pt.tensor(V(t))
    # add as many dimensions to the tensor as there are elements in the shape of the state tensor
    for idim in range(len(state_shape) - 1):
        v = v.unsqueeze(-1)
    #  now we should be able to expand the original tensor along those dimensions
    v = v.expand(state_shape)
    # the only input in this the conductance that has been broadcasted to the shape of a single state
    current =  g * a * r * (v - EK*pt.ones_like(v))
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
####################################################################################################################
# run the hh model - to generate current and two true states
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
# we first only want to fit the PINN to one state - a
# theta_one_state = thetas_true[:4] + [thetas_true[-1]]
# or for state r
# theta_one_state = thetas_true[4:8] + [thetas_true[-1]]
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
pinn = FCN(nInputs, nOutputs, nHidden, nLayers)
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
lambda_ic = 1 # weight on the gradient fitting cost
lambda_rhs = 1 # weight on the right hand side fitting cost
lambda_l1 = 0 # weight on the L1 norm of the parameters
lambda_data = 1e-6 # weight on the data fitting cost
# placeholder for storing the costs
all_cost_names = ['IC', 'RHS', 'L1', 'Data']
stored_costs = {name: [] for name in all_cost_names}
# choose which model we want to train for
IsArchTest = False # if ture, this will fit the PINN output directly to true state, if false, it will fit to the RHS
rhs_name = 'hh'

if rhs_name.lower() == 'test':
    # in this case, if we want to test teh PINN we just use a single conductance value to check dimensions
    IC = [0, 1]
    end_time = 10
    t_scaling_coeff = tlim[-1] / end_time
    g_scaling_coeff = 1
    nSteps = 1500
    nConductances = 1
    t_domain = pt.linspace(0, end_time, nSteps).requires_grad_(True)
    t_domain_scaled = pt.linspace(0, end_time * t_scaling_coeff, nSteps)#.requires_grad_(True)
    g_domain = pt.tensor(thetas_true[-1]).requires_grad_(True)
    input_mesh = pt.meshgrid(t_domain, g_domain)
    stacked_domain = pt.stack(input_mesh, dim=len(input_mesh))
    # get a separate domain to get the IC, in case we are not starting time at 0
    IC_input_mesh = pt.meshgrid(pt.tensor([0.]), g_domain)
    IC_stacked_domain = pt.stack(IC_input_mesh, dim=len(IC_input_mesh))
    time_of_domain = t_domain_scaled.detach().numpy()
    sol_for_x = sp.integrate.solve_ivp(hh_model, [0, t_domain_scaled[-1]], y0=IC, args=[thetas_true], dense_output=True,
                                       method='LSODA', rtol=1e-8, atol=1e-8)
    IC = pt.tensor(sol_for_x.sol(t_domain.detach().numpy()[0]))  # get the IC from true state
    x_true = sol_for_x.sol(time_of_domain)
    measured_current = current_model(time_of_domain, sol_for_x, thetas_true, snr=10)
elif rhs_name.lower() == 'hh':
    IC = [0, 1]
    end_point = 10
    t_scaling_coeff = tlim[-1]/end_point
    g_scaling_coeff = 100/end_point
    nSteps = 1500
    nConductances = 10
    t_domain = pt.linspace(0, end_point, nSteps).requires_grad_(True)
    t_domain_scaled = pt.linspace(0, end_point * t_scaling_coeff, nSteps).requires_grad_(True)
    g_domain = pt.linspace(1e-5, end_point, nConductances).requires_grad_(True)
    g_domain_sclaed = pt.linspace(1e-5*g_scaling_coeff, end_point*g_scaling_coeff, nConductances).requires_grad_(True)
    input_mesh = pt.meshgrid(t_domain, g_domain)
    stacked_domain = pt.stack(input_mesh, dim=len(input_mesh))
    # get a separate domain to get the IC, in case we are not starting time at 0
    IC_input_mesh = pt.meshgrid(pt.tensor([0.]), g_domain)
    IC_stacked_domain = pt.stack(IC_input_mesh, dim=len(IC_input_mesh))
    # get the ODE solution and measured output
    time_of_domain = t_domain_scaled.detach().numpy()
    sol_for_x = sp.integrate.solve_ivp(hh_model, [0, t_domain_scaled[-1]], y0=IC, args=[thetas_true], dense_output=True,
                                          method='LSODA', rtol=1e-8, atol=1e-8)
    IC = pt.tensor(sol_for_x.sol(t_domain.detach().numpy()[0])) # get the IC from true state
    x_true = sol_for_x.sol(time_of_domain)
    measured_current = observation_hh(time_of_domain, sol_for_x, thetas_true, snr=10)
# convert the true state and the measured current into a tensor
measured_current_tensor = pt.tensor(measured_current)
x_true_tensor = pt.tensor(x_true.transpose())
########################################################################################################################
# we need to expand the measured current tensor to match the shape of the current tensor
# pass the domain through the PINN to get the output - it does not matter what the output is, we just want to get the dimensions
x_domain = pinn(stacked_domain)
current_pinn = observation_tensors(time_of_domain, input_mesh[-1], x_domain) # this has to be a 2d tensor spanning times and conductances
current_pinn = current_pinn/g_scaling_coeff
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
fig, axes = plot_pinn_params(pinn)
# set the suptitle
fig.suptitle('test', fontsize=16)
plt.subplots_adjust(left=0, right=1, wspace=0.1, hspace=1)
# save the figure
fig.savefig('Figures/Weights_and_biases.png', dpi=400)

########################################################################################################################
# start the optimisation loop
for i in tqdm(range(400001)):
    optimiser.zero_grad()
    # compute the gradient loss
    x_domain = pinn(stacked_domain)
    # use custom detivarive function to compute the derivatives of outputs, because grad assumed that the output is a scalar
    # (as in we are only computing the grad of the cost)
    dxdt = deriv_multidim(x_domain, stacked_domain)
    dxdt = dxdt / t_scaling_coeff
    if rhs_name.lower() == 'test':
        rhs_pinn = RHS_tensors(time_of_domain, x_domain, thetas_true)
        current_pinn = observation_tensors(time_of_domain, input_mesh[-1], x_domain)
    elif rhs_name.lower() == 'hh':
        rhs_pinn = RHS_tensors(time_of_domain, x_domain, thetas_true)
        # as an option, we can use PINN only for ODE solution and not produce current as an output
        # instead, a model of current measurement can be used directly
        current_pinn = observation_tensors(time_of_domain, input_mesh[-1], x_domain) # this has to be a 2d tensor spanning times and conductances
        current_pinn = current_pinn/g_scaling_coeff # we need to amplify the current to match the order of magnitued of the measured current
    ####################################################################################################################
    # compute the loss function
    if IsArchTest:
        # this is the case where we are testing the architecture of the network
        loss = pt.sum((x_domain[...,:2] - x_true_tensor) ** 2)
    else:
        # this is the case where we are using gradient and data costs
        ################################################################################################################
        # compute the RHS loss
        if lambda_rhs != 0:
            error_rhs = (rhs_pinn - dxdt)**2
            # simple trapezoidal rule to compute the integral
            sumerror = (error_rhs[1:, ...] + error_rhs[:-1, ...])
            loss_rhs = pt.sum( sumerror * dt / 2)
        else:
            loss_rhs = 0
        stored_costs['RHS'].append(loss_rhs.item())
        ################################################################################################################
        # compute the IC loss
        if lambda_ic != 0:
            x_ic = x_domain[0,...] # if we are starting from 0 at the time domain
            # x_ic = pinn(IC_stacked_domain) # or we can put the IC domain through the pinn
            loss_ic = pt.sum((x_ic - IC)**2)
        else:
            loss_ic = pt.tensor(0)
        stored_costs['IC'].append(loss_ic.item())
        ################################################################################################################
        # commpute the data loss w.r.t. the current
        if lambda_data != 0:
            residuals_data = current_pinn - measured_current_tensor
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
        # loss =  loss_rhs
    # end of if statement
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
    if i % 10000 == 0:
        b = list(pinn.named_parameters())[0][1]
        AreTheSame = pt.equal(a, b)
        print("Parameters  in the first layer are the same as before: " + str(AreTheSame))
        a = b
        fig, axes = plt.subplots(2,nOutputs +1 , figsize=(10, 6),sharex=True, dpi=400)
        # genreate 2d ndarray that starts at 0 and ends at 2*nOutputs
        axes = axes.ravel()
        for iOutput in range(nOutputs):
            axes[iOutput].plot(time_of_domain, x_true[iOutput,:], label="IVP solution", linewidth=0.5, color="k", alpha=0.3)
            axes[iOutput].plot(time_of_domain, x_domain[...,iOutput].detach().numpy(), color="m",linewidth=0.5,alpha=0.3)
            axes[iOutput].set_ylabel('State')
            axes[iOutput] = pretty_axis(axes[iOutput], legendFlag=False)
        # plot the gradient error
        for iOutput in range(nOutputs):
            axes[nOutputs+iOutput+1].plot(time_of_domain, dxdt[...,iOutput].detach().numpy() - rhs_pinn[...,iOutput].detach().numpy(),linewidth=0.5, color="k", alpha=0.3)
            axes[nOutputs+iOutput+1].set_xlabel('Time')
            axes[nOutputs+iOutput+1].set_ylabel('Derivative error')
            axes[nOutputs+iOutput+1] = pretty_axis(axes[nOutputs+iOutput+1], legendFlag=False)
        # plot the current and current error
        axes[nOutputs].plot(time_of_domain, measured_current, label="Measured current", color="k",  linewidth=0.5, alpha=0.3)
        # for as many conductances as we put in, plot the current
        for iConductance in range(0, nConductances):
            # plot the current
            axes[nOutputs].plot(time_of_domain, current_pinn[:,iConductance].detach().numpy(), color="m",linewidth=0.5,alpha=0.3) #label = "PINN current"
            # plot the current error
            axes[-1].plot(time_of_domain, measured_current - current_pinn[:,iConductance].detach().numpy(),
                          color="k", linewidth=0.5, alpha=0.2)
        axes[nOutputs].set_ylabel('Current')
        axes[nOutputs] = pretty_axis(axes[nOutputs], legendFlag=False)
        # axes[-1].plot(time_of_domain, measured_current - current_pinn.detach().numpy()[0,:], color="k",linewidth=0.5, alpha=0.6)
        axes[-1].set_xlabel('Time')
        axes[-1].set_ylabel('Current error')
        axes[-1] = pretty_axis(axes[-1], legendFlag=False)
        # set the suptitle  of the figure
        fig.suptitle(f"i = {i}")
        # if rhs_name == 'test':
        #     axes[1].set_ylim(-2, 1)
        fig.tight_layout()
        fig.savefig('Figures/'+rhs_name.lower()+'_NN_approximation_iter_' + str(i) + '.png')
        # if i % 20000 == 0:
        #     plt.show()
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
            if derivative_of_cost < 1e-5:
                print('Cost coverged.')
                break
    #  end of plotting condition
# end of training loop

# save the model to a pickle file
pt.save(pinn.state_dict(), 'Models/'+rhs_name.lower()+'_'+str(nLayers)+'_layers_'+str(nHidden)+'_nodes_'+str(nInputs)+'_ins'+str(nOutputs)+'_outs.pth')
# save the costs to a pickle file
with open('Models/'+rhs_name.lower()+'_'+str(nLayers)+'_layers_'+str(nHidden)+'_nodes_'+str(nInputs)+'_ins'+str(nOutputs)+'_costs.pkl', 'wb') as f:
    pkl.dump(stored_costs, f)

# plot the output of the model on the entire time interval
if rhs_name == 'test':
    times_scaled = times / t_scaling_coeff
    times_all_domain = pt.tensor(times / t_scaling_coeff, dtype=pt.float32).requires_grad_(True)
    g_domain = pt.tensor(thetas_true[-1])#.requires_grad_(True)
    input_mesh = pt.meshgrid(times_all_domain, g_domain)
    stacked_domain = pt.stack(input_mesh, dim=len(input_mesh))
elif rhs_name == 'hh':
    times_scaled = times / t_scaling_coeff
    times_all_domain = pt.tensor(times_scaled, dtype=pt.float32).requires_grad_(True)
    g_domain = pt.linspace(1e-5, end_point, nConductances)#.requires_grad_(True)
    input_mesh = pt.meshgrid(times_all_domain, g_domain)
    stacked_domain = pt.stack(input_mesh, dim=len(input_mesh))
# generate output of the trained PINN and current
pinn_output = pinn(stacked_domain).detach().numpy()
pinn_current = observation_tensors(times, g_domain, pinn_output)
pinn_current = pinn_current.detach().numpy()
# plot outputs
fig_data, axes = plt.subplots(2+nOutputs, 1, figsize=(10, 7), sharex=True, dpi=400)
axes = axes.ravel()
# plot the solution for all outputs
x_true_all = sol_for_x.sol(times)
for iOutput in range(nOutputs):
    axes[iOutput].plot(times, x_true_all[iOutput,:], label='IVP solution')
    # this part could be wrong because we may have a multi-dim tensor where only the first dimension matches times
    for iConductance in range(0, nConductances):
        axes[iOutput].plot(times, pinn_output[:,iConductance, iOutput], '--')
    # axes[iOutput+1].plot(times, pinn_output[..., iOutput], '--', label='PINN solution')
    # axes[iOutput].set_xlabel('Time')
    axes[iOutput].set_ylabel('State')
    axes[iOutput] = pretty_axis(axes[iOutput], legendFlag=False)
# plot the gradient error
iAxis = nOutputs
# plot current vs PINN current
axes[iAxis].plot(times, current_true, label='True current')
for iConductance in range(0, nConductances):
    axes[iAxis].plot(times, pinn_current[:,iConductance], '--')
# axes[iAxis].plot(times, pinn_current., '--', label='PINN current')
# axes[iAxis].set_xlabel('Time')
axes[iAxis].set_ylabel('Current')
axes[iAxis] = pretty_axis(axes[iAxis], legendFlag=True)
iAxis = nOutputs+1
# plot the voltage
axes[iAxis].plot(times, V(times))
axes[iAxis].set_xlabel('Time')
axes[iAxis].set_ylabel('Input voltage')
axes[iAxis] = pretty_axis(axes[iAxis], legendFlag=False)
plt.tight_layout()
plt.savefig('Figures/'+rhs_name.lower()+'_trained_nn_output.png')
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