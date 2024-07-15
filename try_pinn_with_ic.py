import matplotlib.pyplot as plt
import numpy as np

from setup import *
from generate_data import *
from tqdm import tqdm
from plot_figures import *
class FCN(nn.Module):
    """
    Defines a standard fully-connected network in PyTorch. this is a simple feedforward neural network
    with a tanh activation function.

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

def grad(outputs, inputs):
    return pt.autograd.grad(outputs, inputs, grad_outputs=pt.ones_like(outputs), create_graph=True,allow_unused=True)[0]

def voltage(t):
    return pt.sin(t/2)*pt.atan(pt.pi - t/5)

def RHS(t,x):
    v = voltage(t)
    return (-1+v)*x + 1

def RHS_scipy(t,x):
    v = np.sin(t/2) * np.arctan(np.pi - t/5)
    return (-1+v)*x + 1

def my_RHS(t, x, theta):
    *p, g = theta[:5]
    # create tensor from voltage of size len(t) x 1
    v = pt.tensor(V(t)).view(-1, 1)
    k1 = p[0] * pt.exp(p[1] * v)
    k2 = p[2] * pt.exp(-p[3] * v)
    tau_x = 1 / (k1 + k2)
    x_inf = tau_x * k1
    dx = (x_inf - x) / tau_x
    return dx


# load the protocols
tlim = [0, 14899]
times = np.linspace(*tlim, tlim[-1] - tlim[0], endpoint=False)
load_protocols
# generate the segments with B-spline knots and intialise the betas for splines
jump_indeces, times_roi, voltage_roi, knots_roi, collocation_roi, spline_order = generate_knots(times)
jumps_odd = jump_indeces[0::2]
jumps_even = jump_indeces[1::2]
nSegments = len(jump_indeces[:-1])
print('Inner optimisation is split into ' + str(nSegments) + ' segments based on protocol steps.')
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
theta_one_state = thetas_true[4:8] + [thetas_true[-1]]
####################################################################################################################
# set up the neural network
loss_seq = []
pt.manual_seed(123)
nLayers = 2
nHidden = 500
nOutputs = 1
nInputs = 1
# define a neural network to train
pinn = FCN(nInputs, nOutputs, nHidden, nLayers)
a = list(pinn.named_parameters())[0][1]
# storing parameters for plotting
all_names = [name for _, (name, _) in enumerate(pinn.named_parameters())]
# get unique layer names
first_layer_name = all_names[0].split('.')[0]
last_layer_name = all_names[-1].split('.')[0]
hidden_layer_names = [name.split('.')[0] + '.' + name.split('.')[1]  for name in all_names[2:-2]]
# drip elements of layer list that are duplicates but preserve order - done in weird way from stackoverflow!
layer_names = [first_layer_name] + list(dict.fromkeys(hidden_layer_names)) + [last_layer_name]
####################################################################################################################
# define the optimiser and the loss function
optimiser = pt.optim.Adam(pinn.parameters(),lr=1e-3)
loss = pt.tensor(100) # if we are using a while loop, we need to initialise the loss
i = 0
lambda_ic = 1 # weight on the gradient fitting cost
lambda_rhs = 1 # weight on the right hand side fitting cost
lambda_l1 = 0 # weight on the L1 norm of the parameters

# choose which model we want to train for
IsArchTest = False # if ture, this will fit the PINN output directly to true state, if false, it will fit to the RHS
rhs_name = 'hh'
IC = 0
if rhs_name.lower() == 'test':
    end_time = 10
    scaling_coeff = 10
    nSteps  = 1000
    domain = pt.linspace(0, end_time, nSteps).requires_grad_(True).unsqueeze(1)
    domain_scaled = pt.linspace(0, end_time*scaling_coeff, nSteps).requires_grad_(True).unsqueeze(1)
    time_of_domain = domain_scaled.detach().numpy()[:, 0]
    sol_for_x = sp.integrate.solve_ivp(RHS_scipy, [0, time_of_domain[-1]], y0=[IC], dense_output=True,
                                          method='LSODA', rtol=1e-8, atol=1e-8)
    IC = pt.tensor(0.)
    x_true = sol_for_x.sol(time_of_domain)
elif rhs_name.lower() == 'hh':
    end_time = 10
    scaling_coeff = tlim[-1]/10
    nSteps = 2000
    domain = pt.linspace(0, end_time, nSteps).requires_grad_(True).unsqueeze(1)
    domain_scaled = pt.linspace(0, end_time * scaling_coeff, nSteps).requires_grad_(True).unsqueeze(1)
    time_of_domain = domain_scaled.detach().numpy()[:, 0]
    sol_for_x = sp.integrate.solve_ivp(my_RHS, [0, domain_scaled[-1]], y0=[IC], args=[theta_one_state], dense_output=True,
                                          method='LSODA', rtol=1e-8, atol=1e-8)
    IC = pt.tensor(sol_for_x.sol(domain.detach().numpy()[0])[0]) # get the IC from true state
    x_true = sol_for_x.sol(time_of_domain)

# fig = plt.figure(dpi=400)
# plt.plot(x_true[0,:])
# plt.show()
####################################################################################################################
# plot the activation functions of the network as a function of domain
fig, axes = plot_layers_as_bases(pinn, domain, domain)
axes[-1].set_xlabel('Input domain at initialisation')
plt.tight_layout()
# save the figure
plt.savefig('Figures/Activators_as_basis.png',dpi=400)
# plt.show()

# plot the weights and biases of the network to check if everything is plotted correctly
fig, axes = plot_pinn_params(pinn)
# set the suptitle
fig.suptitle('test', fontsize=16)
plt.subplots_adjust(left=0, right=1, wspace=0.1, hspace=1)
# save the figure
fig.savefig('Figures/Weights_and_biases.png', dpi=400)

# ## uncommet this  to fit to the final BC:
# IC = pt.tensor(x_true[0,-1])
test = pt.tensor(x_true.transpose())
# start the optimisation loop
for i in tqdm(range(250001)):
    optimiser.zero_grad()
    # compute the gradient loss
    x_domain = pinn(domain)
    dxdt = grad(x_domain, domain)
    dxdt = dxdt / scaling_coeff
    # divide all elementsof dxdt by 10 to scale the gradient
    # dxdt = dxdt/10
    # error_rhs = (RHS(domain, x_domain) - dxdt)**2
    if rhs_name.lower() == 'test':
        # dxdt = dxdt/scaling_coeff
        rhs_current = RHS(domain_scaled, x_domain)
        # try fitting to the true RHS
        # rhs_current =  RHS(domain_scaled, pt.tensor(x_true.transpose()))
    elif rhs_name.lower() == 'hh':
        # rhs_current = pt.tensor(my_RHS(domain_scaled.detach().numpy(), x_domain.detach().numpy(), theta_one_state),requires_grad=True)
        rhs_current = my_RHS(time_of_domain, x_domain, theta_one_state)
        # try fitting to the true RHS
        # rhs_current = pt.tensor(my_RHS(domain_scaled.detach().numpy(), x_true, theta_one_state),requires_grad=True)
    if IsArchTest:
        loss = pt.sum((x_domain - test) ** 2)
    else:
        error_rhs = (rhs_current - dxdt)**2
        loss_rhs = pt.sum((error_rhs[1:] + error_rhs[:-1]) * (domain_scaled[1:] - domain_scaled[:-1]) / 2)
        # compute the IC loss
        x_ic = pinn(domain[0])
        # if we want to look at the other end of the domain
        # x_ic = x_domain[0,0]
        loss_ic = (x_ic - IC)**2
        # compute the L1 norm
        par_pinn = list(pinn.parameters())
        L1 = pt.tensor([par_pinn[l].abs().sum() for l in range(len(par_pinn))]).sum()
        # compute the total loss
        loss = lambda_ic*loss_ic + lambda_rhs*loss_rhs + lambda_l1*L1
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
        a = list(pinn.named_parameters())[0][1]
        x = pinn(domain).detach().numpy()
        fig, axes = plt.subplots(2,1 , figsize=(10, 6),sharex=True, dpi=400)
        axes = axes.ravel()
        axes[0].plot(time_of_domain, x_true[0,:], label="IVP solution", color="k", alpha=0.6)
        axes[0].plot(time_of_domain, x, label="PINN solution", color="m", marker='o',markersize=2,linestyle=None)
        axes[0].set_title(f"i={i}")
        axes[0].set_ylabel('State')
        axes[0] = pretty_axis(axes[0], legendFlag=True)
        # plot the gradient error
        axes[1].plot(time_of_domain, dxdt.detach().numpy() - rhs_current.detach().numpy(), label="Gradient error", color="k", alpha=0.6)
        axes[1].set_xlabel('Time')
        axes[1].set_ylabel('Derivative error')
        axes[1] = pretty_axis(axes[1], legendFlag=True)
        if rhs_name == 'test':
            axes[1].set_ylim(-2, 1)
        fig.tight_layout()
        fig.savefig('Figures/'+rhs_name.lower()+'_NN_approximation_iter_' + str(i) + '.png')
        # if i % 20000 == 0:
        #     plt.show()
        ################################################################################################################
        # we also want to plot the layers as basis functions
        fig, axes = plot_layers_as_bases(pinn, domain, domain_scaled)
        axes[0].set_title(f"i ={i}")
        fig.tight_layout()
        # save the figure
        fig.savefig('Figures/'+rhs_name.lower()+'_layer_outpusts_iter_' + str(i) + '.png', dpi=400)
        # and parameter values to trace how they are updated
        fig, axes = plot_pinn_params(pinn)
        # set the suptitle
        axes[0].set_ylabel(f"i={i}")
        plt.subplots_adjust(left=0,right=1,wspace=0.1, hspace=1.3)
        # save the figure
        fig.savefig('Figures/'+rhs_name.lower()+'_params_iter_' + str(i) + '.png', dpi=400)
        plt.close('all')
        #  check the convergence of the loss function
        if i > 0:
            derivative_of_cost = np.abs(loss_seq[-1] - loss_seq[-2]) / loss_seq[-1]
            print(derivative_of_cost)
            if derivative_of_cost < 1e-6:
                print('Cost coverged.')
                break
    #  end of plotting condition
# end of training loop

# save the model to a pickle file
pt.save(pinn.state_dict(), 'Models/'+rhs_name.lower()+'_'+str(nLayers)+'_layers_'+str(nHidden)+'_nodes_'+str(nInputs)+'_ins'+str(nOutputs)+'_outs.pth')

# plot the output of the model on the entire time interval
times_scaled = times / scaling_coeff
times_all_domain = pt.tensor(times / scaling_coeff, dtype=pt.float32).requires_grad_(True).unsqueeze(1)
pinn_output = pinn(times_all_domain).detach().numpy()
# plot outputs
fig_data, axes = plt.subplots(3, 1, figsize=(10, 5), dpi=400)
axes = axes.ravel()
# plot the loss
axes[0].plot(loss_seq)
axes[0].set_yscale('log')
axes[0].set_xlabel('Training step')
axes[0].set_ylabel('Loss')
axes[0] = pretty_axis(axes[0], legendFlag=False)
# plot the solution
times_all_domain = pt.tensor(times/scaling_coeff, requires_grad=True)
x_true_all = sol_for_x.sol(times)
axes[1].plot(times, x_true_all[0,:],label='IVP solution')
axes[1].plot(times, pinn_output,'--',label='PINN solution')
axes[1].set_xlabel('Time')
axes[1].set_ylabel('State')
axes[1] = pretty_axis(axes[1], legendFlag=True)
if rhs_name == 'test':
    axes[2].plot(time_of_domain, voltage(domain_scaled).detach().numpy())
elif rhs_name == 'hh':
    axes[2].plot(times, V(times))
axes[2].set_xlabel('Time')
axes[2].set_ylabel('Input disturbance')
axes[2] = pretty_axis(axes[2], legendFlag=False)
plt.tight_layout()
plt.savefig('Figures/'+rhs_name.lower()+'_training_overview.png')
plt.show()