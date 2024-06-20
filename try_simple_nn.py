import matplotlib.pyplot as plt

from setup import *
from generate_data import *
from tqdm import tqdm

# define a simle ODE in 1d

def hh_one_state(t, x, theta):
    *p, g = theta[:5]
    v = V(t)
    k1 = p[0] * np.exp(p[1] * v)
    k2 = p[2] * np.exp(-p[3] * v)
    tau_x = 1 / (k1 + k2)
    x_inf = tau_x * k1
    dx = (x_inf - x) / tau_x
    return dx


class FCN(nn.Module):
    "Defines a standard fully-connected network in PyTorch"

    def __init__(self, N_INPUT, N_OUTPUT, N_HIDDEN, N_LAYERS):
        super().__init__()
        activation = nn.Sigmoid
        self.fcs = nn.Sequential(*[
            nn.Linear(N_INPUT, N_HIDDEN),
            activation()])
        self.fch = nn.Sequential(*[
            nn.Sequential(*[
                nn.Linear(N_HIDDEN, N_HIDDEN),
                activation()]) for _ in range(N_LAYERS - 1)])
        self.fce = nn.Linear(N_HIDDEN, N_OUTPUT)

    def forward(self, x):
        x = self.fcs(x)
        x = self.fch(x)
        x = self.fce(x)
        return x

def loss(inputs):
    # inputs.requires_grad = True
    time = inputs.detach().numpy()
    outputs = Psi_t(inputs)
    ODE_RHS = pt.tensor(np.array(hh_one_state(time, outputs.detach().numpy(), theta_one_state)))
    state_derivs = pt.autograd.grad(outputs, inputs, grad_outputs=pt.ones_like(outputs),
                               create_graph=True)[0]
    state_dot = state_derivs
    # this is just the average difference between the RHS and the derivative of the NN output
    return pt.mean((state_dot - ODE_RHS) ** 2), state_dot.detach().numpy(), ODE_RHS.detach().numpy()

def loss_gradient_matching(inputs):
    time = inputs.detach().numpy()
    # outputs = pt.tensor(IC) + inputs* pinn(inputs)
    outputs = Psi_t(inputs)
    ODE_RHS = np.array(hh_one_state(time, outputs.detach().numpy(), theta_one_state))
    state_derivs = pt.autograd.grad(outputs, inputs, grad_outputs=pt.ones_like(outputs),
                                    create_graph=True)[0]
    state_dot = state_derivs.detach().numpy()
    d_deriv = (state_dot - ODE_RHS)**2
    integral_quad = pt.tensor(sp.integrate.simpson(y=d_deriv, even='avg', axis=0))
    gradient_match_cost = pt.sum(integral_quad, axis=0).requires_grad_(True)
    return gradient_match_cost, state_dot, ODE_RHS

def closure():
    optimiser.zero_grad()
    l = loss(x)
    l.backward()
    return l
def model_capacity(net):
    """
    Prints the number of parameters and the number of layers in the network
    -> Requires a neural network as input
    """
    number_of_learnable_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    num_layers = len(list(net.parameters()))
    print("\nThe number of layers in the model: %d" % num_layers)
    print("The number of learnable parameters in the model: %d\n" % number_of_learnable_params)

if __name__ == '__main__':
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
    current_test = current_model(times, solution, thetas_true, snr=snr)
    ####################################################################################################################
    # we first only want to fit the PINN to one state - a
    theta_one_state = thetas_true[:4] + [thetas_true[-1]]
    state_0 = 0.
    x_true = true_states[0,:]
    volts = V(times)
    ####################################################################################################################
    # define the neural network
    device = pt.device("cpu")
    # we do not need to explicitly pass the voltage into the network -
    # it does not have to be the function of voltage, as it is already incorportated in the cost
    pinn = FCN(1, 1, 15, 2)
    # print network parameter values
    model_capacity(pinn)
    a = list(pinn.parameters())[0]
    test_that_grad_is_none = list(pinn.parameters())[0].grad
    # print(a)

    # make a solution function that takes the input and returns the output of the NN
    Psi_t = lambda input: IC + input * pinn(input)
    # Initial condition
    IC = 0. # this is the intital condtion of the state that we wish to aproximate with the NN - so for now it is set for a
    # inputs = pt.tensor(np.column_stack((times, volts)), requires_grad=True)
    inputs = pt.tensor(times[:, None], requires_grad=True)
    inputs = inputs.float()

    # try training the NN with the loss function
    # optimiser = pt.optim.LBFGS(pinn.parameters(), lr=1e-3)
    optimiser = pt.optim.Adam(pinn.parameters(), lr=1e-3)

    loss_list = []
    # sol_for_a = sp.integrate.solve_ivp(hh_one_state, [0, times[-1]], y0=[IC], args=[theta_one_state], dense_output=True,
    #                                           method='LSODA', rtol=1e-8, atol=1e-8)
    # a_true = sol_for_a.sol(times)
    for i in tqdm(range(1001)):
        # reset the gradient
        optimiser.zero_grad()
        # Evaluate the loss
        l, x_dot, rhs = loss(inputs)
        # l, x_dot, rhs = loss_gradient_matching(inputs)
        # Add the loss to the list
        loss_list.append(l.item())
        # Evaluate the derivative of the loss with respect to all parameters
        l.backward()
        # Make a step
        optimiser.step()
        #  occasionally plot the output
        if i % 100 == 0:
            # check if parameters have changed
            b = list(pinn.parameters())[0]
            AreTheSame = pt.equal(a, b)
            print("Parameters are the same as before: " + str(AreTheSame))
            a = list(pinn.parameters())[0]
            output = pinn(inputs).detach().numpy()
            plt.figure(figsize=(6, 2.5))
            plt.plot(times, x_true, label="IVP solution", color="tab:grey", alpha=0.6)
            plt.plot(times, output[:, 0], label="PINN solution", color="m")
            plt.title(f"Training step {i}")
            plt.legend()
            plt.savefig('Figures/NN_approximation_iter_'+str(i)+'.png')
            plt.show()
            # plot the gradient error
            plt.figure(figsize=(6, 2.5))
            plt.plot(times, x_dot - rhs, label="Gradient error", color="k", alpha=0.6)
            plt.title(f"Training step {i}")
            plt.legend()
            plt.savefig('Figures/deriv_error_iter_'+str(i)+'.png')
            plt.show()
        #  end of plotting condition
    # end of training loop

    # plot the loss evolution into a new figure and save it to file in Figures directory
    plt.figure()
    plt.plot(loss_list)
    plt.xlabel('Training step')
    plt.ylabel('Loss')
    plt.title('Loss evolution during training')
    # plt.show()
    # save the figure
    plt.savefig('Figures/loss_evolution.png')

    print('pause here')








