from setup import *
from generate_data import *

# define a simle ODE in 1d

def hh_one_state(t, x, theta):
    *p, g = theta[:5]
    v = V(t)
    k1 = p[0] * np.exp(p[1] * v)
    k2 = p[2] * np.exp(-p[3] * v)
    tau_x = 1 / (k1 + k2)
    x_inf = tau_x * k1
    dx = (x_inf - x) / tau_x
    return [dx]

def loss(x):
    x.requires_grad = True
    outputs = Psi_t(x)
    Psi_t_x = pt.autograd.grad(outputs, x, grad_outputs=pt.ones_like(outputs),
                               create_graph=True)[0]
    return pt.mean((Psi_t_x - f(x, outputs)) ** 2)

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
    x0 = 0.
    t = times
    x_true = true_states[0][times_roi]
    volts = V(times)
    ####################################################################################################################
    # define the neural network
    device = pt.device("cpu")
    # we creat first the neural network that takes two inputs: times and voltage, and outputs the state
    NN = nn.Sequential(nn.Linear(2, 50), nn.Sigmoid(), nn.Linear(50, 1, bias=False))
    # Initial condition
    A = 0.
    # The Psi_t function
    Psi_t = lambda x: A + x * NN(x)







