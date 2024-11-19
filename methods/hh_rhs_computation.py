from methods.generate_data import *

def RHS_tensors(t, x, theta):
    """
    This function computes the right hand side of the Hodgkin Huxley ODE
    :param t: time vector for which ODE is solved
    :param x: vector of gating variables a and r
    :param theta: vector of model parameters
    :return:
    dx = [da/dt, dr/dt] - derivatives of gating variables a and r
    """
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

# in the function above, the only thing that will change from epoch to epoch is the state tensor x, so we should be able
# to pre-compute everything up until tau_a, a_inf, tau_r, r_inf, and then just compute the RHS from those tensors
# this will save us some time in the computation
def RHS_tensors_precompute(t, x, theta, device):
    """
    This function procudes a part of the right hand side of the ODE that is independent of the state tensor x.
    We can compute this becasue we know all parameter points for taining the PINN
    Only need to run this once before entering the training loop
    :param t: time vector - has to be numpy to use interpolate from scipy
    :param x: state tensor of size [nSamples x nTimes x nStates]
    :param theta: parameter tensor of size [nSampes x nTimes x nParams]
    :param device: device to run the computations on
    :return: precomputed RHS tensors of size [nSamples x nTimes x nStates]
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

def observation_tensors(t, x, theta, device):
    """
    This function produces the current tensor from the state tensor x and the parameter tensor theta
    :param t: time vector - has to be numpy to use interpolate from scipy
    :param x: state tensor of size [nSamples x nTimes x nStates]
    :param theta: parameter tensor of size [nSampes x nTimes x nParams]
    :param device: device to run the computations on
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
    # EK = -80
    EK_tensor = pt.tensor(EK, dtype=pt.float32).expand(state_shape).to(device)
    # the only input in this the conductance that has been broadcasted to the shape of a single state
    current =  g * a * r * (v - EK_tensor)
    return current

if __name__ == '__main__':
    print('Generating precomputed tensors for the right hand side of the Hodgkin Huxley ODE')