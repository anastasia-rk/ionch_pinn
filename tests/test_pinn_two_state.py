from methods.generate_data import *
from methods.plot_figures import *


# this does not work for non-scalar outputs!!!
def grad(outputs, inputs):
    # pt.autograd.grad(t, inputs, grad_outputs=[torch.Tensor([1., 1.])]) for t in outputs.t()]) # something like this?
    return pt.autograd.grad(outputs, inputs, grad_outputs=pt.ones_like(outputs), create_graph=True,allow_unused=True)

def deriv(outputs,inputs):
    list_of_grads = []
    # get the tensor shape of outputs
    outputs_shape = outputs.shape
    # if the second dimenstion is smaller than the first, we need to transpose the outputs
    if outputs_shape[1] < outputs_shape[0]:
        outputs = outputs.t()
    for output in outputs:
        ones = pt.ones_like(output)
        grad = pt.autograd.grad(output, inputs, grad_outputs=ones, create_graph=True)[0]
        list_of_grads.append(grad)
    #  create a tensor from the list of tensor by stacking
    return pt.cat(list_of_grads,dim=1)


def voltage(t):
    return pt.sin(t/2)*pt.atan(pt.pi - t/5)

def RHS(t,x):
    v = voltage(t)
    return (-1+v)*x + 1

def RHS_scipy(t,x):
    v = np.sin(t/2) * np.arctan(np.pi - t/5)
    return (-1+v)*x + 1

def my_RHS(t, x, theta):
    *p, g = theta[:9]
    # extract a and r from the x tensor
    a = x[:, 0].view(-1, 1)
    r = x[:, 1].view(-1, 1)
    # create tensor from voltage of size len(t) x 1
    v = pt.tensor(V(t)).view(-1, 1)
    k1 = p[0] * pt.exp(p[1] * v)
    k2 = p[2] * pt.exp(-p[3] * v)
    k3 = p[4] * pt.exp(p[5] * v)
    k4 = p[6] * pt.exp(-p[7] * v)
    tau_a = 1 / (k1 + k2)
    a_inf = tau_a * k1
    tau_r = 1 / (k3 + k4)
    r_inf = tau_r * k4
    # combine rhs for a and r into one tensor
    dx = pt.cat([(a_inf - a) / tau_a, (r_inf - r) / tau_r], dim=1)
    return dx

def hh_model(t, x, theta):
    *p, g = theta[:9]
    a, r = x[:2]
    # create tensor from voltage of size len(t) x 1
    v = V(t)
    k1 = p[0] * np.exp(p[1] * v)
    k2 = p[2] * np.exp(-p[3] * v)
    k3 = p[4] * np.exp(p[5] * v)
    k4 = p[6] * np.exp(-p[7] * v)
    tau_a = 1 / (k1 + k2)
    a_inf = tau_a * k1
    tau_r = 1 / (k3 + k4)
    r_inf = tau_r * k4
    da = (a_inf - a) / tau_a
    dr = (r_inf - r) / tau_r
    return [da,dr]


if __name__ == '__main__':
    # set the folders for figures and pickles
    figureFolder = direcory_names['figures']
    pickleFolder = direcory_names['pickles']
    modelFolder = direcory_names['models']
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
    # theta_one_state = thetas_true[4:8] + [thetas_true[-1]]
    ####################################################################################################################
    # set up the neural network
    loss_seq = []
    pt.manual_seed(123)
    nLayers = 2
    nHidden = 500
    nOutputs = 2
    nInputs = 1
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
    
    # choose which model we want to train for
    IsArchTest = False # if ture, this will fit the PINN output directly to true state, if false, it will fit to the RHS
    rhs_name = 'hh'
    
    if rhs_name.lower() == 'test':
        IC = 0.
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
        IC = [0, 1]
        end_time = 10
        scaling_coeff = tlim[-1]/10
        nSteps = 2000
        domain = pt.linspace(0, end_time, nSteps).requires_grad_(True).unsqueeze(1)
        domain_scaled = pt.linspace(0, end_time * scaling_coeff, nSteps).requires_grad_(True).unsqueeze(1)
        time_of_domain = domain_scaled.detach().numpy()[:, 0]
        sol_for_x = sp.integrate.solve_ivp(hh_model, [0, domain_scaled[-1]], y0=IC, args=[thetas_true], dense_output=True,
                                              method='LSODA', rtol=1e-8, atol=1e-8)
        IC = pt.tensor(sol_for_x.sol(domain.detach().numpy()[0])) # get the IC from true state
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
    plt.savefig(figureFolder + '/Activators_as_basis.png',dpi=400)
    # plt.show()
    
    # plot the weights and biases of the network to check if everything is plotted correctly
    fig, axes = plot_pinn_params(pinn)
    # set the suptitle
    fig.suptitle('test', fontsize=16)
    plt.subplots_adjust(left=0, right=1, wspace=0.1, hspace=1)
    # save the figure
    fig.savefig(figureFolder + '/Weights_and_biases.png', dpi=400)
    
    # ## uncommet this  to fit to the final BC:
    # IC = pt.tensor(x_true[0,-1])
    test = pt.tensor(x_true.transpose())
    # start the optimisation loop
    maxIter = 2001
    plotEvery = 200
    for i in tqdm(range(maxIter)):
        optimiser.zero_grad()
        # compute the gradient loss
        x_domain = pinn(domain)
        # use custom detivarive function to compute the derivatives of outputs, because grad assumed that the output is a scalar (as in we are only computing the grad of the cost)
        dxdt = deriv(x_domain, domain)
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
            # numpy_x = x_domain.detach().numpy()
            rhs_current = my_RHS(time_of_domain, x_domain, thetas_true)
            # try fitting to the true RHS
            # rhs_current = pt.tensor(my_RHS(domain_scaled.detach().numpy(), x_true, theta_one_state),requires_grad=True)
        if IsArchTest:
            loss = pt.sum((x_domain - test) ** 2)
        else:
            error_rhs = (rhs_current - dxdt)**2
            loss_rhs = pt.sum((error_rhs[1:] + error_rhs[:-1]) * (domain_scaled[1:] - domain_scaled[:-1]) / 2)
            # compute the IC loss
            x_ic = pinn(domain[0]).unsqueeze(1)
            # if we want to look at the other end of the domain
            # x_ic = x_domain[0,0]
            loss_ic = pt.sum((x_ic - IC)**2)
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
        if i % plotEvery == 0:
            x = pinn(domain).detach().numpy()
            fig, axes = plt.subplots(2,nOutputs , figsize=(10, 6),sharex=True, dpi=400)
            axes = axes.ravel()
            for iOutput in range(nOutputs):
                axes[iOutput].plot(time_of_domain, x_true[iOutput,:], label="IVP solution", color="k", alpha=0.6)
                axes[iOutput].plot(time_of_domain, x[:,iOutput], label="PINN solution", color="m", marker='o',markersize=2,linestyle=None)
                axes[iOutput].set_ylabel('State')
                axes[iOutput] = pretty_axis(axes[iOutput], legendFlag=True)
            # plot the gradient error
            for iOutput in range(nOutputs):
                axes[nOutputs+iOutput].plot(time_of_domain, dxdt.detach().numpy()[:,iOutput] - rhs_current.detach().numpy()[:,iOutput], label="Gradient error", color="k", alpha=0.6)
                axes[nOutputs+iOutput].set_xlabel('Time')
                axes[nOutputs+iOutput].set_ylabel('Derivative error')
                axes[nOutputs+iOutput] = pretty_axis(axes[nOutputs+iOutput], legendFlag=True)
            # set the suptitle  of the figure
            fig.suptitle(f"i = {i}")
            if rhs_name == 'test':
                axes[1].set_ylim(-2, 1)
            fig.tight_layout()
            fig.savefig(figureFolder + '/'+rhs_name.lower()+'_NN_approximation_iter_' + str(i) + '.png')
            # if i % 20000 == 0:
            #     plt.show()
            ################################################################################################################
            # we also want to plot the layers as basis functions
            fig, axes = plot_layers_as_bases(pinn, domain, domain_scaled)
            axes[0].set_title(f"i ={i}")
            fig.tight_layout()
            # save the figure
            fig.savefig(figureFolder + '/'+rhs_name.lower()+'_layer_outpusts_iter_' + str(i) + '.png', dpi=400)
            # and parameter values to trace how they are updated
            fig, axes = plot_pinn_params(pinn)
            # set the suptitle
            axes[0].set_ylabel(f"i={i}")
            plt.subplots_adjust(left=0,right=1,wspace=0.1, hspace=1.3)
            # save the figure
            fig.savefig(figureFolder + '/'+rhs_name.lower()+'_params_iter_' + str(i) + '.png', dpi=400)
            plt.close('all')
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
    pt.save(pinn.state_dict(), modelFolder + '/'+rhs_name.lower()+'_'+str(nLayers)+'_layers_'+str(nHidden)+'_nodes_'+str(nInputs)+'_ins'+str(nOutputs)+'_outs.pth')
    
    # plot the output of the model on the entire time interval
    times_scaled = times / scaling_coeff
    times_all_domain = pt.tensor(times / scaling_coeff, dtype=pt.float32).requires_grad_(True).unsqueeze(1)
    pinn_output = pinn(times_all_domain).detach().numpy()
    # plot outputs
    fig_data, axes = plt.subplots(2+nOutputs, 1, figsize=(10, 5), dpi=400)
    axes = axes.ravel()
    # plot the loss
    axes[0].plot(loss_seq)
    axes[0].set_yscale('log')
    axes[0].set_xlabel('Training step')
    axes[0].set_ylabel('Loss')
    axes[0] = pretty_axis(axes[0], legendFlag=False)
    # plot the solution for all outputs
    times_all_domain = pt.tensor(times/scaling_coeff, requires_grad=True)
    x_true_all = sol_for_x.sol(times)
    for iOutput in range(nOutputs):
        axes[iOutput+1].plot(times, x_true_all[iOutput,:], label='IVP solution')
        axes[iOutput+1].plot(times, pinn_output[:, iOutput], '--', label='PINN solution')
        axes[iOutput+1].set_xlabel('Time')
        axes[iOutput+1].set_ylabel('State')
        axes[iOutput+1] = pretty_axis(axes[iOutput+1], legendFlag=True)
    # plot the gradient error
    iAxis = nOutputs+1
    if rhs_name == 'test':
        axes[iAxis].plot(time_of_domain, voltage(domain_scaled).detach().numpy())
    elif rhs_name == 'hh':
        axes[iAxis].plot(times, V(times))
    axes[iAxis].set_xlabel('Time')
    axes[iAxis].set_ylabel('Input disturbance')
    axes[iAxis] = pretty_axis(axes[iAxis], legendFlag=False)
    plt.tight_layout()
    plt.savefig(figureFolder + '/'+rhs_name.lower()+'_training_overview.png')
    plt.show()