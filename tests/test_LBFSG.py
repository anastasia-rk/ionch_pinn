from methods.generate_data import *
from methods.hh_rhs_computation import *
from methods.plot_figures import *

def grad(outputs, inputs):
    """
    This is useful for taking derivatives. It is a simple wrapper around the autograd.grad function in PyTorch.
    """
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

def closure():
    optimiser.zero_grad()
    # compute the gradient loss
    x_domain = pinn(domain)
    dxdt = grad(x_domain, domain)
    # divide all elementsof dxdt by 10 to scale the gradient
    dxdt = dxdt/10
    # error_rhs = (RHS(domain, x_domain) - dxdt)**2
    if rhs_name == 'test':
        # rhs_current = RHS(domain_scaled, x_domain)
        # try fitting to the true RHS
        rhs_current =  RHS(domain_scaled, pt.tensor(x_true.transpose()))
    elif rhs_name == 'hh':
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
        # x_ic = pinn(domain[0])
        x_ic = x_domain[-1,0]
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
    return loss

########################################################################################################################
# start the main script
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
    theta_one_state = thetas_true[4:8] + [thetas_true[-1]]
    ####################################################################################################################
    # approximate on a smaller time domain than what I have
    loss_seq = []
    # define a neural network to train
    pinn = FCN(1, 1, 100, 2)
    a = list(pinn.parameters())[0]
    print(pinn.parameters())
    params = pinn.parameters()
    test_that_grad_is_none = list(pinn.parameters())[0].grad
    
    # optimiser = pt.optim.Adam(pinn.parameters(),lr=1e-2)
    optimiser = pt.optim.LBFGS(pinn.parameters(),max_iter=10)
    loss = pt.tensor(100) # if we are using a while loop, we need to initialise the loss
    i = 0
    lambda_ic = 1 # weight on the gradient fitting cost
    lambda_rhs = 1e4 # weight on the right hand side fitting cost
    lambda_l1 = 0 # weight on the L1 norm of the parameters
    
    # choose which model we want to train for
    IsArchTest = True # if ture, this will fit the PINN output directly to true state, if false, it will fit to the RHS
    rhs_name = 'test'
    IC = 0
    if rhs_name == 'test':
        domain = pt.linspace(0, 10, 1000).requires_grad_(True).unsqueeze(1)
        domain_scaled = pt.linspace(0, 100, 1000).requires_grad_(True).unsqueeze(1)
        time_of_domain = domain_scaled.detach().numpy()[:, 0]
        sol_for_x = sp.integrate.solve_ivp(RHS_scipy, [0, time_of_domain[-1]], y0=[IC], dense_output=True,
                                              method='LSODA', rtol=1e-8, atol=1e-8)
        IC = pt.tensor(0.)
    elif rhs_name == 'hh':
        domain = pt.linspace(3300, 6500, 1000).requires_grad_(True).unsqueeze(1)
        domain_scaled = domain
        time_of_domain = domain_scaled.detach().numpy()[:, 0]
        sol_for_x = sp.integrate.solve_ivp(my_RHS, [0, domain[-1]], y0=[IC], args=[theta_one_state], dense_output=True,
                                              method='LSODA', rtol=1e-8, atol=1e-8)
        IC = pt.tensor(sol_for_x.sol(domain.detach().numpy()[0])[0]) # get the IC from true state
    
    x_true = sol_for_x.sol(time_of_domain)
    ## uncommet this  to fit to the final BC:
    IC = pt.tensor(x_true[0,-1])
    test = pt.tensor(x_true.transpose())
    # start the optimisation loop
    maxIter = 1001
    plotEvery = 100
    for i in tqdm(range(maxIter)):
        optimiser.step(closure)
         # occasionally plot the output
        if i % 500 == 0:
            # check if parameters have changed
            b = list(pinn.parameters())[0]
            AreTheSame = pt.equal(a, b)
            print("Parameters  in the first layer are the same as before: " + str(AreTheSame))
            a = list(pinn.parameters())[0]
            x_domain = pinn(domain)
            x = x_domain.detach().numpy()
            dxdt = grad(x_domain, domain)
            if rhs_name == 'test':
                # rhs_current = RHS(domain_scaled, x_domain)
                # try fitting to the true RHS
                rhs_current = RHS(domain_scaled, pt.tensor(x_true.transpose()))
            elif rhs_name == 'hh':
                # rhs_current = pt.tensor(my_RHS(domain_scaled.detach().numpy(), x_domain.detach().numpy(), theta_one_state),requires_grad=True)
                rhs_current = my_RHS(time_of_domain, x_domain, theta_one_state)
                # try fitting to the true RHS
                # rhs_current = pt.tensor(my_RHS(domain_scaled.detach().numpy(), x_true, theta_one_state),requires_grad=True)
            fig, axes = plt.subplots(2,1 , figsize=(10, 6),sharex=True, dpi=400)
            axes = axes.ravel()
            axes[0].plot(time_of_domain, x_true[0,:], label="IVP solution", color="k", alpha=0.6)
            axes[0].plot(time_of_domain, x, label="PINN solution", color="m", marker='o',markersize=2,linestyle=None)
            axes[0].set_title(f"Training step {i}")
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
            fig.savefig(figureFolder + '/'+rhs_name+'_NN_approximation_iter_' + str(i) + '.png')
            if i % plotEvery == 0:
                plt.show()
            #  check the convergence of the loss function
            if i > 0:
                derivative_of_cost = np.abs(loss_seq[-1] - loss_seq[-2]) / loss_seq[-1]
                print(derivative_of_cost)
                if derivative_of_cost < 1e-6:
                    print('Convergence reached')
                    break
        #  end of plotting condition
    # end of training loop
    
    
    fig_data, axes = plt.subplots(3, 1, figsize=(10, 5), dpi=400)
    axes = axes.ravel()
    # plot the loss
    axes[0].plot(loss_seq)
    axes[0].set_yscale('log')
    axes[0].set_xlabel('Training step')
    axes[0].set_ylabel('Loss')
    axes[0] = pretty_axis(axes[0], legendFlag=False)
    # plot the solution
    axes[1].plot(time_of_domain, x_true[0,:],label='IVP solution')
    axes[1].plot(time_of_domain, pinn(domain).detach().numpy(),'--',label='PINN solution')
    axes[1].set_xlabel('Time')
    axes[1].set_ylabel('State')
    axes[1] = pretty_axis(axes[1], legendFlag=True)
    if rhs_name == 'test':
        axes[2].plot(time_of_domain, voltage(domain_scaled).detach().numpy())
    elif rhs_name == 'hh':
        axes[2].plot(time_of_domain, V(time_of_domain))
    axes[2].set_xlabel('Time')
    axes[2].set_ylabel('Input disturbance')
    axes[2] = pretty_axis(axes[2], legendFlag=False)
    plt.tight_layout()
    plt.savefig(figureFolder + '/Example_with_disturbance.png')
    plt.show()