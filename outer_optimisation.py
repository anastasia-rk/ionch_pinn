from methods.generate_training_set import *
from methods.plot_figures import *
import methods.pints_classes as pints_classes
from methods.pints_classes import *
import pints
import multiprocessing as mp
from itertools import repeat
import csv
import os
import time as tm

def run_pinn_forward(Thetas_ODE, times, pinn, scaling_coeffs):
    """
    This function runs the PINN forward for a given set of parameters and times
    :param Thetas_ODE: the instance of thetas for the ODE RHS
    :param times: the times at which we want to evaluate the PINN
    :param pinn: the trained PINN model
    :param scaling_coeffs: the scaling coefficient for the parameters and time domain
    :return: a tuple of stacked domain, pinn output and the current at the point
    """
    # make a stacked domain from the theta instance and the times
    t_scaling_coeff, param_scaling_coeff = scaling_coeffs
    thetas_tensor = pt.tensor(Thetas_ODE, dtype=pt.float32).unsqueeze(-1) * param_scaling_coeff
    times_tensor = pt.tensor(times, dtype=pt.float32).unsqueeze(-1) * t_scaling_coeff
    # require grad for the domain
    times_tensor.requires_grad_(True)
    thetas_tensor.requires_grad_(True)
    stacked_domain = stack_inputs(times_tensor, thetas_tensor)
    # get the true current
    pinn_output = pinn(stacked_domain)
    pinn_current_at_point = observation_tensors(times, pinn_output, stacked_domain, "cpu")
    pinn_current_at_point = pinn_current_at_point / param_scaling_coeff
    pinn_output_at_point = pinn_output
    # we only need to detach the current tensor
    pinn_current_at_point = pinn_current_at_point.cpu().detach().numpy()
    # the output is the pinn states and the computed current as numpy arrays
    return stacked_domain, pinn_output_at_point, pinn_current_at_point


def run_ode(Thetas_ODE, times, fitted_model, init_conds):
    """
    This function runs the ODE model for a given set of parameters and times
    :param Thetas_ODE: the instance of thetas for the ODE RHS
    :param times: the times at which we want to evaluate the PINN
    :param fitted_model: the trained ODE model
    :return: the tuple containing the states, derivatives and the right hand side of the ODE
    """
    solution = sp.integrate.solve_ivp(fitted_model, [0, times[-1]], init_conds, args=[Thetas_ODE],
                                                dense_output=True, method='LSODA', rtol=1e-8, atol=1e-8)
    states = solution.sol(times)
    derivs = np.gradient(states, times, axis=1)
    rhs = np.array(fitted_model(times, states, Thetas_ODE))
    return (states, derivs, rhs, solution)


# main
if __name__ == '__main__':
    inLogScale = True
    device = "cpu"
    ncpu = mp.cpu_count()
    ncores = 16
    max_iter_outer = 800
    iter_for_convergence = 20
    convergence_threshold = 1e-6
    # set the random seed
    np.random.seed(42)
    isATest = False
    ####################################################################################################################
    # set the training data
    model_name = 'hh'  # model we use to generate the synthetic data for data cost
    rhs_name = 'hh'  # the misspecified right hand side model to be used in gradient cost
    snr_in_db = 30  # signal to noise ratio in dB for the synthetic data generation
    scaled_domain_size = 10  # size of the domain for the scaled input
    if isATest:
        nSamples = 10
    else:
        nSamples = 500
    # set the folders for figures and pickles
    figureFolder = direcory_names['figures']
    modelFolder = direcory_names['models']
    pickleFolder = direcory_names['pickles']
    # folder to save the figures
    FigFolderName = figureFolder + '/' + model_name.lower() + '_data_cuda'
    # folder to read the trained model from
    ModelFolderName = modelFolder + '/' + model_name.lower() + '_data_cuda'
    # folder to read the training data from
    TrainingSetFolderName = modelFolder + '/' + model_name.lower() + '_data'
    # set parmeter names
    if rhs_name.lower() == 'hh':
        fitted_model = hh_model
        observation_model = observation_hh
        Thetas_ODE = thetas_true = thetas_hh_baseline
        param_names = [f'p_{i}' for i in range(1, len(Thetas_ODE) + 1)]
        state_names = ['a', 'r']
        nThetas = len(Thetas_ODE)
        theta_lower_boundary = [np.log(10e-5), np.log(10e-5), np.log(10e-5), np.log(10e-5), np.log(10e-5),
                                np.log(10e-5),np.log(10e-5), np.log(10e-5), np.log(10e-3)]
        theta_upper_boundary = [np.log(10e3), np.log(0.4), np.log(10e3), np.log(0.4), np.log(10e3), np.log(0.4),
                                np.log(10e3), np.log(0.4), np.log(10)]
        n_outputs = 1
    #######################################################################################################
    # load voltage protocol and get times at which we want to train the pinn
    load_protocols
    # generate the segments with B-spline knots and intialise the betas for splines
    jump_indeces, times_roi, voltage_roi, knots_roi, *_ = generate_knots(times)
    jumps_odd = jump_indeces[0::2]
    jumps_even = jump_indeces[1::2]
    nSegments = len(jump_indeces[:-1])
    # use collocation points as an array to get the training times
    unique_times = np.unique(np.hstack(knots_roi))
    times_finest_resolution = times.copy()
    times = unique_times
    ####################################################################################################################
    training_set_files = [name for name in os.listdir(TrainingSetFolderName) if name.endswith('.npy') and 'train' in name]
    if len(training_set_files) > 0:
        #  load stacked domain and measured current from files
        stacked_domain = pt.tensor(np.load(TrainingSetFolderName + '/stacked_scaled_domain_used_for_training.npy'),
                                   dtype=pt.float32).requires_grad_(True)
        stacked_domain_unscaled = pt.tensor(np.load(TrainingSetFolderName + '/stacked_unscaled_domain_used_for_training.npy'),
                                            dtype=pt.float32).requires_grad_(True)
        IC_stacked_domain = pt.tensor(np.load(TrainingSetFolderName + '/IC_stacked_domain_used_for_training.npy'),
                                      dtype=pt.float32).requires_grad_(True)
        if model_name.lower() == 'hh':
            pinn_state = pt.tensor(np.load(TrainingSetFolderName + '/true_states_used_for_training_hh_only.npy'),
                                   dtype=pt.float32).requires_grad_(True)
        measured_current = np.load(TrainingSetFolderName + '/current_data_used_for_training.npy')
        measured_current_tensor = pt.tensor(measured_current, dtype=pt.float32).requires_grad_(True)
    # derive other necessary values for training
    measured_current = measured_current_tensor[0, :].detach().numpy()
    IC = pt.tensor([0, 1])  # I think for training on Kemp, we have nothing to compare our initial conditions to.
    t_scaling_coeff = scaled_domain_size / unique_times[-1]
    param_scaling_coeff = scaled_domain_size / pt.max(stacked_domain_unscaled)
    scaling_coeffs = (t_scaling_coeff, param_scaling_coeff)
    ####################################################################################################################
    # set up the neural network
    domain_shape = stacked_domain.shape
    nLayers = 4
    nHidden = 500
    nOutputs = 2
    nInputs = domain_shape[-1]
    # define a neural network to train
    pinn = FCN(nInputs, nOutputs, nHidden, nLayers).to(device)
    # give this PINN a name for saving
    pinnName = (rhs_name.lower() + '_' + str(nLayers) + '_layers_' + str(nHidden) + '_nodes_'
                + str(nInputs) + '_ins_' + str(nOutputs) + '_outs')
    ########################################################################################################################
    # storing parameter names for plotting
    all_names = [name for _, (name, _) in enumerate(pinn.named_parameters())]
    # get unique layer names
    first_layer_name = all_names[0].split('.')[0]
    last_layer_name = all_names[-1].split('.')[0]
    hidden_layer_names = [name.split('.')[0] + '.' + name.split('.')[1] for name in all_names[2:-2]]
    # drip elements of layer list that are duplicates but preserve order - done in weird way from stackoverflow!
    layer_names = [first_layer_name] + list(dict.fromkeys(hidden_layer_names)) + [last_layer_name]
    ########################################################################################################################
    # check if we already have pre-trained weights for this pinn configuration in the modelFolder
    previous_training_output = ModelFolderName + '/' + pinnName + '.pth'
    if os.path.exists(previous_training_output):
        # make sure we can load networks across devices
        checkpoint = pt.load(previous_training_output, map_location=device, weights_only=True)
        print('Pre-trained weights found. Loading the network.')
        if 'model_state_dict' in checkpoint.keys():
            pinn.load_state_dict(checkpoint['model_state_dict'])
            # check if checkpoint contains the key optimiser state
            if 'lambdas' in checkpoint.keys():
                lambdas = checkpoint['lambdas']
            if 'loss_names' in checkpoint.keys():
                all_cost_names = checkpoint['loss_names']
        else:
            print('No model state found in the checkpoint. Initalsing fist iteation')
            pinn, lambdas, all_cost_names = initialise_optimisation(pinn)
    else:
        print('No pre-trained weights found. Initialising the network.')
        # initialise the costs
        pinn, lambdas, all_cost_names = initialise_optimisation(pinn)
    ####################################################################################################################
    # set up table for saving results
    # gen_model means generative model - we are denoting which model we are using to generate the data
    resultsFolderName = 'Results_gen_model_' + model_name + '_pinn_solver'
    if not os.path.exists(resultsFolderName):
        os.makedirs(resultsFolderName)
    ####################################################################################################################
    # set up boundaries for thetas and initialise depending on the scale of search
    # set initial values and boundaries
    if inLogScale:
        # theta in log scale
        init_thetas = -5 * np.ones(nThetas)
        sigma0_thetas = 2 * np.ones(nThetas)
        # boundaries_thetas = pints.RectangularBoundaries(theta_lower_boundary, theta_upper_boundary)
        boundaries_thetas_Michael = BoundariesTwoStates()
    else:
        # theta in decimal scale
        init_thetas = 0.001 * np.ones(nThetas)
        sigma0_thetas = 0.0005 * np.ones(nThetas)
        boundaries_thetas = pints.RectangularBoundaries(np.exp(theta_lower_boundary), np.exp(theta_upper_boundary))

    Thetas_ODE = init_thetas.copy()
    ####################################################################################################################
    ## create pints objects for the outer optimisation
    model_pinn = PinnOutput() # - this must be replace by a class that can take the pinn current and just pass it through
    ## create the problem of comparing the modelled current with measured curren
    problem_outer = pints.MultiOutputProblem(model=model_pinn, times=times,values=measured_current)
    ## associate the cost with it
    error_outer = OuterCriterion(problem=problem_outer)
    error_outer_no_model = OuterCriterionNoModel(problem=problem_outer) # test if running without model is faster
    ## create the optimiser
    optimiser_outer = pints.CMAES(x0=init_thetas, sigma0=sigma0_thetas,boundaries=boundaries_thetas_Michael)
    optimiser_outer.set_population_size(min(len(Thetas_ODE) * 7, 30)) # restrict the population size to 30
    ####################################################################################################################
    # take 1: loosely based on ask-tell example from  pints
    ## Run optimisation
    theta_visited = []
    theta_guessed = []
    f_guessed = []
    theta_best = []
    f_outer_best = []
    f_gradient_best = []
    OuterCosts_all = []
    GradientCosts_all = []
    # create a logger file
    csv_file_name = resultsFolderName + '/iterations_both_states.csv'
    theta_names = ['Theta_' + str(i) for i in range(len(Thetas_ODE))] # create names for all thetas
    column_names = ['Iteration', 'Walker'] +  theta_names +  ['Outer Cost']
    ####################################################################################################################
    # open the file to write to
    big_tic = tm.time()
    with open(csv_file_name, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(column_names)
        # run the outer optimisation
        for i in range(max_iter_outer):
            # get the next points (multiple locations)
            thetas = optimiser_outer.ask()
            # deal with the log scale
            if inLogScale:
                # convert thetas to decimal scale for inner optimisation
                thetasForInner = np.exp(thetas)
            else:
                thetasForInner = thetas
            # create the placeholder for cost functions
            OuterCosts = []
            GradientCosts = []
            # for each theta in the sample
            tic = tm.time()
            # run inner optimisation for each theta sample
            # with mp.get_context('fork').Pool(processes=min(ncpu, ncores)) as pool:
            #     results = pool.starmap(run_pinn_forward,zip(thetasForInner, repeat(times), repeat(pinn), repeat(param_scaling_coeff)))
            # # package results is a list of tuples
            # extract the results
            for iSample, theta_sample in enumerate(thetasForInner):
                tic_inner = tm.time()
                result = run_pinn_forward(theta_sample, times, pinn, scaling_coeffs)
                toc = tm.time()
                # print('Time taken for sample: ', toc - tic_inner)
                pinn_input, pinn_output, current_at_sample = result
                # store the data costs
                pints_classes.current_pinn = current_at_sample
                OuterCost = error_outer(thetasForInner[iSample, :])
                # get the derivative and rhs at sample
                dxdt = derivative_multi_input(pinn_output, pinn_input)
                dxdt = dxdt * t_scaling_coeff  # restore to the original scale
                # compute the RHS from the precomputed parameters - note that this is in the original scale too
                rhs_pinn = RHS_tensors(times, pinn_output, thetasForInner[iSample, :])
                # compute the gradient cost
                error_rhs = (rhs_pinn - dxdt) ** 2
                # aplify the error along the second state dimension by multiplying it by 10
                for iState in range(error_rhs.shape[-1]):
                    error_rhs[..., iState] = error_rhs[..., iState]
                # simple trapezoidal rule to compute the integral
                sumerror = error_rhs[:, 1:, ...] + error_rhs[:, :-1, ...]
                dt = pinn_input[:, 1:, 0] - pinn_input[:, :-1, 0]
                # this should only expand it once I think
                for iDim in range(len(sumerror.shape) - len(dt.shape)):
                    dt = dt.unsqueeze(-1)
                dt = dt.expand(sumerror.shape)
                loss_rhs = pt.sum(sumerror * dt / 2)
                GradCost = loss_rhs.detach().numpy()
                # store the costs
                OuterCosts.append(OuterCost)
                GradientCosts.append(GradCost)
            # tell the optimiser about the costs
            optimiser_outer.tell(OuterCosts)
            # store the best point
            index_best = OuterCosts.index(min(OuterCosts))
            theta_best.append(thetas[index_best, :])
            f_outer_best.append(OuterCosts[index_best])
            f_gradient_best.append(GradientCosts[index_best])
            OuterCosts_all.append(OuterCosts)
            GradientCosts_all.append(GradientCosts)
            # store the visited points
            theta_visited.append(thetas)
            # print the results
            print('Iteration: ', i+1)
            print('Best parameters: ', theta_best[-1])
            print('Best objective: ', f_outer_best[-1])
            print('Mean objective: ', np.mean(OuterCosts))
            print('Gradient objective at best sample: ', f_gradient_best[-1])
            print('Time elapsed: ', tm.time() - tic)
            # write the results to a csv file
            for iWalker in range(len(thetas)):
                row = [i, iWalker] + list(thetas[iWalker]) + [OuterCosts[iWalker]]
                writer.writerow(row)
            file.flush()

            # check for convergence
            if (i > iter_for_convergence):
                # check how the cost increment changed over the last 10 iterations
                d_cost = np.diff(f_outer_best[-iter_for_convergence:])
                # if all incrementa are below a threshold break the loop
                if all(d <= convergence_threshold for d in d_cost):
                    print("No changes in" + str(iter_for_convergence) + "iterations. Terminating")
                    break
            ## end convergence check condition
        ## end loop over iterations
    ## close the cvs file into which we were writing the results
    big_toc = tm.time()
    # convert the lists to numpy arrays
    theta_best = np.array(theta_best)
    f_outer_best = np.array(f_outer_best)
    f_gradient_best = np.array(f_gradient_best)
    print('Total time taken: ', big_toc - big_tic)
    print('===========================================================================================================')
    ####################################################################################################################
    ## simulate the optimised model using B-splines
    Thetas_ODE = theta_best[-1]
    if inLogScale:
        # convert thetas to decimal scale for inner optimisation
        Thetas_ODE = np.exp(Thetas_ODE)
    else:
        Thetas_ODE = Thetas_ODE
    ## simulate the model using the best thetas and the ODE model used
    pinn_input, pinn_output, current_at_sample = run_pinn_forward(theta_sample, times, pinn, scaling_coeffs)
    init_conds = pinn_output[0, 0, :].detach().numpy()
    # limit init conds to be between 0 and 1
    init_conds = np.clip(init_conds, 0, 1)  # limit the initial conditions to be between 0 and 1
    # get the model current
    states_optimised_ODE, deriv_optimised_ODE, RHS_optimised_ODE, ode_solution = run_ode(Thetas_ODE, times, fitted_model, init_conds)
    current_optimised_ODE = observation_model(times, ode_solution, Thetas_ODE)
    ####################################################################################################################
    ## create figures so you can populate them with the data
    fig, axes = plt.subplot_mosaic([['a)'], ['b)'], ['c)']], layout='constrained', sharex=True)
    y_labels = ['I'] + state_names
    for _, ax in axes.items():
        for iSegment, SegmentStart in enumerate(jumps_odd[:-1]):
            ax.axvspan(times_finest_resolution[SegmentStart], times_finest_resolution[jumps_even[iSegment]], facecolor='0.2', alpha=0.1)
    axes['a)'].plot(times, measured_current, '-k', label=r'Current true', linewidth=1.5, alpha=0.5)
    # add test to plots - compare the modelled current with the true current
    axes['a)'].plot(times, current_optimised_ODE, '-m',
                    label=r'Current from ODE solution', linewidth=1, alpha=0.7)
    axes['b)'].plot(times, states_optimised_ODE[0, :], '-m',
                    label=r'Fitted ODE solution using PINN',
                    linewidth=1, alpha=0.7)
    axes['c)'].plot(times, states_optimised_ODE[1, :], '-m',
                    label=r'Fitted ODE solution using PINN',
                    linewidth=1, alpha=0.7)
    iAx = 0
    for _, ax in axes.items():
        ax.set_ylabel(y_labels[iAx], fontsize=12)
        ax.set_facecolor('white')
        ax.grid(which='major', color='grey', linestyle='solid', alpha=0.2, linewidth=1)
        ax.legend(fontsize=12, loc='best')
        iAx += 1
    # plt.tight_layout(pad=0.3)
    fig.savefig(resultsFolderName + '/states_model_output.png', dpi=400)
    ####################################################################################################################
    fig1, axes1 = plt.subplot_mosaic([['a)'], ['b)'], ['c)']], layout='constrained', sharex=True)
    y_labels1 = ['I_{true} - I_{model}', 'da - RHS', 'dr - RHS']
    for _, ax in axes1.items():
        for iSegment, SegmentStart in enumerate(jumps_odd[:-1]):
            ax.axvspan(times_finest_resolution[SegmentStart], times_finest_resolution[jumps_even[iSegment]], facecolor='0.2', alpha=0.1)
    axes1['a)'].plot(times, measured_current - current_optimised_ODE, '--k',
                     label=r'Current from ODE solution', linewidth=1, alpha=0.7)
    axes1['b)'].plot(times, deriv_optimised_ODE[0,:] - RHS_optimised_ODE[0,:],
                     '--k', label=r'Gradient matching error using PINN', linewidth=1, alpha=0.7)
    axes1['c)'].plot(times, deriv_optimised_ODE[1,:] - RHS_optimised_ODE[1,:],
                     '--k', label=r'Gradient matching using PINN', linewidth=1, alpha=0.7)
    ## save the figures
    iAx = 0
    for _, ax in axes1.items():
        ax.set_ylabel(y_labels1[iAx], fontsize=12)
        ax.set_facecolor('white')
        ax.grid(which='major', color='grey', linestyle='solid', alpha=0.2, linewidth=1)
        ax.legend(fontsize=12, loc='best')
        iAx += 1
    # plt.tight_layout(pad=0.3)
    fig1.savefig(resultsFolderName + '/errors_model_output.png', dpi=400)
    ####################################################################################################################

    # plot evolution of outer costs
    plt.figure(figsize=(10, 6))
    plt.semilogy()
    plt.xlabel('Iteration')
    plt.ylabel('Outer optimisation cost')
    for iIter in range(len(f_outer_best) - 1):
        plt.scatter(iIter * np.ones(len(OuterCosts_all[iIter])), OuterCosts_all[iIter], c='k', marker='.', alpha=.5,
                    linewidths=0)
    iIter += 1
    plt.scatter(iIter * np.ones(len(OuterCosts_all[iIter])), OuterCosts_all[iIter], c='k', marker='.', alpha=.5,
                linewidths=0, label='Sample cost: H(Theta / C, Y)')
    plt.plot(f_outer_best, '-b', linewidth=1.5,
             label='Best cost:H(Theta_{best} / C, Y) = ' + "{:.5e}".format(f_outer_best[-1]))
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig(resultsFolderName + '/outer_cost_evolution.png', dpi=400)

    # plot evolution of outer costs
    plt.figure(figsize=(10, 6))
    plt.semilogy()
    plt.xlabel('Iteration')
    plt.ylabel('Gradient matching cost')
    for iIter in range(len(f_gradient_best) - 1):
        plt.scatter(iIter * np.ones(len(GradientCosts_all[iIter])), GradientCosts_all[iIter], c='k', marker='.', alpha=.5,
                    linewidths=0)
    iIter += 1
    plt.scatter(iIter * np.ones(len(GradientCosts_all[iIter])), GradientCosts_all[iIter], c='k', marker='.', alpha=.5,
                linewidths=0, label='Sample cost: G_{ODE}(Theta, Y)')
    # plt.plot(range(len(f_gradient_best)), np.ones(len(f_gradient_best)) * GradCost_given_true_theta, '--m', linewidth=2.5, alpha=.5,label='Collocation solution: G_{ODE}( C /  Theta_{true}, Y) = ' + "{:.5e}".format(
    #              GradCost_given_true_theta))
    plt.plot(f_gradient_best, '-b', linewidth=1.5,
             label='Best cost:G_{ODE}(C / Theta, Y) = ' + "{:.5e}".format(f_gradient_best[-1]))
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig(resultsFolderName + '/gradient_cost_evolution.png', dpi=400)


    # plot parameter values after search was done on decimal scale
    if inLogScale:
        fig, axes = plt.subplots(len(Thetas_ODE)-2, 1, figsize=(3 * (len(Thetas_ODE)-2), 16), sharex=True)
        for iAx, ax in enumerate(axes.flatten()):
            for iIter in range(len(theta_best)):
                x_visited_iter = theta_visited[iIter][:, iAx]
                ax.scatter(iIter * np.ones(len(x_visited_iter)), x_visited_iter, c='k', marker='.', alpha=.2, linewidth=0)
            # ax.plot(range(iIter+1),np.ones(iIter+1)*theta_true[iAx], '--m', linewidth=2.5,alpha=.5, label=r"true: log("+param_names[iAx]+") = " +"{:.6f}".format(theta_true[iAx]))
            # ax.plot(theta_guessed[:,iAx],'--r',linewidth=1.5,label=r"guessed: $\theta_{"+str(iAx+1)+"} = $" +"{:.4f}".format(theta_guessed[-1,iAx]))
            ax.plot(theta_best[:, iAx], '-m', linewidth=1.5,
                    label=r"best: log(" + param_names[iAx] + ") = " + "{:.6f}".format(theta_best[-1, iAx]))
            ax.set_ylabel('log(' + param_names[iAx] + ')')
            ax.legend(loc='best')
        plt.tight_layout()
        plt.savefig(resultsFolderName + '/ODE_params_log_scale.png', dpi=400)
        # separate for ICs
        fig, axes = plt.subplots(2, 1, figsize=(3 * 2, 16), sharex=True)
        for iAx, ax in enumerate(axes.flatten()):
            iAx += len(Thetas_ODE) - 2
            for iIter in range(len(theta_best)):
                x_visited_iter = theta_visited[iIter][:, iAx]
                ax.scatter(iIter * np.ones(len(x_visited_iter)), x_visited_iter, c='k', marker='.', alpha=.2,
                           linewidth=0)
            ax.plot(theta_best[:, iAx], '-m', linewidth=1.5,
                    label=r"best: log(" + param_names[iAx] + ") = " + "{:.6f}".format(theta_best[-1, iAx]))
            ax.set_ylabel('log(' + param_names[iAx] + ')')
            ax.legend(loc='best')
        plt.tight_layout()
        plt.savefig(resultsFolderName + '/ICs_log_scale.png', dpi=400)
        # plot parameter values converting from log scale to decimal
        fig, axes = plt.subplots(len(Thetas_ODE) - 2, 1, figsize=(3 * (len(Thetas_ODE) - 2), 16), sharex=True)
        for iAx, ax in enumerate(axes.flatten()):
            for iIter in range(len(theta_best)):
                x_visited_iter = theta_visited[iIter][:, iAx]
                ax.scatter(iIter * np.ones(len(x_visited_iter)), np.exp(x_visited_iter), c='k', marker='.', alpha=.2,
                           linewidth=0)
            ax.plot(np.exp(theta_best[:, iAx]), '-m', linewidth=1.5,
                    label=r"best: " + param_names[iAx] + " = " + "{:.6f}".format(np.exp(theta_best[-1, iAx])))
            ax.set_ylabel('log(' + param_names[iAx] + ')')
            ax.legend(loc='best')
        plt.tight_layout()
        plt.savefig(resultsFolderName + '/ODE_params_decimal.png', dpi=400)
    else:
        fig, axes = plt.subplots(len(Thetas_ODE) - 2, 1, figsize=(3 * (len(Thetas_ODE) - 2), 16), sharex=True)
        for iAx, ax in enumerate(axes.flatten()):
            for iIter in range(len(theta_best)):
                x_visited_iter = theta_visited[iIter][:, iAx]
                ax.scatter(iIter * np.ones(len(x_visited_iter)), x_visited_iter, c='k', marker='.', alpha=.2,
                           linewidth=0)
            ax.plot(theta_best[:, iAx], '-m', linewidth=1.5,
                    label=r"best: " + param_names[iAx] + " = " + "{:.6f}".format(theta_best[-1, iAx]))
            ax.set_ylabel('log(' + param_names[iAx] + ')')
            ax.legend(loc='best')
        plt.tight_layout()
        plt.savefig(resultsFolderName + '/ODE_params_decimal.png', dpi=400)
