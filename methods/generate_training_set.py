from methods.generate_data import *
from methods.nn_classes import *
from methods.hh_rhs_computation import *
import math

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
    :return:
    randomly generated paramer values, a tensor of size nParams x nSamples
    """
    # check if nParams matches the length of the lower and upper bounds
    if len(lower) != nParams or len(upper) != nParams:
        raise ValueError('The length of the lower and upper bounds should match the number of parameters.')
    ###############################################################################################################
    # generate the tensor
    mean = -3.5
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
    :param time_input: tensor of time points
    :param generated_param_samples: tensor of parameter samples
    :return:
    stacked_input: tensor of size nSamples x nTimes x nInputs
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

def generate_HH_training_set_to_files(unique_times, nSamples, model_name, snr_db=20, scaled_domain_size=10, IsArchTest = False, figureFolder=direcory_names['figures'], modelFolder=direcory_names['models']):
    """
    This function generates a training set for training the PINN on Hodgkin-Huxley structute and saves it to files
    It will also produce the plots of parameter samples used for training and their histograms
    :param unique_times: unique time points for the training set
    :param nSamples: number of samples to generate
    :param model_name: name of the model to generate the data from (default is 'hh')
    :param snr_db: signal to noise ratio in decibels
    :param scaled_domain_size: the maximum value of the domain - for scaling the inputs (default is 10)
    :param IsArchTest: PINN architecture test flag. if True, the function will only generate a single sample at the true HH parameters
    :param figureFolder: folder to save the figures
    :param modelFolder: folder to save the model data
    :return:
    stacked_domain - tensor of size [nSamples x nTimes x nInputs] - the input tensor for the PINN
    stacked_domain_unscaled - tensor of size [nSamples x nTimes x nInputs] - the input tensor for the PINN in the original scale
    measured_current_tensor - tensor of size [nSamples x nTimes x 1] - the measured current tensor
    pinn_state - tensor of size [nSamples x nTimes x nOutputs] - the state tensor produced by the PINN
    """
    ## make sure the folders for model and figures exist
    FigFolderName = figureFolder + '/' + model_name.lower() + '_data'
    if not os.path.exists(FigFolderName):
        os.makedirs(FigFolderName)
    # create the folder for data storage
    ModelFolderName = modelFolder
    if not os.path.exists(ModelFolderName):
        os.makedirs(ModelFolderName)
    ####################################################################################################################
    # generate the data for training
    if model_name.lower() not in available_models:
        raise ValueError(f'Unknown model name: {model_name}. Available models are: {available_models}.')
    elif model_name.lower() == 'hh':
        thetas_true = thetas_hh_baseline
        nSamples -= 1 # we will add the true parameters to the sample
    elif model_name.lower() == 'kemp':
        thetas_true = thetas_kemp
    elif model_name.lower() == 'wang':
        thetas_true = thetas_wang
    solution, current_model = generate_synthetic_data(model_name, thetas_true, times)
    true_states = solution.sol(times)
    # set signal to noise ratio in decibels
    snr = 10 ** (snr_db / 10)
    ####################################################################################################################
    # set up the doamin for PINN training
    t_scaling_coeff = scaled_domain_size / unique_times[-1]
    t_domain_unscaled = pt.tensor(unique_times, dtype=pt.float32).requires_grad_(True)
    t_domain = t_domain_unscaled * t_scaling_coeff
    if IsArchTest:
        # if we are testing architecture - just generate a single sample at the true HH parameters
        nSamples = 1
        # in this case, we just want to generate the sample of of true parameters
        param_sample_unscaled = pt.tensor(thetas_true).unsqueeze(-1).requires_grad_(True)
        # colour for plottin network output
        colours = ['darkorange']  # for the test, we only have one instance
    else:
        # for the normal use, generate sample of size nSamples for all HH parameters within given bounds
        min_val = pt.tensor([1e-7, 1e-7, 1e-7, 1e-7, 1e-7, 1e-7, 1e-7, 1e-7, 1e-5])
        max_val = pt.tensor([1e3, 0.4, 1e3, 0.4, 1e3, 0.4, 1e3, 0.4, 10])
        param_sample_unscaled = generate_parameter_sample(nSamples, len(max_val), min_val, max_val, rateConstraint=True)
        # I want to add the true data point to the sample to see if the PINN is able to fit that even when it struggles with others
        true_params_unscaled = pt.tensor(thetas_true).unsqueeze(-1)
        if model_name.lower() == 'hh':
            # add the true params to the sample
            param_sample_unscaled = pt.cat([param_sample_unscaled, true_params_unscaled], dim=-1)
        nSamples = param_sample_unscaled.shape[-1]
        # if nSamples % nPerBatch != 0:
        #     raise ValueError('The number of samples should be divisible by the number of samples per batch.')
        # plot histograms of smapled values for each parameter
        param_names = ['p1', 'p2', 'p3', 'p4', 'p5', 'p6', 'p7', 'p8', 'g'] # always 9 parameters for the HH model
        # set up the colour wheel for plotting output at different training samples - this will be useful for plotting
        colours = plt.cm.PuOr(np.linspace(0, 1, nSamples))
        # make a throwaway counntour plot to generate a heatmap of conductances
        fig, ax = plt.subplots(1, 1, figsize=(10, 6), dpi=400)
        Z = [[0, 0], [0, 0]]
        levels = np.linspace(0, nSamples - 1,
                             nSamples)  # in this case we want to iterate over samples rather than values of parameters
        cond_heatmap = plt.contourf(Z, levels, cmap='PuOr')
        plt.clf()
        fig, axs = plt.subplots(3, 3, figsize=(15, 15))
        axs = axs.ravel()
        for i in range(param_sample_unscaled.shape[0]):
            axs[i].hist(param_sample_unscaled[i, :].detach().numpy(), density=True, bins=20, alpha=0.3,
                        color='darkorange')
            # axs[i].set_title(f'Parameter {i + 1} Histogram')
            axs[i].set_xlabel(param_names[i])
            axs[i] = pretty_axis(axs[i], legendFlag=False)
            axs[i].set_xscale('log')
            # axs[i].set_ylabel('Frequency')
        # save the figure
        fig.tight_layout()
        fig.savefig(FigFolderName + '/hh_parameter_histograms.png', dpi=400)
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
            axs[i].axvline(min_val[2 * i].detach().numpy(), linestyle='--', alpha=0.7, color='grey')
            axs[i].axvline(max_val[2 * i].detach().numpy(), linestyle='--', alpha=0.7, color='grey')
            axs[i].axhline(min_val[2 * i + 1].detach().numpy(), linestyle='--', alpha=0.7, color='grey')
            axs[i].axhline(max_val[2 * i + 1].detach().numpy(), linestyle='--', alpha=0.7, color='grey', label='bounds')
            # add minimal and maximal rate lines
            p1_values = pt.logspace(pt.log(min_val[2 * i]), pt.log(max_val[2 * i]), base=math.e, steps=100)
            p2_values_min = (pt.log(rr_min) - pt.log(p1_values)) / (sign_per_rate[i] * V_bound_per_rate[i])
            p2_values_min = pt.max(p2_values_min, min_val[2 * i + 1] * pt.ones_like(p2_values_min))
            p2_values_max = (pt.log(rr_max) - pt.log(p1_values)) / (sign_per_rate[i] * V_bound_per_rate[i])
            axs[i].plot(p1_values.detach().numpy(), p2_values_min.detach().numpy(), linestyle='--', alpha=0.7,
                        color='black', label=r'$r_{min}$')
            axs[i].plot(p1_values.detach().numpy(), p2_values_max.detach().numpy(), linestyle='-', alpha=0.7,
                        color='black', label=r'$r_{max}$')
            if model_name.lower() == 'hh':
                axs[i].axvline(thetas_true[2 * i], linestyle='--', alpha=0.3, color='magenta')
                axs[i].axhline(thetas_true[2 * i + 1], linestyle='--', alpha=0.3, color='magenta', label='truth')
            axs[i].scatter(param_sample_unscaled[2 * i, :].detach().numpy(),
                           param_sample_unscaled[2 * i + 1, :].detach().numpy(), c=colours)
            axs[i].set_xlabel(param_names[2 * i])
            axs[i].set_ylabel(param_names[2 * i + 1])
            axs[i].set_xscale('log')
            axs[i].set_yscale('log')
            axs[i] = pretty_axis(axs[i], legendFlag=False)
        axs[-1].legend(loc='lower right')
        fig.tight_layout()  # , w_pad=0.3, h_pad=0.3)
        cbar = fig.colorbar(cond_heatmap, ax=axs.tolist(), location='right', aspect=50)  # ticks=levels
        # cbar.ax.set_yticklabels(["{:.2f}".format(j + 1) for j in levels])
        cbar.ax.set_xlabel('j')
        fig.savefig(FigFolderName + '/hh_parameter_samples.png', dpi=400)
        # plot the conductances
        fig, ax = plt.subplots(1, 1, figsize=(5.5, 5), dpi=400)
        ax.axhline(min_val[-1].detach().numpy(), linestyle='--', alpha=0.7, color='grey')
        ax.axhline(max_val[-1].detach().numpy(), linestyle='--', alpha=0.7, color='grey', label='bounds')
        if model_name.lower() == 'hh':
            ax.axhline(thetas_true[-1], linestyle='--', alpha=0.3, color='magenta', label='truth')
        ax.scatter(np.arange(0, nSamples), param_sample_unscaled[-1, :].detach().numpy(), c=colours)
        ax = pretty_axis(ax, legendFlag=True)
        ax.legend(loc='lower left')
        cbar = fig.colorbar(cond_heatmap, ax=ax, location='right', aspect=50)  # ticks=levels
        # cbar.ax.set_yticklabels(["{:.2f}".format(j + 1) for j in levels])
        # cbar.ax.set_xlabel('j')
        # cbar.ax.yaxis.label.set_rotation(90)
        ax.set_xlabel('Sample index')
        ax.set_yscale('log')
        ax.set_ylabel('Conductance')
        fig.tight_layout()
        fig.savefig(FigFolderName + '/hh_conductance_samples.png', dpi=400)
    ####################################################################################################################
    # stack inputs to be fed into PINN
    # scale the parameter so that the largest value there does not exceed scaled_domain_size - all parameters are above 0
    param_scaling_coeff = scaled_domain_size / pt.max(param_sample_unscaled)
    param_sample = param_sample_unscaled * param_scaling_coeff
    stacked_domain_unscaled = stack_inputs(t_domain_unscaled, param_sample_unscaled)
    stacked_domain = stack_inputs(t_domain, param_sample)
    # get a separate domain to get the IC, in case we are not starting time at 0
    IC_t_domain = pt.tensor([unique_times[0]], dtype=pt.float32)
    IC_stacked_domain = stack_inputs(IC_t_domain, param_sample) # get the IC from true state
    state_true = solution.sol(unique_times)
    measured_current = current_model(unique_times, solution, thetas_true, snr=snr)
    measured_current_tensor = pt.tensor(measured_current, dtype=pt.float32)
    ####################################################################################################################
    # set up the neural network - here we only need it to extend the measured current to the dimensions of the input tensors
    # so the internal structure of the network does not matter
    domain_shape = stacked_domain.shape
    pt.manual_seed(123)
    nLayers = 1 # in addition to the first linear layer
    nHidden = 500
    nOutputs = 2 # always two states for the HH model
    nInputs = domain_shape[-1]
    # define a neural network to train
    pinn = FCN(nInputs, nOutputs, nHidden, nLayers).to(device)
    ########################################################################################################################
    # we need to expand the measured current tensor to match the shape of the pinn current tensor
    # pass the domain through the PINN to get the output - it does not matter what the output is, we just want to get the dimensions
    stacked_domain = stacked_domain.to(device)
    state_on_domain = pinn(stacked_domain)
    # for the training purposes, we can pre-compute part of the RHS since it will only depend on the domain that does not change
    # this will save us some time in the computation - note that this is in the original scale
    current_pinn = observation_tensors(unique_times, state_on_domain, stacked_domain_unscaled, device)
    # we only use conductance out of all of stacked domain, and it can be unscaled since we need to compare it to the true current anyway
    # now expand the measured current tensor to match the shape of the current tensor that will be produced by the PINN
    current_shape = current_pinn.shape
    for iDim in range(len(current_shape) - 1):
        measured_current_tensor = measured_current_tensor.unsqueeze(0)
    measured_current_tensor = measured_current_tensor.expand(current_shape)
    if model_name.lower() == 'hh':
        # get the true state expanded to the shape of the PINN output
        true_states_tensor = pt.tensor(state_true.transpose(),dtype=pt.float32)
        true_state_shape = true_states_tensor.shape
        for iDim in range(len(true_state_shape) - 1):
            true_states_tensor = true_states_tensor.unsqueeze(0)
        true_states_tensor = true_states_tensor.expand(state_on_domain.shape)
    ######################################################################################################################
    # save all tensors used for training into files
    # create a pandas dataframe where each column contains a true parameter value from thetas_true
    column_names = [f'p{i + 1}' for i in range(len(thetas_true) - 1)] + ['g']
    true_params_df = pd.DataFrame([thetas_true],columns=column_names)
    true_params_df['Generative model'] = model_name
    true_params_df['snr (dB)'] = snr_db
    true_params_df.to_csv(ModelFolderName + '/true_generative_model.csv', index=False)
    np.save(ModelFolderName + '/stacked_scaled_domain_used_for_training.npy', stacked_domain.cpu().detach().numpy())
    np.save(ModelFolderName + '/stacked_unscaled_domain_used_for_training.npy',
            stacked_domain_unscaled.cpu().detach().numpy())
    np.save(ModelFolderName + '/IC_stacked_domain_used_for_training.npy', IC_stacked_domain.cpu().detach().numpy())
    np.save(ModelFolderName + '/current_data_used_for_training.npy', measured_current_tensor.cpu().detach().numpy())
    if model_name.lower() == 'hh':
        np.save(ModelFolderName + '/true_states_used_for_training_hh_only.npy', true_states_tensor.cpu().detach().numpy())
    return t_domain_unscaled, t_domain, param_sample_unscaled, param_sample, measured_current_tensor, state_on_domain # this last bit only needed to compute the RHS
########################################################################################################################
# start the main script
if __name__ == '__main__':
    figureFolder = direcory_names['figures']
    modelFolder = direcory_names['models']
    # define the cpu as device because we don't need to send anything to the GPU at this stage
    device = pt.device('cpu')
    model_name = 'hh'
    nSamples = 500
    # load the protocols
    # EK = -80
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
    t_domain_unscaled, t_domain, param_sample_unscaled, param_sample, measured_current_tensor, pinn_state = generate_HH_training_set_to_files(unique_times,
                                                                nSamples, model_name=model_name, snr_db=30, scaled_domain_size=10)
    stacked_domain_unscaled = stack_inputs(t_domain_unscaled, param_sample_unscaled)
    stacked_domain = stack_inputs(t_domain, param_sample)
    precomputed_RHS_params = RHS_tensors_precompute(unique_times, pinn_state, stacked_domain_unscaled, device)
    # ^^ note that pinn_state is only used to get the correct shape in this function
    # check the dimensions of the generated tensors
    print(f"Stacked domain shape: {stacked_domain.shape}")
    print(f"Stacked unscaled domain shape: {stacked_domain_unscaled.shape}")
    print(f"Precomputed RHS parameters shape (per parameter): {precomputed_RHS_params[0].shape}")
    print(f"Length of the list precomputed RHS parameters: {len(precomputed_RHS_params)}")
    print(f"Note that RHS paramerers have not been stored to file and will have to be recomputed from the stacked domains")
    print(f"Measured current tensor shape: {measured_current_tensor.shape}")
    del stacked_domain, stacked_domain_unscaled, measured_current_tensor, precomputed_RHS_params
    ####################################################################################################################
    ## test loading the data from the files
    ####################################################################################################################
    # get the model folder name
    ModelFolderName = modelFolder + '/' + model_name.lower() + '_data_test'
    true_model_params = pd.read_csv(ModelFolderName + '/true_generative_model.csv')
    stacked_domain = pt.tensor(np.load(ModelFolderName + '/stacked_scaled_domain_used_for_training.npy'), dtype=pt.float32)
    stacked_domain_unscaled = pt.tensor(np.load(ModelFolderName + '/stacked_unscaled_domain_used_for_training.npy'), dtype=pt.float32)
    IC_stacked_domain = pt.tensor(np.load(ModelFolderName + '/IC_stacked_domain_used_for_training.npy'), dtype=pt.float32)
    measured_current_tensor = pt.tensor(np.load(ModelFolderName + '/current_data_used_for_training.npy'), dtype=pt.float32)
    if model_name.lower() == 'hh':
        true_states_tensor = pt.tensor(np.load(ModelFolderName + '/true_states_used_for_training_hh_only.npy'), dtype=pt.float32)
    print(f"True model parameters: \n{true_model_params}")
    print(f"Stacked domain shape: {stacked_domain.shape}")
    print(f"Stacked unscaled domain shape: {stacked_domain_unscaled.shape}")
    print(f"IC stacked domain shape: {IC_stacked_domain.shape}")
    print(f"Measured current tensor shape: {measured_current_tensor.shape}")
    if model_name.lower() == 'hh':
        print(f"True states tensor shape: {true_states_tensor.shape}")
    precomputed_RHS_params = RHS_tensors_precompute(unique_times, pinn_state, stacked_domain_unscaled, device)
    print(f"Precomputed RHS parameters shape (per parameter): {precomputed_RHS_params[0].shape}")
    print(f"Length of the list precomputed RHS parameters: {len(precomputed_RHS_params)}")
