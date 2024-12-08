from methods.generate_training_set import *
from methods.plot_figures import *
from torch.utils.data import DataLoader, TensorDataset
import torch.multiprocessing as mp

########################################################################################################################
# start the main script
if __name__ == '__main__':
    isATest = False
    # set the training data
    model_name = 'kemp'  # model we use to generate the synthetic data for data cost
    rhs_name = 'hh'  # the misspecified right hand side model to be used in gradient cost
    snr_in_db = 30  # signal to noise ratio in dB for the synthetic data generation
    scaled_domain_size = 10  # size of the domain for the scaled input
    if isATest:
        nSamples = 10
        nPerBatch = 2
        maxIter = 101
        plotEvery = 10
    else:
        nSamples = 500
        nPerBatch = 50
        maxIter = 60001
        plotEvery = 20000
    ######################################################################################################
    # set the folders for figures and pickles
    figureFolder = direcory_names['figures']
    modelFolder = direcory_names['models']
    pickleFolder = direcory_names['pickles']
    # create folder for figure storage
    FigFolderName = figureFolder + '/' + model_name.lower() + '_data_' + device.type
    # create the folder for data storage
    ModelFolderName = modelFolder + '/' + model_name.lower() + '_data_' + device.type
    #  creat folder for pickles
    PickleFolderName = pickleFolder + '/' + model_name.lower() + '_data_' + device.type

    if isATest:
        ModelFolderName = ModelFolderName + '_test'
        FigFolderName = FigFolderName + '_test'
        PickleFolderName = PickleFolderName + '_test'

    if not os.path.exists(FigFolderName):
        os.makedirs(FigFolderName)
    if not os.path.exists(ModelFolderName):
        os.makedirs(ModelFolderName)
    if not os.path.exists(PickleFolderName):
        os.makedirs(PickleFolderName)
    ####################################################################################################################
    # set up the colour wheel for plotting output at different training samples - this will be useful for plotting
    colours = plt.cm.PuOr(np.linspace(0, 1, nSamples))
    # make a throwaway counntour plot to generate a heatmap of conductances
    fig, ax = plt.subplots(1, 1, figsize=(10, 6), dpi=400)
    Z = [[0, 0], [0, 0]]
    levels = np.linspace(0, nSamples - 1,
                         nSamples)  # in this case we want to iterate over samples rather than values of parameters
    cond_heatmap = plt.contourf(Z, levels, cmap='PuOr')
    plt.clf()
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
    ####################################################################################################################
    # generate the input sample for training the PINN and generate all necessseary intermediate values for the training
    ArchTestFlag = False
    training_set_files = [name for name in os.listdir(ModelFolderName) if name.endswith('.npy') and 'train' in name]
    if len(training_set_files) > 0:
        #  load stacked domain and measured current from files
        stacked_domain = pt.tensor(np.load(ModelFolderName + '/stacked_scaled_domain_used_for_training.npy'),
                                   dtype=pt.float32).requires_grad_(True)
        stacked_domain_unscaled = pt.tensor(np.load(ModelFolderName + '/stacked_unscaled_domain_used_for_training.npy'),
                                            dtype=pt.float32).requires_grad_(True)
        IC_stacked_domain = pt.tensor(np.load(ModelFolderName + '/IC_stacked_domain_used_for_training.npy'),
                                      dtype=pt.float32).requires_grad_(True)
        pinn_state = pt.tensor(np.load(ModelFolderName + '/true_states_used_for_training_hh_only.npy'),
                               dtype=pt.float32).requires_grad_(True)
        measured_current = np.load(ModelFolderName + '/current_data_used_for_training.npy')
        measured_current_tensor = pt.tensor(measured_current, dtype=pt.float32).requires_grad_(True)
    else:
        t_domain_unscaled, t_domain, param_sample_unscaled, param_sample, measured_current_tensor, pinn_state = (
            generate_HH_training_set_to_files(unique_times,
                                              nSamples, model_name=model_name, snr_db=snr_in_db,
                                              scaled_domain_size=scaled_domain_size))
        stacked_domain_unscaled = stack_inputs(t_domain_unscaled, param_sample_unscaled)
        stacked_domain = stack_inputs(t_domain, param_sample)
        IC_t_domain = pt.tensor([unique_times[0]], dtype=pt.float32)
        IC_stacked_domain = stack_inputs(IC_t_domain, param_sample)
    # derive other necessary values for training
    measured_current = measured_current_tensor[0, :].detach().numpy()
    IC = pt.tensor([0, 1])  # I think for training on Kemp, we have nothing to compare our initial conditions to.
    t_scaling_coeff = scaled_domain_size / unique_times[-1]
    param_scaling_coeff = scaled_domain_size / pt.max(stacked_domain_unscaled)
    # send everything to device
    stacked_domain_unscaled = stacked_domain_unscaled.to(device)
    stacked_domain = stacked_domain.to(device)
    measured_current_tensor = measured_current_tensor.to(device)
    IC_stacked_domain = IC_stacked_domain.to(device)
    IC = IC.to(device)
    precomputed_RHS_params = RHS_tensors_precompute(unique_times, pinn_state, stacked_domain_unscaled, device)
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
    # define the optimiser and the loss function weights
    optimiser = pt.optim.Adam(pinn.parameters(), lr=1e-4, weight_decay=1e-4)
    ## at this stage, everything that is used for training has to be on the device!
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
            if 'optimizer_state_dict' in checkpoint.keys():
                optimiser.load_state_dict(checkpoint['optimizer_state_dict'])
            if 'epoch' in checkpoint.keys():
                firstIter = checkpoint['epoch']
            if 'lambdas' in checkpoint.keys():
                lambdas = checkpoint['lambdas']
            if 'loss_names' in checkpoint.keys():
                all_cost_names = checkpoint['loss_names']
            # then rename all files in the model folder by adding '_epoch' + str(firstIter) to the end of the name
            for filename in os.listdir(ModelFolderName):
                if filename.endswith('.pth'):
                    os.rename(ModelFolderName + '/' + filename,
                              ModelFolderName + '/' + filename[:-4] + '_epoch_' + str(firstIter) + '.pth')
                if filename.endswith('.pkl'):
                    os.rename(ModelFolderName + '/' + filename,
                              ModelFolderName + '/' + filename[:-4] + '_epoch_' + str(firstIter) + '.pkl')
        else:
            print('No model state found in the checkpoint. Initalsing fist iteation')
            firstIter = 0
            pinn, lambdas, all_cost_names = initialise_optimisation(pinn)
    else:
        print('No pre-trained weights found. Initialising the network.')
        # initialise the costs
        firstIter = 0
        pinn, lambdas, all_cost_names = initialise_optimisation(pinn)
    ########################################################################################################################
    ## plots to check the network architecture
    # plot the activation functions of the network as a function of domain
    # fig, axes = plot_layers_as_bases(pinn, t_domain, t_domain)
    # axes[-1].set_xlabel('Input domain at initialisation')
    # plt.tight_layout()
    # # save the figure
    # plt.savefig(figureFolder + '/Activators_as_basis.png',dpi=400)
    # # plt.show()
    # plot the weights and biases of the network to check if everything is set correctly
    # marks = [int(i) for i in np.linspace(0, nHidden, 3)]
    # fig, axes = plot_pinn_params_all_inputs(pinn)
    # # set the suptitle
    # fig.suptitle('test', fontsize=16)
    # plt.subplots_adjust(left=0, right=1, wspace=0.1, hspace=1)
    # # save the figure
    # fig.savefig(FigFolderName + '/Weights_and_biases.png', dpi=400)
    ####################3##################################################################################################
    # create a tensor dataset - we must include parts of RHS parameters that are precomputed to split them into appropriate parts
    # note that precomputed_RHS_params is a tuple of tensors - we need to unpack it to send it into the dataloader
    dataset = TensorDataset(stacked_domain, *precomputed_RHS_params, measured_current_tensor)
    # if the device we use is cpu, set num_workers to 60, if it is cuda then set them to 0
    num_workers = 0
    if device.type == 'cuda':
        num_workers = 0
    elif device.type == 'cpu':
        # have not setup multiprocessing properly, so this does not work yet
        num_workers = min(60, os.cpu_count())
    print(f'Number of workers used:{num_workers}')
    dataloader = DataLoader(dataset, batch_size=nPerBatch, shuffle=False, num_workers=num_workers,
                            generator=worker_generator)
    # send the IC domain to device
    ########################################################################################################################
    rhs_error_state_weights = [1, 1]
    scaling_coeffs = [t_scaling_coeff, param_scaling_coeff, rhs_error_state_weights]
    stored_costs = {name: [] for name in all_cost_names}
    loss_seq = []
    # start the optimisation loop
    for i in tqdm(range(firstIter, firstIter + maxIter)):
        # prepare losses for cumulation
        running_loss = 0.0
        running_IC_loss = 0.0
        running_RHS_loss = 0.0
        running_data_loss = 0.0
        running_L1_loss = 0.0
        running_penalty_loss = 0.0
        for i_batch, (input_batch, *precomputed_RHS_batch, target_batch) in enumerate(dataloader):
            # if we sent all the parts of the dataset to device, we do not need to pass them individually
            # zero the gradients
            optimiser.zero_grad()
            output_batch = pinn(input_batch)
            losses = compute_pinn_loss(pinn, input_batch, output_batch, target_batch, lambdas,
                                       scaling_coeffs, IC, precomputed_RHS_batch, device)
            loss, loss_ic, loss_rhs, loss_data, L1, target_penalty = losses
            ################################################################################################################
            # compute the total loss
            # the backward pass computes the gradient of the loss with respect to the parameters
            loss.backward(retain_graph=True)
            # make a step in the parameter space
            optimiser.step()
            # store the losses
            running_loss += loss.item()
            running_IC_loss += loss_ic.item()
            running_RHS_loss += loss_rhs.item()
            running_data_loss += loss_data.item()
            running_L1_loss += L1.item()
            running_penalty_loss += target_penalty.item()
            running_losses = [running_IC_loss, running_RHS_loss, running_data_loss, running_L1_loss,
                              running_penalty_loss]
        ####################################################################################################################
        # store the loss values
        for iLoss in range(len(all_cost_names)):
            stored_costs[all_cost_names[iLoss]].append(running_losses[iLoss])
        loss_seq.append(running_loss)
        ####################################################################################################################
        # occasionally plot the output, save the network state and plot the costs
        if i % plotEvery == 0:
            # save the model to a pickle file
            pt.save({
                'epoch': i,
                'model_state_dict': pinn.state_dict(),
                'optimizer_state_dict': optimiser.state_dict(),
                'loss': loss,
                'losses': running_losses,
                'lambdas': lambdas,
                'loss_names': all_cost_names
            }, ModelFolderName + '/' + pinnName + '.pth')
            # save the costs to a pickle file - this is just for plotting
            with open(ModelFolderName + '/' + pinnName + '_training_costs.pkl', 'wb') as f:
                pkl.dump(stored_costs, f)
            # plotting for different samples - we need to call the correct tenso since we have changed how the input tensors are generated
            # in order to plot over the whole interval, we need to produce output
            state_domain = pinn(stacked_domain)
            # use custom detivarive function to compute the derivatives of outputs, because grad assumed that the output is a scalar
            dxdt, rhs_pinn, current_pinn = compute_derivs_and_current(stacked_domain, state_domain,
                                                                      precomputed_RHS_params, scaling_coeffs, device)
            ################################################################################################################
            # plot network output and errors
            fig, axes = plt.subplots(2,nOutputs +1 , figsize=(10, 6),sharex=True, dpi=400)
            # genreate 2d ndarray that starts at 0 and ends at 2*nOutputs
            axes = axes.ravel()
            for iOutput in range(nOutputs):
                # axes[iOutput].plot(unique_times, state_true[iOutput,:], label="IVP solution", linewidth=1, color="k", alpha=0.3)
                for iSample in range(0, nSamples):
                    axes[iOutput].plot(unique_times, state_domain[iSample,...,iOutput].cpu().detach().numpy(),
                                       color=colours[iSample],linewidth=0.5,alpha=0.7)
                axes[iOutput].set_ylabel('State')
                axes[iOutput] = pretty_axis(axes[iOutput], legendFlag=False)
                axes[iOutput].set_ylim([-0.5,1.5])
            # plot the gradient error
            for iOutput in range(nOutputs):
                for iSample in range(0, nSamples):
                    if iSample == 0:
                        # give a label
                        axes[nOutputs + iOutput + 1].plot(unique_times, rhs_error_state_weights[iOutput] * (
                                    dxdt[iSample, ..., iOutput].cpu().detach().numpy() - rhs_pinn[
                                iSample, ..., iOutput].cpu().detach().numpy()),
                                                          linewidth=0.5, color=colours[iSample], alpha=0.7,
                                                          label=f"Error weight: {rhs_error_state_weights[iOutput]}")
                    else:
                        # plot without a label
                        axes[nOutputs + iOutput + 1].plot(unique_times, rhs_error_state_weights[iOutput] * (
                                    dxdt[iSample, ..., iOutput].cpu().detach().numpy() - rhs_pinn[
                                iSample, ..., iOutput].cpu().detach().numpy()),
                                                          linewidth=0.5, color=colours[iSample], alpha=0.7)
                axes[nOutputs+iOutput+1].set_xlabel('Time')
                axes[nOutputs+iOutput+1].set_ylabel('Derivative error')
                axes[nOutputs+iOutput+1] = pretty_axis(axes[nOutputs+iOutput+1], legendFlag=True)
                axes[nOutputs+iOutput+1].set_ylim([-0.2, 0.2])
            # plot the current and current error
            axes[nOutputs].plot(unique_times, measured_current, label="Measured current", color="k",  linewidth=1, alpha=0.3)
            # for as many conductances as we put in, plot the current
            for iSample in range(0, nSamples):
                # plot the current
                axes[nOutputs].plot(unique_times, current_pinn[iSample,:].cpu().detach().numpy(), color=colours[iSample],linewidth=0.5,alpha=0.7) #label = "PINN current"
                # plot the current error
                axes[-1].plot(unique_times, measured_current - current_pinn[iSample,:].cpu().detach().numpy(),
                              color=colours[iSample], linewidth=0.5, alpha=0.7)
            axes[nOutputs].set_ylabel('Current')
            axes[nOutputs] = pretty_axis(axes[nOutputs], legendFlag=False)
            axes[nOutputs].set_ylim([-4, 5])
            # axes[-1].plot(time_of_domain, measured_current - current_pinn.detach().numpy()[0,:], color="k",linewidth=0.5, alpha=0.6)
            axes[-1].set_xlabel('Time')
            axes[-1].set_ylabel('Current error')
            axes[-1] = pretty_axis(axes[-1], legendFlag=False)
            axes[-1].set_ylim([-10, 10])
            fig.tight_layout(pad=0.3, w_pad=0.4, h_pad=0.2)
            if not ArchTestFlag:
                cbar = fig.colorbar(cond_heatmap, ax=axes.tolist(), location='top', aspect=50) #ticks=levels
                # cbar.ax.set_xticklabels(["{:.2f}".format(j+1) for j in levels])
                cbar.ax.set_ylabel('j')
                cbar.ax.yaxis.label.set_rotation(90)
            # set the suptitle  of the figure
            fig.suptitle(f"i = {i}")
            fig.savefig(FigFolderName + '/'+rhs_name.lower()+'_NN_approximation_iter_' + str(i) + '.png')
            ################################################################################################################
            # plot costs of the iteration
            fig_costs, axes = plot_costs(loss_seq, stored_costs, lambdas, all_cost_names)
            fig_costs.tight_layout()
            fig_costs.savefig(FigFolderName + '/' + rhs_name.lower() + '_costs_iter_' + str(i) + '.png')
            plt.close('all')
            ################################################################################################################
            # # we also want to plot the layers as basis functions
            # fig, axes = plot_layers_as_bases(pinn, domain, domain_scaled)
            # axes[0].set_title(f"i ={i}")
            # fig.tight_layout()
            # # save the figure
            # fig.savefig(figureFolder + '/'+rhs_name.lower()+'_layer_outpusts_iter_' + str(i) + '.png', dpi=400)
            # # and parameter values to trace how they are updated
            # fig, axes = plot_pinn_params_all_inputs(pinn)
            # # set the suptitle
            # axes[0].set_ylabel(f"i={i}")
            # plt.subplots_adjust(left=0,right=1,wspace=0.1, hspace=1.3)
            # save the figure
            # fig.savefig(figureFolder + '/'+rhs_name.lower()+'_params_iter_' + str(i) + '.png', dpi=400)
            # plt.close('all')
            #  check the convergence of the loss function
            if i > firstIter:
                diff_of_cost = np.abs(loss_seq[-1] - loss_seq[-2]) / loss_seq[-1]
                print(diff_of_cost)
                if diff_of_cost < 1e-6:
                    print('Cost coverged.')
                    break
        #  end of plotting condition
    # end of training loop
    ########################################################################################################################
    # save the model to a pickle file
    pt.save({
        'epoch': i,
        'model_state_dict': pinn.state_dict(),
        'optimizer_state_dict': optimiser.state_dict(),
        'loss': loss,
        'losses': running_losses,
        'lambdas': lambdas,
        'loss_names': all_cost_names
    }, ModelFolderName + '/' + pinnName + '.pth')
    # save the costs to a pickle file
    with open(ModelFolderName + '/' + pinnName + '_training_costs.pkl', 'wb') as f:
        pkl.dump(stored_costs, f)
    ########################################################################################################################
    # plot the output of the model on the entire time interval
    times_scaled = times * t_scaling_coeff
    times_all_domain = pt.tensor(times_scaled, dtype=pt.float32)
    stacked_domain = stack_inputs(times_all_domain, param_sample.detach())
    # send the domain to device
    stacked_domain = stacked_domain.to(device)
    # generate output of the trained PINN and current
    pinn_output = pinn(stacked_domain)
    pinn_current = observation_tensors(times, pinn_output, stacked_domain, device)
    pinn_current = pinn_current/param_scaling_coeff
    pinn_output = pinn_output.cpu().detach().numpy()
    pinn_current = pinn_current.cpu().detach().numpy()
    ########################################################################################################################
    # plot outputs at training points
    fig, axes = plt.subplots(2+nOutputs, 1, figsize=(10, 7), sharex=True, dpi=400)
    axes = axes.ravel()
    # plot the solution for all outputs
    for iOutput in range(nOutputs):
        # axes[iOutput].plot(times, state_true_all[iOutput,:], label='IVP solution', color='k', alpha=0.3)
        # this part could be wrong because we may have a multi-dim tensor where only the first dimension matches times
        for iSample in range(0, nSamples):
            axes[iOutput].plot(times, pinn_output[iSample,:, iOutput], '--', color=colours[iSample], alpha=0.7, linewidth=0.5)
        # axes[iOutput+1].plot(times, pinn_output[..., iOutput], '--', label='PINN solution')
        # axes[iOutput].set_xlabel('Time')
        axes[iOutput].set_ylabel('State')
        axes[iOutput] = pretty_axis(axes[iOutput], legendFlag=False)
    iAxis = nOutputs
    # plot current vs PINN current
    # axes[iAxis].plot(times, current_true, label='True current', color='k', alpha=0.3)
    for iSample in range(0, nSamples):
        axes[iAxis].plot(times, pinn_current[iSample,:], '--',color=colours[iSample], alpha=0.7, linewidth=0.5)
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
    fig.tight_layout(pad=0.3, w_pad=0.4, h_pad=0.2)
    if rhs_name.lower() == 'hh_all_inputs_model':
        cbar = fig.colorbar(cond_heatmap, ax=axes.tolist(), location='top', aspect=50) #ticks=levels
        # cbar.ax.set_xticklabels(["{:.2f}".format(j+1) for j in levels])
        cbar.ax.set_ylabel('j')
        cbar.ax.yaxis.label.set_rotation(90)
    # set the suptitle  of the figure
    fig.suptitle("Trained PINN output at training points")
    plt.savefig(FigFolderName + '/'+rhs_name.lower()+'_trained_nn_output_at_training_values.png')
    ########################################################################################################################
    if model_name.lower() == 'hh':
        thetas_true = thetas_hh_baseline
        thetas_true_tensor = pt.tensor(thetas_true).unsqueeze(-1) * param_scaling_coeff
        stacked_true = stack_inputs(times_all_domain, thetas_true_tensor)
        # get the true current
        stacked_true = stacked_true.to(device)
        pinn_output_at_truth = pinn(stacked_true)
        pinn_current_at_truth = observation_tensors(times, pinn_output_at_truth, stacked_true,device)
        pinn_current_at_truth = pinn_current_at_truth / param_scaling_coeff
        pinn_output_at_truth = pinn_output_at_truth.cpu().detach().numpy()
        pinn_current_at_truth = pinn_current_at_truth.cpu().detach().numpy()
        # plot the outputs at the true conductance
        fig_data, axes = plt.subplots(2+nOutputs, 1, figsize=(10, 7), sharex=True, dpi=400)
        axes = axes.ravel()
        for iOutput in range(nOutputs):
            # axes[iOutput].plot(times, state_true_all[iOutput,:], label='True state',color='k', alpha=0.3)
            axes[iOutput].plot(times, pinn_output_at_truth[0,..., iOutput], '--', label='PINN solution')
            axes[iOutput].set_ylabel('State')
            axes[iOutput] = pretty_axis(axes[iOutput], legendFlag=True)
        iAxis = nOutputs
        # plot current vs PINN current
        # axes[iAxis].plot(times, current_true, label='True current',color='k', alpha=0.3)
        axes[iAxis].plot(times, pinn_current_at_truth[0,:], '--', label='PINN current')
        axes[iAxis].set_ylabel('Current')
        axes[iAxis] = pretty_axis(axes[iAxis], legendFlag=True)
        iAxis = nOutputs+1
        # plot the voltage
        axes[iAxis].plot(times, V(times),color='k', alpha=0.3)
        axes[iAxis].set_xlabel('Time')
        axes[iAxis].set_ylabel('Input voltage')
        axes[iAxis] = pretty_axis(axes[iAxis], legendFlag=False)
        plt.tight_layout()
        plt.savefig(FigFolderName + '/'+rhs_name.lower()+'_trained_nn_output_at_truth.png')
    ########################################################################################################################
    fig, axes = plot_costs(loss_seq, stored_costs, lambdas, all_cost_names)
    fig.tight_layout()
    fig.savefig(FigFolderName + '/' + rhs_name.lower() + '_costs.png')