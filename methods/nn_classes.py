# This file contains the classes for the neural networks used in the PINN training. The first class is a simple
# feedforward neural network with a sigmoid activation function. The second class is a multi-input neural network
# that can take multiple inputs and outputs. The class contains the training method that is used to train the
# network using the PINN loss function. The training method takes a data loader, the number of epochs, and the validation data
# as input and returns the losses for each epoch.

from methods.hh_rhs_computation import *
from methods.plot_figures import *

def derivative_multi_input(outputs, inputs):
    """
    This function computes the derivative of the outputs w.r.t. the inputs and returns only derivatives w.r.t to time
    as a stacked tensor
    :param outputs: outputs of the NN
    :param inputs: inputs of the NN, the first one being time
    :return: the derivatives of all outputs w.r.t. time of size [nSamples x nTimes x nOutputs]
    """
    list_of_grads = []
    # the number of outputs is the last element of the shape of the tensor - this will hold becaus of the way we stack inputs
    nOutputs = outputs.shape[-1]
    # we want to iterate over the outputs, not their shape
    for iOutput in range(nOutputs):
        output = outputs[..., iOutput]
        ones = pt.ones_like(output)
        grad = pt.autograd.grad(output, inputs, grad_outputs=ones, create_graph=True)[
            0]  # this will compute the gradient of the output w.r.t. all inputs!
        # the time is alway the first in the list of inputs, so we need to only store the first element of the last dimension of grad
        # I think with the way we now create the input tensor this line is wrong - check this!
        grad_wrt_time_only = grad[..., 0].unsqueeze(
            -1)  # need to unsquze to make sure we can stack them along the state dimension
        list_of_grads.append(grad_wrt_time_only)
    #  create a tensor from the list of tensor by stacking them along the last dimension
    return pt.cat(list_of_grads, dim=-1)
#
class FCN(nn.Module):
    """
    Defines a standard fully-connected network in PyTorch.This is a simple feedforward multi-input
    multi-output networks with a sigmoid activation function. The training methods are defined outside of the class.
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

# define a multi-input neural network with all methods
class FCN_multi_input(nn.Module):
    """
    Defines a standard fully-connected network in PyTorch. This is a simple feedforward multi-input
    multi-output networks with a sigmoid activation function.
    :method forward: defines the forward pass of the network
    :method train: trains the network using the PINN loss function
    :method _train_iteration: trains the network for one epoch
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

    def train(self, data_loader, N_EPOCHS, validation_data=None):
        all_cost_names = ['IC', 'RHS', 'L1', 'Data', 'Penalty', 'Total']
        stored_costs = {name: [] for name in all_cost_names}
        for epoch in range(N_EPOCHS):
            running_losses = self._train_iteration(data_loader)
            for iLoss in range(len(all_cost_names)):
                stored_costs[all_cost_names[iLoss]].append(running_losses[iLoss])
            running_loss = running_losses[-1]
            # validation placeholder
            val_loss = None
            if validation_data is not None:
                y_hat = self(validation_data['X'])
                val_loss = self.lossFct(input=y_hat, target=validation_data['y']).detach().numpy()
                print('[%d] loss: %.3f | validation loss: %.3f' %
                      (epoch + 1, running_loss, val_loss))
            else:
                print('[%d] loss: %.3f' %
                      (epoch + 1, running_loss))

    def _train_iteration(self, data_loader, lambdas, time_of_domain, IC_stacked_domain, IC, scaling_args):
        rhs_error_state_weights, t_scaling_coeff, param_scaling_coeff = scaling_args
        lambda_ic, lambda_rhs, lambda_data, lambda_l1, lambda_penalty = lambdas
        # prepare losses for cumulation
        running_loss = 0.0
        running_IC_loss = 0.0
        running_RHS_loss = 0.0
        running_data_loss = 0.0
        running_L1_loss = 0.0
        running_penalty_loss = 0.0
        for i_batch, (input_batch, *precomputed_RHS_batch, target_batch) in enumerate(data_loader):
            # if we sent all the parts of the dataset to device, we do not need to pass them individually
            # # extract time from the input batch - in the tensor dataset, the time is the first element of the input batch
            # time_of_batch = input_batch[0,:,0]
            # time_of_batch = time_of_batch.cpu().detach().numpy()
            # time_of_batch =time_of_batch/t_scaling_coeff
            time_of_batch = time_of_domain
            # compute the gradient loss
            state_batch = self(input_batch)
            # use custom detivarive function to compute the derivatives of outputs, because grad assumed that the output is a scalar
            dxdt = derivative_multi_input(state_batch, input_batch)
            dxdt = dxdt * t_scaling_coeff  # restore to the original scale
            # compute the RHS from the precomputed parameters - note that this is in the original scale too
            rhs_pinn = RHS_from_precomputed(state_batch, precomputed_RHS_batch)
            # compute the current tensor to compare with the measured current
            current_pinn = observation_tensors(time_of_batch, state_batch, input_batch)
            current_pinn = current_pinn / param_scaling_coeff
            ################################################################################################################
            # compute the loss function
            ################################################################################################################
            # compute the RHS loss
            if lambda_rhs != 0:
                error_rhs = (rhs_pinn - dxdt) ** 2
                # aplify the error along the second state dimension by multiplying it by 10
                for iState in range(error_rhs.shape[-1]):
                    error_rhs[..., iState] = error_rhs[..., iState] * rhs_error_state_weights[iState]
                # simple trapezoidal rule to compute the integral
                sumerror = error_rhs[:, 1:, ...] + error_rhs[:, :-1, ...]
                dt = input_batch[:, 1:, 0] - input_batch[:, :-1, 0]
                # this should only expand it once I think
                for iDim in range(len(sumerror.shape) - len(dt.shape)):
                    dt = dt.unsqueeze(-1)
                dt = dt.expand(sumerror.shape)
                loss_rhs = pt.sum(sumerror * dt / 2)
                del sumerror
            else:
                loss_rhs = pt.tensor(0, dtype=pt.float32).to(device)
            ################################################################################################################
            # compute the IC loss
            if lambda_ic != 0:
                # state_ic = state_domain[0,...] # if we are starting from 0 at the time domain
                state_ic = self(IC_stacked_domain)  # or we can put the IC domain through the pinn
                loss_ic = pt.sum((state_ic - IC) ** 2)
            else:
                loss_ic = pt.tensor(0, dtype=pt.float32).to(device)
            ################################################################################################################
            # commpute the data loss w.r.t. the current
            if lambda_data != 0:
                residuals_data = current_pinn - target_batch
                loss_data = pt.sum((residuals_data) ** 2)  # by default, pytorch sum goes over all dimensions
            else:
                loss_data = pt.tensor(0, dtype=pt.float32).to(device)
            ################################################################################################################
            # compute the L1 norm
            if lambda_l1 != 0:
                par_pinn = list(self.parameters())
                L1 = pt.tensor([par_pinn[l].abs().sum() for l in range(len(par_pinn))]).sum()
            else:
                L1 = pt.tensor(0, dtype=pt.float32).to(device)
            ################################################################################################################
            #  compute network output penalty (we know that each output must be between 0 and 1)
            target_penalty = pt.tensor(0, dtype=pt.float32).to(device)
            if lambda_penalty != 0:
                for iOutput in range(state_batch.shape[-1]):
                    lower_bound = pt.zeros_like(state_batch[..., iOutput]).to(device)
                    upper_bound = pt.ones_like(state_batch[..., iOutput]).to(device)
                    target_penalty += pt.sum(pt.relu(lower_bound - state_batch[..., iOutput])) + pt.sum(
                        pt.relu(state_batch[..., iOutput] - upper_bound))
            ################################################################################################################
            def closure():
                # zero the gradients
                if pt.is_grad_enabled():
                    self.optim.zero_grad()
                # compute the total loss
                loss = lambda_ic * loss_ic + lambda_rhs * loss_rhs + lambda_data * loss_data + lambda_l1 * L1 + lambda_penalty * target_penalty
                # the backward pass computes the gradient of the loss with respect to the parameters
                loss.backward(retain_graph=True)
                return loss

            # make a step in the parameter space, then combine the batch losses
            self.optim.step(closure)
            loss = closure()
            running_loss += loss.item()
            running_IC_loss += loss_ic.item()
            running_RHS_loss += loss_rhs.item()
            running_data_loss += loss_data.item()
            running_L1_loss += L1.item()
            running_penalty_loss += target_penalty.item()
        return running_IC_loss, running_RHS_loss, running_data_loss, running_L1_loss, running_penalty_loss, running_loss

def initialise_optimisation(pinn):
    """
    This function initialises the optimisation settings for the PINN training
    :param pinn: the neural network - for initial setting of the biases
    :return:
    lambdas: the weights for the constituent loss terms
    all_cost_names: the names of the constituent loss terms
    """
    # initialise the cost weights
    lambda_ic = 1e-2  # 1e-2 # weight on the gradient fitting cost
    lambda_rhs = 1  # weight on the right hand side fitting cost
    lambda_l1 = 0  # weight on the L1 norm of the parameters
    lambda_data = 1e-7  # weight on the data fitting cost
    lambda_penalty = 1e-3  # weight on the output penalty
    lambdas = [lambda_ic, lambda_rhs, lambda_l1, lambda_data, lambda_penalty]
    # placeholder for storing the costs
    all_cost_names = ['IC', 'RHS', 'L1', 'Data', 'Penalty']
    # get the biases of the first layer
    biases = pinn.first_layer[0].bias.data
    # provide larger biases for the first layer as the initialsiation
    a = pt.tensor(-10)
    b = pt.tensor(10)
    biases_new = (b - a) * pt.rand_like(biases) + a
    # set the biases of the first layer
    pinn.first_layer[0].bias.data = biases_new
    return pinn, lambdas, all_cost_names

def compute_derivs_and_current(pinn_input, pinn_output, precomputed_rhs_part, scaling_coeffs, device):
    """
    This function computes the derivatives of the outputs w.r.t. the inputs and the current tensor
    :param pinn_input: pinn input tensor
    :param pinn_output: pinn output tensor
    :param precomputed_rhs_part: precomputed part of the right hand side
    :param scaling_coeffs: scaling coefficients for the domain and the parameters
    :param device: device to send the tensors to
    :return: derivative of the output, right hand side from the precomputed part, current tensor
    """
    t_scaling_coeff, param_scaling_coeff, rhs_error_state_weights = scaling_coeffs
    ################################################################################################################
    time_only = pinn_input[0, :, 0].cpu().detach().numpy() / t_scaling_coeff
    # use custom detivarive function to compute the derivatives of outputs, because grad assumed that the output is a scalar
    dxdt = derivative_multi_input(pinn_output, pinn_input)
    dxdt = dxdt * t_scaling_coeff  # restore to the original scale
    # compute the RHS from the precomputed parameters - note that this is in the original scale too
    rhs_pinn = RHS_from_precomputed(pinn_output, precomputed_rhs_part)
    # compute the current tensor to compare with the measured current
    current_pinn = observation_tensors(time_only, pinn_output, pinn_input, device)
    current_pinn = current_pinn / param_scaling_coeff
    return dxdt, rhs_pinn, current_pinn

def compute_pinn_loss(pinn, pinn_input, pinn_output, target, lambdas, scaling_coeffs, IC, precomputed_rhs_part, device):
    """
    This function computes the PINN loss function for a given input, output, target, lambdas and scaling coefficients
    :param pinn: the neural network
    :param pinn_input: input tensor
    :param pinn_output: pinn output tensor
    :param target: target tensor
    :param lambdas: weights for constituent loss terms
    :param scaling_coeffs: coefficients that rescale the domain
    :param IC: initial condition tensor
    :param precomputed_rhs_part: precomputed part of the right hand side
    :param device: the devide where to send losses
    :return: the combined loss and the individual losses
    """
    lambda_rhs, lambda_ic, lambda_data, lambda_l1, lambda_penalty = lambdas
    t_scaling_coeff, param_scaling_coeff, rhs_error_state_weights = scaling_coeffs
    ################################################################################################################
    dxdt, rhs_pinn, current_pinn = compute_derivs_and_current(pinn_input, pinn_output, precomputed_rhs_part, scaling_coeffs, device)
    ################################################################################################################
    # compute the loss function
    ################################################################################################################
    # compute the RHS loss
    if lambda_rhs != 0:
        error_rhs = (rhs_pinn - dxdt) ** 2
        # aplify the error along the second state dimension by multiplying it by 10
        for iState in range(error_rhs.shape[-1]):
            error_rhs[..., iState] = error_rhs[..., iState] * rhs_error_state_weights[iState]
        # simple trapezoidal rule to compute the integral
        sumerror = error_rhs[:, 1:, ...] + error_rhs[:, :-1, ...]
        dt = pinn_input[:, 1:, 0] - pinn_input[:, :-1, 0]
        # this should only expand it once I think
        for iDim in range(len(sumerror.shape) - len(dt.shape)):
            dt = dt.unsqueeze(-1)
        dt = dt.expand(sumerror.shape)
        loss_rhs = pt.sum(sumerror * dt / 2)
        del sumerror
    else:
        loss_rhs = pt.tensor(0, dtype=pt.float32).to(device)
    ################################################################################################################
    # compute the IC loss
    if lambda_ic != 0:
        # if we are starting from 0 at the time domain
        IC_stacked_domain = pinn_input[:, 0, ...].unsqueeze(1)
        state_IC = pinn_output[:, 0, ...]
        # state_IC = pinn(IC_stacked_domain)  # or we can put the IC domain through the pinn
        loss_ic = pt.sum((state_IC - IC) ** 2)
    else:
        loss_ic = pt.tensor(0, dtype=pt.float32).to(device)
    ################################################################################################################
    # commpute the data loss w.r.t. the current
    if lambda_data != 0:
        residuals_data = current_pinn - target
        loss_data = pt.sum((residuals_data) ** 2)  # by default, pytorch sum goes over all dimensions
    else:
        loss_data = pt.tensor(0, dtype=pt.float32).to(device)
    ################################################################################################################
    # compute the L1 norm
    if lambda_l1 != 0:
        par_pinn = list(pinn.parameters())
        L1 = pt.tensor([par_pinn[l].abs().sum() for l in range(len(par_pinn))]).sum()
    else:
        L1 = pt.tensor(0, dtype=pt.float32).to(device)
    ###############################################################################################################
    # #
    #  compute network output penalty (we know that each output must be between 0 and 1)
    target_penalty = pt.tensor(0, dtype=pt.float32).to(device)
    if lambda_penalty != 0:
        nOutputs = pinn_output.shape[-1]
        for iOutput in range(nOutputs):
            lower_bound = pt.zeros_like(pinn_output[..., iOutput]).to(device)
            upper_bound = pt.ones_like(pinn_output[..., iOutput]).to(device)
            target_penalty += pt.sum(pt.relu(lower_bound - pinn_output[..., iOutput])) + pt.sum(
                pt.relu(pinn_output[..., iOutput] - upper_bound))
    ################################################################################################################
    combined_loss = lambda_ic * loss_ic + lambda_rhs * loss_rhs + lambda_data * loss_data + lambda_l1 * L1 + lambda_penalty * target_penalty
    return (combined_loss, loss_rhs, loss_ic, loss_data, L1, target_penalty)

"""
Below here is the training loop as a function. It is not complete so is not used in the code!
"""

def train_pinn_loop(pinn, dataloader, optimiser, lambdas, scaling_coeffs, var_names, maxIter, plotEvery, device):
    nSamples = len(dataloader.dataset)
    nOutputs = dataloader.dataset.nOutputs
    rhs_name, pinnName, all_cost_names = var_names
    stored_costs = {name: [] for name in all_cost_names}
    loss_seq = []
    figureFolder = direcory_names["figures"]
    modelFolder = direcory_names["models"]
    for i in tqdm(range(maxIter)):
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
            losses = compute_pinn_loss(input_batch, output_batch, target_batch, lambdas, scaling_coeffs, IC, precomputed_rhs_part, device)
            loss, loss_rhs, loss_ic, loss_data, L1, target_penalty = losses
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
            running_losses = [running_IC_loss, running_RHS_loss, running_L1_loss, running_data_loss, running_penalty_loss]
        ####################################################################################################################
        # store the loss values
        for iLoss in range(len(all_cost_names)):
            stored_costs[all_cost_names[iLoss]].append(running_losses[iLoss])
        loss_seq.append(running_loss)
        ####################################################################################################################
        # occasionally plot the output, save the network state and plot the costs
        if i % plotEvery == 0:
            # save the model to a pickle file
            pt.save(pinn.state_dict(), ModelFolderName + '/' + pinnName + '.pth')
            # save the costs to a pickle file
            with open(ModelFolderName + '/' + pinnName + '_training_costs.pkl', 'wb') as f:
                pkl.dump(stored_costs, f)
            # plotting for different samples - we need to call the correct tenso since we have changed how the input tensors are generated
            # in order to plot over the whole interval, we need to produce output
            state_domain = pinn(stacked_domain)
            # use custom detivarive function to compute the derivatives of outputs, because grad assumed that the output is a scalar
            dxdt, rhs_pinn, current_pinn = compute_derivs_and_current(stacked_domain, state_domain, precomputed_rhs_part, scaling_coeffs, device)
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
            if i > 0:
                diff_of_cost = np.abs(loss_seq[-1] - loss_seq[-2]) / loss_seq[-1]
                print(diff_of_cost)
                if diff_of_cost < 1e-6:
                    print('Cost coverged.')
                    break
        #  end of plotting condition
    # end of training loop
    return pinn, loss_seq, stored_costs

if __name__ == '__main__':
    device = pt.device("cpu")
    figureFolder = direcory_names["figures"]

    # create neural network
    N = nn.Sequential(nn.Linear(1, 30), nn.Sigmoid(), nn.Linear(30, 1, bias=False))

    # Initial condition
    A = 0.

    # The Psi_t function
    Psi_t = lambda x: A + x * N(x)

    # The right hand side function
    f = lambda x, Psi: pt.exp(-x / 5.0) * pt.cos(x) - Psi / 5.0


    # The loss function
    def loss(x):
        x.requires_grad = True
        outputs = Psi_t(x)
        Psi_t_x = pt.autograd.grad(outputs, x, grad_outputs=pt.ones_like(outputs),
                                   create_graph=True)[0]
        return pt.mean((Psi_t_x - f(x, outputs)) ** 2)


    # Optimize (same algorithm as in Lagaris)
    optimizer = pt.optim.LBFGS(N.parameters())

    # The collocation points used by Lagaris
    x = pt.Tensor(np.linspace(0, 2, 200)[:, None])


    # Run the optimizer
    def closure():
        optimizer.zero_grad()
        l = loss(x)
        l.backward()
        return l


    for i in range(10):
        optimizer.step(closure)

    # compare the result to the true solution
    xx = np.linspace(0, 2, 200)[:, None]
    with pt.no_grad():
        yy = Psi_t(pt.Tensor(xx)).numpy()
    ## this is true solution
    yt = np.exp(-xx / 5.0) * np.sin(xx)

    fig, ax = plt.subplots(dpi=100)
    ax.plot(xx, yt, label='True')
    ax.plot(xx, yy, '--', label='Neural network approximation')
    ax.set_xlabel('$x$')
    ax.set_ylabel('$\Psi(x)$')
    ax = pretty_axis(ax)
    plt.legend(loc='best')
    fig.savefig(figureFolder + '/neural_network_approximation.png')