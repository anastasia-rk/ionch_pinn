from methods.preliminaries import *
from methods.nn_classes import *

def sigmoid(x,scale=1,shift=0):
    return 1/(1+pt.exp(-scale*(x-shift)))

def plot_pinn_params_all_inputs(pinn):
    """
    This function plots the weights and biases of all layers of the network
    :param pinn: the fully connected network
    :return:
    """
    all_names = [name for _, (name, _) in enumerate(pinn.named_parameters())]
    # get unique layer names
    first_layer_name = all_names[0].split('.')[0]
    last_layer_name = all_names[-1].split('.')[0]
    hidden_layer_names = [name.split('.')[0] + '.' + name.split('.')[1] for name in all_names[2:-2]]
    nHiddenLayers = len(hidden_layer_names)
    # drip elements of layer list that are duplicates but preserve order - done in weird way from stackoverflow!
    layer_names = [first_layer_name] + list(dict.fromkeys(hidden_layer_names)) + [last_layer_name]
    param_tensor_list = list(pinn.named_parameters())
    # all_parameters = [param_tensor_list[i][1].data for i in range(len(param_tensor_list))]
    # find minimum and maximum values of all parameters to limit the colorbars
    # min_val = min([par.min() for par in all_parameters]).detach().numpy()
    # max_val = max([par.max() for par in all_parameters]).detach().numpy()
    # plot heatmaps of the weights and biases for each layer
    fig, axes = plt.subplots(1, 2*len(layer_names)-1, figsize=(nHiddenLayers*4, nHiddenLayers+0.5), dpi=400)
    axes = axes.ravel()
    for iLayer, layerName in enumerate(layer_names):
        # find which elements of param_tensor_list corresponds to the current layer name
        params_of_layer = [param_tensor_list[i] for i in range(len(param_tensor_list)) if
                           layerName in param_tensor_list[i][0]]
        # get the weights and biases
        weights = params_of_layer[0][1].data
        weights = weights.cpu()
        max_weight = weights.max().item()
        min_weight = weights.min().item()
        # print(weights.shape)
        biases = params_of_layer[1][1].data.unsqueeze(1)
        biases = biases.cpu()
        max_bias = biases.max().item()
        min_bias = biases.min().item()
        # print(biases.shape)
        # plot the weights
        if iLayer == len(layer_names) - 1:
            hm = axes[2 * iLayer].imshow(weights.t(), cmap='plasma',
                                         # vmin=min_val, vmax=max_val,
                                    extent=[0, weights.shape[0], 0, weights.shape[1]], aspect=0.1)
            axes[2 * iLayer].set_title(r'$w_{' + str(iLayer) + '}^{T}$')
            height = axes[2 * iLayer].get_ylim()[1]
            width = axes[2 * iLayer].get_xlim()[1]
            strtext = f'{max_weight:.6f}'
            strtext = r'$max(w_{' + str(iLayer) + '})$ =  ' + strtext
            axes[2 * iLayer].text(width + 2, height / 2, strtext, rotation=90, ha='center', va='center')
            strtext = f'{min_weight:.6f}'
            strtext = r'$min(w_{' + str(iLayer) + '})$ =  ' + strtext
            axes[2 * iLayer].text(width + 5, height / 2, strtext, rotation=90, ha='center', va='center')
        elif iLayer == 0:
            hm = axes[2 * iLayer].imshow(weights, cmap='plasma',
                                    extent=[0, weights.shape[1], 0, weights.shape[0]], aspect=0.1)
            axes[2 * iLayer].set_title(r'$w_{' + str(iLayer) + '}$')
            height = axes[2 * iLayer].get_ylim()[1]
            width = axes[2 * iLayer].get_xlim()[1]
            strtext = f'{max_weight:.6f}'
            strtext = r'$max(w_{' + str(iLayer) + '})$ =  ' + strtext
            axes[2 * iLayer].text(width + 2, height / 2, strtext, rotation=90, ha='center', va='center')
            strtext = f'{min_weight:.6f}'
            strtext = r'$min(w_{' + str(iLayer) + '})$ =  ' + strtext
            axes[2 * iLayer].text(width + 5, height / 2, strtext, rotation=90, ha='center', va='center')
        else:
            hm = axes[2 * iLayer].imshow(weights, cmap='plasma',
                                    extent=[0, weights.shape[1], 0, weights.shape[0]], aspect=1)
            axes[2 * iLayer].set_title(r'$w_{' + str(iLayer) + '}$')
            height = axes[2 * iLayer].get_ylim()[1]
            width = axes[2 * iLayer].get_xlim()[1]
            strtext = f'{max_weight:.6f}'
            strtext = r'$max(w_{' + str(iLayer) + '})$ =  ' + strtext
            axes[2 * iLayer].text(width + 15, height / 2, strtext, rotation=90, ha='center', va='center')
            strtext = f'{min_weight:.6f}'
            strtext = r'$min(w_{' + str(iLayer) + '})$ =  ' + strtext
            axes[2 * iLayer].text(width + 50, height / 2, strtext, rotation=90, ha='center', va='center')
            axes[2 * iLayer].set_title(r'$w_{' + str(iLayer) + '}$')
        # plot the biases
        if iLayer == len(layer_names) - 1:
            # no need to plot it as it is a single value
            strtext = f'{max_bias:.6f}'
            strtext = r'$b_{out}$ =  ' + strtext
            plt.gcf().text(0.98, 0.5, strtext, rotation=90,ha='center', va='center')
        else:
            hm = axes[2 * iLayer + 1].imshow(biases, cmap='plasma',
                                        extent=[0, biases.shape[1], 0, biases.shape[0]], aspect=0.1)
            # axes[2 * iLayer + 1].figure.colorbar(hm, ax=axes[2 * iLayer + 1], orientation='vertical', location='bottom')
            axes[2 * iLayer + 1].set_title(r'$b_{' + str(iLayer) + '}$')
            # get ymax and xmax of the axis
            height = axes[2 * iLayer + 1].get_ylim()[1]
            width = axes[2 * iLayer + 1].get_xlim()[1]
            strtext = f'{max_bias:.6f}'
            strtext = r'$max(b_{'+str(iLayer)+'})$ =  ' + strtext
            axes[2 * iLayer + 1].text(width+2, height/2, strtext, rotation=90, ha='center', va='center')
            strtext = f'{min_bias:.6f}'
            strtext = r'$min(b_{'+str(iLayer)+'})$ =  ' + strtext
            axes[2 * iLayer + 1].text(width+5, height/2, strtext, rotation=90, ha='center', va='center')
        axes[2 * iLayer].grid(False)
        if iLayer < len(layer_names) - 1:
            axes[2 * iLayer + 1].grid(False)
        # for first layer, only set xtick labels at 1
        # list_of_ticks = [0, 100, 200, 300]
        # list_of_ticks = [0, 150, 300, 450]
        list_of_ticks = [0, 250, 500]
        if iLayer == 0:
            axes[2 * iLayer].set_xticks([0.5, 9.5])
            axes[2 * iLayer].set_xticklabels([1, 10])
            axes[2 * iLayer + 1].set_xticks([0.5])
            axes[2 * iLayer + 1].set_xticklabels([1])
        #     for last layer, only set ytick labels at 1
        elif iLayer == len(layer_names) - 1:
            axes[2 * iLayer].set_xticks([0.5])
            axes[2 * iLayer].set_xticklabels([1])
            axes[2 * iLayer].set_yticks(list_of_ticks)
            axes[2 * iLayer].set_yticklabels(list_of_ticks)
            # axes[2 * iLayer + 1].set_xticks([0.5])
            # axes[2 * iLayer + 1].set_xticklabels([1])
            # axes[2 * iLayer + 1].set_yticks([0.5])
            # axes[2 * iLayer + 1].set_yticklabels([1])
        else:
            axes[2 * iLayer].set_xticks(list_of_ticks)
            axes[2 * iLayer].set_xticklabels(list_of_ticks)
            # for weights, set ytick labels to match xtick labels
            axes[2 * iLayer].set_yticks(list_of_ticks)
            axes[2 * iLayer].set_yticklabels(list_of_ticks)
            #  for biases set the ytick labels to match the number of nodes
            axes[2 * iLayer + 1].set_yticks(list_of_ticks)
            axes[2 * iLayer + 1].set_yticklabels(list_of_ticks)
            # for biases, only set xtick labels at 1
            axes[2 * iLayer + 1].set_xticks([0.5])
            axes[2 * iLayer + 1].set_xticklabels([1])
    # end of for loop
    return fig, axes

def plot_pinn_params(pinn):
    all_names = [name for _, (name, _) in enumerate(pinn.named_parameters())]
    # get unique layer names
    first_layer_name = all_names[0].split('.')[0]
    last_layer_name = all_names[-1].split('.')[0]
    hidden_layer_names = [name.split('.')[0] + '.' + name.split('.')[1] for name in all_names[2:-2]]
    nHiddenLayers = len(hidden_layer_names)
    # drip elements of layer list that are duplicates but preserve order - done in weird way from stackoverflow!
    layer_names = [first_layer_name] + list(dict.fromkeys(hidden_layer_names)) + [last_layer_name]
    param_tensor_list = list(pinn.named_parameters())
    # all_parameters = [param_tensor_list[i][1].data for i in range(len(param_tensor_list))]
    # find minimum and maximum values of all parameters to limit the colorbars
    # min_val = min([par.min() for par in all_parameters]).detach().numpy()
    # max_val = max([par.max() for par in all_parameters]).detach().numpy()
    # plot heatmaps of the weights and biases for each layer
    fig, axes = plt.subplots(1, 2*len(layer_names)-1, figsize=(nHiddenLayers*4, nHiddenLayers+0.5), dpi=400)
    axes = axes.ravel()
    for iLayer, layerName in enumerate(layer_names):
        # find which elements of param_tensor_list corresponds to the current layer name
        params_of_layer = [param_tensor_list[i] for i in range(len(param_tensor_list)) if
                           layerName in param_tensor_list[i][0]]
        # get the weights and biases
        weights = params_of_layer[0][1].data
        weights = weights.cpu()
        max_weight = weights.max().item()
        min_weight = weights.min().item()
        # print(weights.shape)
        biases = params_of_layer[1][1].data.unsqueeze(1)
        biases = biases.cpu()
        max_bias = biases.max().item()
        min_bias = biases.min().item()
        # print(biases.shape)
        # plot the weights
        if iLayer == len(layer_names) - 1:
            hm = axes[2 * iLayer].imshow(weights.t(), cmap='plasma',
                                         # vmin=min_val, vmax=max_val,
                                    extent=[0, weights.shape[0], 0, weights.shape[1]], aspect=0.1)
            axes[2 * iLayer].set_title(r'$w_{' + str(iLayer) + '}^{T}$')
            height = axes[2 * iLayer].get_ylim()[1]
            width = axes[2 * iLayer].get_xlim()[1]
            strtext = f'{max_weight:.6f}'
            strtext = r'$max(w_{' + str(iLayer) + '})$ =  ' + strtext
            axes[2 * iLayer].text(width + 2, height / 2, strtext, rotation=90, ha='center', va='center')
            strtext = f'{min_weight:.6f}'
            strtext = r'$min(w_{' + str(iLayer) + '})$ =  ' + strtext
            axes[2 * iLayer].text(width + 5, height / 2, strtext, rotation=90, ha='center', va='center')
        elif iLayer == 0:
            hm = axes[2 * iLayer].imshow(weights, cmap='plasma',
                                    extent=[0, weights.shape[1], 0, weights.shape[0]], aspect=0.1)
            axes[2 * iLayer].set_title(r'$w_{' + str(iLayer) + '}$')
            height = axes[2 * iLayer].get_ylim()[1]
            width = axes[2 * iLayer].get_xlim()[1]
            strtext = f'{max_weight:.6f}'
            strtext = r'$max(w_{' + str(iLayer) + '})$ =  ' + strtext
            axes[2 * iLayer].text(width + 2, height / 2, strtext, rotation=90, ha='center', va='center')
            strtext = f'{min_weight:.6f}'
            strtext = r'$min(w_{' + str(iLayer) + '})$ =  ' + strtext
            axes[2 * iLayer].text(width + 5, height / 2, strtext, rotation=90, ha='center', va='center')
        else:
            hm = axes[2 * iLayer].imshow(weights, cmap='plasma',
                                    extent=[0, weights.shape[1], 0, weights.shape[0]], aspect=1)
            axes[2 * iLayer].set_title(r'$w_{' + str(iLayer) + '}$')
            height = axes[2 * iLayer].get_ylim()[1]
            width = axes[2 * iLayer].get_xlim()[1]
            strtext = f'{max_weight:.6f}'
            strtext = r'$max(w_{' + str(iLayer) + '})$ =  ' + strtext
            axes[2 * iLayer].text(width + 15, height / 2, strtext, rotation=90, ha='center', va='center')
            strtext = f'{min_weight:.6f}'
            strtext = r'$min(w_{' + str(iLayer) + '})$ =  ' + strtext
            axes[2 * iLayer].text(width + 50, height / 2, strtext, rotation=90, ha='center', va='center')
            axes[2 * iLayer].set_title(r'$w_{' + str(iLayer) + '}$')
        # plot the biases
        if iLayer == len(layer_names) - 1:
            # no need to plot it as it is a single value
            strtext = f'{max_bias:.6f}'
            strtext = r'$b_{out}$ =  ' + strtext
            plt.gcf().text(0.98, 0.5, strtext, rotation=90,ha='center', va='center')
        else:
            hm = axes[2 * iLayer + 1].imshow(biases, cmap='plasma',
                                        extent=[0, biases.shape[1], 0, biases.shape[0]], aspect=0.1)
            # axes[2 * iLayer + 1].figure.colorbar(hm, ax=axes[2 * iLayer + 1], orientation='vertical', location='bottom')
            axes[2 * iLayer + 1].set_title(r'$b_{' + str(iLayer) + '}$')
            # get ymax and xmax of the axis
            height = axes[2 * iLayer + 1].get_ylim()[1]
            width = axes[2 * iLayer + 1].get_xlim()[1]
            strtext = f'{max_bias:.6f}'
            strtext = r'$max(b_{'+str(iLayer)+'})$ =  ' + strtext
            axes[2 * iLayer + 1].text(width+2, height/2, strtext, rotation=90, ha='center', va='center')
            strtext = f'{min_bias:.6f}'
            strtext = r'$min(b_{'+str(iLayer)+'})$ =  ' + strtext
            axes[2 * iLayer + 1].text(width+5, height/2, strtext, rotation=90, ha='center', va='center')
        axes[2 * iLayer].grid(False)
        if iLayer < len(layer_names) - 1:
            axes[2 * iLayer + 1].grid(False)
        # for first layer, only set xtick labels at 1
        # list_of_ticks = [0, 100, 200, 300]
        # list_of_ticks = [0, 150, 300, 450]
        list_of_ticks = [0, 250, 500]
        if iLayer == 0:
            axes[2 * iLayer].set_xticks([0.5])
            axes[2 * iLayer].set_xticklabels([1])
            axes[2 * iLayer + 1].set_xticks([0.5])
            axes[2 * iLayer + 1].set_xticklabels([1])
        #     for last layer, only set ytick labels at 1
        elif iLayer == len(layer_names) - 1:
            axes[2 * iLayer].set_xticks([0.5])
            axes[2 * iLayer].set_xticklabels([1])
            axes[2 * iLayer].set_yticks(list_of_ticks)
            axes[2 * iLayer].set_yticklabels(list_of_ticks)
            # axes[2 * iLayer + 1].set_xticks([0.5])
            # axes[2 * iLayer + 1].set_xticklabels([1])
            # axes[2 * iLayer + 1].set_yticks([0.5])
            # axes[2 * iLayer + 1].set_yticklabels([1])
        else:
            axes[2 * iLayer].set_xticks(list_of_ticks)
            axes[2 * iLayer].set_xticklabels(list_of_ticks)
            # for weights, set ytick labels to match xtick labels
            axes[2 * iLayer].set_yticks(list_of_ticks)
            axes[2 * iLayer].set_yticklabels(list_of_ticks)
            #  for biases set the ytick labels to match the number of nodes
            axes[2 * iLayer + 1].set_yticks(list_of_ticks)
            axes[2 * iLayer + 1].set_yticklabels(list_of_ticks)
            # for biases, only set xtick labels at 1
            axes[2 * iLayer + 1].set_xticks([0.5])
            axes[2 * iLayer + 1].set_xticklabels([1])
    # end of for loop
    return fig, axes


def plot_layers_as_bases(pinn, domain,domain_scaled):
    all_names = [name for _, (name, _) in enumerate(pinn.named_parameters())]
    # get unique layer names
    first_layer_name = all_names[0].split('.')[0]
    last_layer_name = all_names[-1].split('.')[0]
    hidden_layer_names = [name.split('.')[0] + '.' + name.split('.')[1] for name in all_names[2:-2]]
    # drip elements of layer list that are duplicates but preserve order - done in weird way from stackoverflow!
    layer_names = [first_layer_name] + list(dict.fromkeys(hidden_layer_names)) + [last_layer_name]
    # now we plot output of each layer
    previous_activator_outputs = domain
    fig, axes = plt.subplots(len(layer_names), 1, figsize=(10, 6), dpi=400, sharex=True)
    axes = axes.ravel()
    param_tensor_list = list(pinn.named_parameters())
    for iLayer, layerName in enumerate(layer_names):
        # find which elements of param_tensor_list corresponds to the current layer name
        params_of_layer = [param_tensor_list[i] for i in range(len(param_tensor_list)) if
                           layerName in param_tensor_list[i][0]]
        # get the weights and biases
        weights = params_of_layer[0][1].data.t()
        biases = params_of_layer[1][1].data
        # note that this tensor will always have the lenght of the domain as the first dimension
        activator_inputs = pt.matmul(previous_activator_outputs, weights) + biases.repeat(len(domain), 1)
        # compute activator outputs and store them for the next round
        previous_activator_outputs = pt.sigmoid(activator_inputs)
        # get shape of activation outputs
        nRows, nCols = previous_activator_outputs.shape
        # create the table of colours
        colours = plt.cm.tab20(np.linspace(0, 1, nCols))
        for iCol in range(nCols):
            # plot the sigmoid of the input
            if iLayer == len(layer_names) - 1:  # if the layer is the output layer, plot the output of the network
                axes[iLayer].plot(domain_scaled.detach().numpy(), activator_inputs.detach().numpy(),
                                  color=colours[iCol], linewidth=0.5,
                                  alpha=0.5)
            else:  # in every other case, just output activator output
                axes[iLayer].plot(domain_scaled.detach().numpy(), previous_activator_outputs[:, iCol].detach().numpy(),
                                  color=colours[iCol], linewidth=0.5, alpha=0.5)
        axes[iLayer].set_ylabel(layerName)
        axes[iLayer] = pretty_axis(axes[iLayer], legendFlag=False)
        if iLayer < len(layer_names) - 1:
            axes[iLayer].set_ylim(-0.01, 1.01)
    # end of for loop
    axes[iLayer].set_xlabel('Time')
    return fig, axes

def plot_layers_as_bases_over_time(pinn, domain, time_domain_unscaled):
    all_names = [name for _, (name, _) in enumerate(pinn.named_parameters())]
    # get unique layer names
    first_layer_name = all_names[0].split('.')[0]
    last_layer_name = all_names[-1].split('.')[0]
    hidden_layer_names = [name.split('.')[0] + '.' + name.split('.')[1] for name in all_names[2:-2]]
    # drip elements of layer list that are duplicates but preserve order - done in weird way from stackoverflow!
    layer_names = [first_layer_name] + list(dict.fromkeys(hidden_layer_names)) + [last_layer_name]
    # now we plot output of each layer
    previous_activator_outputs = domain
    fig, axes = plt.subplots(len(layer_names), 1, figsize=(10, 6), dpi=400, sharex=True)
    axes = axes.ravel()
    param_tensor_list = list(pinn.named_parameters())
    for iLayer, layerName in enumerate(layer_names):
        # find which elements of param_tensor_list corresponds to the current layer name
        params_of_layer = [param_tensor_list[i] for i in range(len(param_tensor_list)) if
                           layerName in param_tensor_list[i][0]]
        # get the weights and biases
        weights = params_of_layer[0][1].data.t()
        biases = params_of_layer[1][1].data
        # note that this tensor will always have the lenght of the domain as the first dimension
        activator_inputs = pt.matmul(previous_activator_outputs, weights) + biases.repeat(len(domain), 1)
        # compute activator outputs and store them for the next round
        previous_activator_outputs = pt.sigmoid(activator_inputs)
        # get shape of activation outputs
        nRows, nCols = previous_activator_outputs.shape
        # create the table of colours
        colours = plt.cm.tab20(np.linspace(0, 1, nCols))
        for iCol in range(nCols):
            # plot the sigmoid of the input
            if iLayer == len(layer_names) - 1:  # if the layer is the output layer, plot the output of the network
                axes[iLayer].plot(time_domain_unscaled.detach().numpy(), activator_inputs.detach().numpy(),
                                  color=colours[iCol], linewidth=0.5,
                                  alpha=0.5)
            else:  # in every other case, just output activator output
                axes[iLayer].plot(time_domain_unscaled.detach().numpy(), previous_activator_outputs[:, iCol].detach().numpy(),
                                  color=colours[iCol], linewidth=0.5, alpha=0.5)
        axes[iLayer].set_ylabel(layerName)
        axes[iLayer] = pretty_axis(axes[iLayer], legendFlag=False)
        if iLayer < len(layer_names) - 1:
            axes[iLayer].set_ylim(-0.01, 1.01)
    # end of for loop
    axes[iLayer].set_xlabel('Time')
    return fig, axes

def plot_costs(loss_total, stored_costs, lambdas, all_cost_names):
    # plot all the cost functions and the total cost, all in separate axes with shared xaxis
    # find which costs have positive lambdas
    active_cost_names = [cost_name for cost_name in all_cost_names if lambdas[cost_name] > 0]
    fig_costs, axes = plt.subplots(len(active_cost_names) + 1, 1, figsize=(10, 7), sharex=True, dpi=400)
    axes = axes.ravel()
    axes[0].plot(loss_total)
    axes[0].set_yscale('log')
    axes[0].set_ylabel('Total loss')
    axes[0] = pretty_axis(axes[0], legendFlag=False)
    for iCost, cost_name in enumerate(active_cost_names):
        axes[iCost + 1].plot(stored_costs[cost_name], label='lambda=' + '{:.4E}'.format(lambdas[cost_name]),
                             linewidth=1)
        axes[iCost + 1].set_yscale('log')
        axes[iCost + 1].set_ylabel(cost_name)
        axes[iCost + 1] = pretty_axis(axes[iCost + 1], legendFlag=True)
    axes[-1].set_xlabel('Training step')
    return fig_costs, axes

if __name__ == '__main__':
    pinn = FCN(1, 2, 500, 2)
    a = list(pinn.named_parameters())[0][1]
    end_time = 1
    scaling_coeff = 100
    nSteps  = 1000
    domain = pt.linspace(0, end_time, nSteps).requires_grad_(True).unsqueeze(1)
    domain_scaled = pt.linspace(0, end_time*scaling_coeff, nSteps).requires_grad_(True).unsqueeze(1)
    ####################################################################################################################
    # create the domains over which we want to make pinn
    end_point = 10
    time_scaling_coeff = 6000 / end_point
    g_scaling_coeff = 100 /end_point
    nSteps_t = 1000
    nSteps_g = 10
    t_domain = pt.linspace(0, end_point, nSteps_t).requires_grad_(True).unsqueeze(1)
    t_domain_scaled = pt.linspace(0, end_point * time_scaling_coeff, nSteps_t).requires_grad_(True).unsqueeze(1)
    g_domain = pt.linspace(0, end_point, nSteps_g).requires_grad_(True).unsqueeze(1)
    g_domain_scaled = pt.linspace(0, end_point * g_scaling_coeff, nSteps_g).requires_grad_(True).unsqueeze(1)
    # Try making a multi-input pinn
    nInputs = 2
    lenInputs = [1, 1] # this is basically to show dimension of the input, it is not the same as the size of the domain!
    dimInputs = [nSteps_t,nSteps_g]
    nLayers = 3
    nHidden = 500
    nOutputs = 3
    # We require the PINN that will produce as an output the tensor tha corresponds to all combinations of input values
    # for example, if we have an input of size 1000x1 and input of size 100x1, we want the output to be of size 1000x100
    pinn_multi_input = FCN_multi_input(nInputs, lenInputs, nOutputs, nHidden, nLayers)
    # try passing domains through the pinn as a list
    # x_out = pinn_multi_input([t_domain_scaled, g_domain_scaled])
    ####################################################################################################################
    # try creating a FCN with the meshgrid of input tensors instead
    t_domain = pt.linspace(0, end_point, nSteps_t).requires_grad_(True)
    t_domain_scaled = pt.linspace(0, end_point * time_scaling_coeff, nSteps_t).requires_grad_(True)
    g_domain = pt.linspace(0, end_point, nSteps_g).requires_grad_(True)
    g_domain_scaled = pt.linspace(0, end_point * g_scaling_coeff, nSteps_g).requires_grad_(True)
    # create the meshgrid
    t_mesh, g_mesh = pt.meshgrid(t_domain, g_domain)
    # stack two tensors along a new dimension
    stacked_domain = pt.stack((t_mesh, g_mesh), dim=2)
    pinn_for_mesh = FCN(2, 2, 500, 2)
    x_out = pinn_for_mesh(stacked_domain)
    ####################################################################################################################
    # plot the activation functions of the network as a function of domain
    fig, axes = plot_layers_as_bases(pinn, domain, domain)
    axes[-1].set_xlabel('Input domain at initialisation')
    plt.tight_layout()
    # save the figure
    plt.savefig('figures/Activators_as_basis.png', dpi=400)
    # plt.show()

    # plot the weights and biases of the network to check if everything is plotted correctly
    fig, axes = plot_pinn_params(pinn)
    # set the suptitle
    axes[0].set_ylabel('test',fontsize=12)
    plt.subplots_adjust(left=0,right=1,wspace=0.1, hspace=1.4)
    # plt.subplots_adjust(wspace=0, hspace=0)
    # fig.tight_layout(pad=0,w_pad=-2.5, h_pad=0)
    # save the figure
    fig.savefig('figures/Weights_and_biases.png', dpi=400)

    # #     test loading the presaved network
    # rhs_name = 'hh'
    # nLayers = 2
    # nHidden = 250
    # nOutputs = 1
    # nInputs = 1
    # PATH = 'models/' + rhs_name.lower() +'_'+str(nLayers)+'_layers_'+str(nHidden)+'_nodes_'+str(nInputs)+'_ins'+str(nOutputs)+'_outs.pth'
    # pinn_states = pt.load(PATH)
    # pinn = FCN(nInputs, nOutputs, nHidden, nLayers)
    # pinn.load_state_dict(pt.load(PATH))
    # tlim = [0, 14899]
    # scaling_coeff = tlim[-1]/10
    # times = np.linspace(*tlim, tlim[-1] - tlim[0], endpoint=False)
    # times_scaled = times/scaling_coeff
    # times_all_domain = pt.tensor(times / scaling_coeff, dtype=pt.float32).requires_grad_(True).unsqueeze(1)
    # pinn_output = pinn(times_all_domain).detach().numpy()
    # # plot the output of the model
    # fig_data, axes = plt.subplots(1, 1, figsize=(10, 5), dpi=400)
    # axes.plot(times, pinn_output,'--',label='PINN solution')
    # axes.set_xlabel('Time')
    # axes.set_ylabel('State')
    # axes = pretty_axis(axes, legendFlag=True)
    # plt.tight_layout()
    # plt.savefig('figures/'+rhs_name.lower()+'_training_overview.png')
    # plt.show()

    # plot the output of the model


    # plot a sigmoid
    x = pt.linspace(0, 15000, 10000)
    y = pt.sigmoid(x)
    fig, ax = plt.subplots(dpi=400)
    ax.plot(x, y)
    ax = pretty_axis(ax)
    plt.show()


#     generate a uniform sample between -1 and 1
    scales = pt.rand(1000, 1) * 2 - 1
    shifts = pt.rand(10000, 1) * 10 - 5

    x = pt.linspace(0, 15000, 10000)
    for i in range(1000):
        y = sigmoid(x, scales[i], shifts[i])
        plt.plot(x, y)
    axes = pretty_axis(plt.gca())
    plt.tight_layout()
    plt.savefig('figures/sigmoid_samples_on_full.png', dpi=400)
    plt.close()

    x_scaled = pt.linspace(-10, 10, 10000)
    for i in range(1000):
        y = sigmoid(x_scaled, scales[i], shifts[i])
        plt.plot(x_scaled, y)
    axes = pretty_axis(plt.gca())
    plt.tight_layout()
    plt.savefig('figures/sigmoid_samples_on_10.png', dpi=400)
    plt.close()

    x_scaled_to_1 = pt.linspace(-1, 1, 10000)
    for i in range(1000):
        y = sigmoid(x_scaled_to_1, scales[i], shifts[i])
        plt.plot(x_scaled_to_1, y)
    axes = pretty_axis(plt.gca())
    plt.tight_layout()
    plt.savefig('figures/sigmoid_samples_on_1.png', dpi=400)
    plt.close()


