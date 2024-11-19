from methods.preliminaries import *
import imageio.v2 as imageio
import os

indeces = [i for i in range(0, 300001, 10000)]
figureFolder = direcory_names['figures']
device_to_plot = 'cuda'
generative_model_name = 'hh_all_inputs_model'
RHS_name = 'hh'

if RHS_name.lower() == 'test':
        filenames_output = [f'{figureFolder}/test_NN_approximation_iter_{i}.png' for i in indeces]
        filenames_layer = [f'{figureFolder}/test_layer_outpusts_iter_' + str(i) + '.png' for i in indeces]
        filenames_params = [f'{figureFolder}/test_params_iter_' + str(i) + '.png' for i in indeces]
elif RHS_name.lower() == 'hh':
        figFolderName = figureFolder + '/' + RHS_name + '_data_' + device_to_plot
        filenames_output = [f'{figFolderName}/{generative_model_name}_NN_approximation_iter_{i}.png' for i in indeces]
        # filenames_layer = [f'{figFolderName}/{generative_model_name}_layer_outpusts_iter_' + str(i) + '.png' for i in indeces]
        # filenames_params = [f'{figFolderName}/{generative_model_name}_params_iter_' + str(i) + '.png' for i in indeces]
elif RHS_name.lower() == 'hh_current_model':
        filenames_output = [f'{figureFolder}/hh_current_model_NN_approximation_iter_{i}.png' for i in indeces]
        # filenames_layer = [f'{figureFolder}/hh_current_model_layer_outpusts_iter_' + str(i) + '.png' for i in indeces]
        # filenames_params = [f'{figureFolder}/hh_current_model_params_iter_' + str(i) + '.png' for i in indeces]
elif RHS_name.lower() == 'hh_w_current':
        filenames_output = [f'{figureFolder}/hh_w_current_NN_approximation_iter_{i}.png' for i in indeces]
        # filenames_layer = [f'{figureFolder}/hh_w_current_layer_outpusts_iter_' + str(i) + '.png' for i in indeces]
        # filenames_params = [f'{figureFolder}/hh_w_current_params_iter_' + str(i) + '.png' for i in indeces]
elif RHS_name.lower() == 'test_all_inputs_model':
        figFolderName = figureFolder + '/'
        filenames_output = [f'{figFolderName}/test_all_inputs_model_NN_approximation_iter_{i}.png' for i in indeces]
        # filenames_layer = [f'{figFolderName}/test_all_inputs_model_layer_outpusts_iter_' + str(i) + '.png' for i in indeces]
        # filenames_params = [f'{figFolderName}/test_all_inputs_model_params_iter_' + str(i) + '.png' for i in indeces]



images_output = [imageio.imread(filename) for filename in filenames_output]
# images_layer = [imageio.imread(filename) for filename in filenames_layer]
# images_params = [imageio.imread(filename) for filename in filenames_params]

# save the images to a gif file
imageio.mimsave(figFolderName + RHS_name.lower()+'_NN_approximation.gif', images_output, fps=2)
# # save the images to a gif file
# imageio.mimsave(figureFolder +'/'+RHS_name.lower()+'_layer_outputs.gif', images_layer, fps=2)
# # save the images to a gif file
# imageio.mimsave(figureFolder +'/'+RHS_name.lower()+'_params.gif', images_params, fps=2)


