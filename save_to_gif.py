import imageio.v2 as imageio
import os

indeces = [i for i in range(0, 400001, 10000)]

RHS_name = 'hh'
if RHS_name.lower() == 'test':
        filenames_output = [f'Figures/test_NN_approximation_iter_{i}.png' for i in indeces]
        filenames_layer = [f'Figures/test_layer_outpusts_iter_' + str(i) + '.png' for i in indeces]
        filenames_params = [f'Figures/test_params_iter_' + str(i) + '.png' for i in indeces]
elif RHS_name.lower() == 'hh':
        filenames_output = [f'Figures/hh_NN_approximation_iter_{i}.png' for i in indeces]
        filenames_layer = [f'Figures/HH_layer_outpusts_iter_' + str(i) + '.png' for i in indeces]
        filenames_params = [f'Figures/hh_params_iter_' + str(i) + '.png' for i in indeces]

images_output = [imageio.imread(filename) for filename in filenames_output]
images_layer = [imageio.imread(filename) for filename in filenames_layer]
images_params = [imageio.imread(filename) for filename in filenames_params]

# save the images to a gif file
imageio.mimsave('Figures/'+RHS_name.lower()+'_NN_approximation.gif', images_output, fps=2)
# save the images to a gif file
imageio.mimsave('Figures/'+RHS_name.lower()+'_layer_outputs.gif', images_layer, fps=2)
# save the images to a gif file
imageio.mimsave('Figures/'+RHS_name.lower()+'_params.gif', images_params, fps=2)


