import imageio
import os

indeces = [i for i in range(0, 200001, 10000)]
filenames = [f'Figures/test_NN_approximation_iter_{i}.png' for i in indeces]
images = [imageio.imread(filename) for filename in filenames]
# save the images to a gif file
imageio.mimsave('Figures/test_NN_approximation.gif', images, fps=2)

