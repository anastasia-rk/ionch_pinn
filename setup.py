import torch as pt
from generate_data import *

# load the protocols
load_protocols
# generate the segments with B-spline knots and intialise the betas for splines
jump_indeces, times_roi, voltage_roi, knots_roi, collocation_roi, spline_order = generate_knots(times)
jumps_odd = jump_indeces[0::2]
jumps_even = jump_indeces[1::2]
nSegments = len(jump_indeces[:-1])
print('Inner optimisation is split into ' + str(nSegments) + ' segments based on protocol steps.')
