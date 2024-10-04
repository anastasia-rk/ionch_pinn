## a module that only generates the knots for the B-spline representation,
# trying to remove everything related to synthetic data generation
# imports
import matplotlib
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (20,10)
plt.rcParams['figure.dpi'] = 400
plt.rcParams['axes.facecolor']='white'
plt.rcParams['savefig.facecolor']='white'
plt.style.use("ggplot")
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica"]
        })
import numpy as np
import scipy as sp
from scipy.interpolate import BSpline
import torch as pt


# definitions
def pretty_axis(ax, legendFlag=True):
    """
    Function to set the axis colour, label, and grid
    :param ax: axis object
    :param legendFlag: flag to show the legend
    :return: axis object
    """
    # set axis labels and grid
    ax.set_facecolor('white')
    ax.grid(which='major', color='grey', linestyle='solid', alpha=0.2, linewidth=1)
    if legendFlag:
        ax.legend(loc='best', fontsize=12)
    return ax

def V(t):
    return volts_interpolated(t/ 1000)

def V_torch(t):
    return pt.tensor(volts_interpolated(t/ 1000))

def collocm(splinelist, tau):
    # collocation matrix for B-spline values (0-derivative)
    # inputs: splinelist - list of splines along one axis, tau - interval on which we wish to evaluate splines
    # outputs: collocation matrix
    mat = [[0] * len(tau) for _ in range(len(splinelist))]
    for i in range(len(splinelist)):
        mat[i][:] = splinelist[i](tau)
    return np.array(mat)

def generate_knots(times):
    """
    Function to generate the knots for the B-spline representation
    :param times: time instances
    :return:
    jump_indeces - indeces of jumps in the voltage protocol,
    times_roi - time instances for each segment,
    voltage_roi - voltage values for each segment,
    knots_roi - B-spline knots for each segment,
    collocation_roi - collocation matrix values for each segment
    degree - degree of the B-spline
    """
    volts_new = V(times)
    ####################################################################################################################
    ## B-spline representation setup
    # set times of jumps and a B-spline knot sequence
    # nPoints_closest = 4  # the number of points from each jump where knots are placed at the finest grid
    # nPoints_between_closest = 2  # step between knots at the finest grid
    # nPoints_around_jump = 80  # the time period from jump on which we place medium grid
    # step_between_knots = 16  # this is the step between knots around the jump in the medium grid
    # nPoints_between_jumps = 2  # this is the number of knots at the coarse grid corresponding to slowly changing values

    ## try a finer grid
    nPoints_closest = 10  # the number of points from each jump where knots are placed at the finest grid
    nPoints_between_closest = 1  # step between knots at the finest grid
    nPoints_around_jump = 100  # the time period from jump on which we place medium grid
    step_between_knots = 10  # this is the step between knots around the jump in the medium grid
    nPoints_between_jumps = 10  # this is the number of knots at the coarse grid corresponding to slowly changing values
    ## find switchpoints
    d2v_dt2 = np.diff(volts_new, n=2)
    dv_dt = np.diff(volts_new)
    der1_nonzero = np.abs(dv_dt) > 1e-1
    der2_nonzero = np.abs(d2v_dt2) > 1e-1
    switchpoints = [a and b for a, b in zip(der1_nonzero, der2_nonzero)]
    ####################################################################################################################
    # get the times of all jumps
    a = [0] + [i for i, x in enumerate(switchpoints) if x] + [
        len(times) - 1]  # get indeces of all the switchpoints, add t0 and tend
    # remove consecutive numbers from the list
    b = []
    for i in range(len(a)):
        if len(b) == 0:  # if the list is empty, we add first item from 'a' (In our example, it'll be 2)
            b.append(a[i])
        else:
            if a[i] > a[i - 1] + 1:  # for every value of a, we compare the last digit from list b
                b.append(a[i])
    jump_indeces = b.copy()
    ## create multiple segments limited by time instances of jumps
    times_roi = []
    voltage_roi = []
    knots_roi = []
    collocation_roi = []
    for iJump, jump in enumerate(jump_indeces[:-1]):  # loop oversegments (nJumps - )
        # define a region of interest - we will need this to preserve the
        # trajectories of states given the full clamp and initial position, while
        ROI_start = jump
        ROI_end = jump_indeces[iJump + 1] + 1  # add one to ensure that t_end equals to t_start of the following segment
        ROI = times[ROI_start:ROI_end]
        # get time points to compute the fit to ODE cost
        times_roi.append(ROI)
        # save voltage
        voltage_roi.append(V(ROI))
        ## add colloation points
        abs_distance_lists = [[(num - index) for num in range(ROI_start, ROI_end)] for index in
                              [ROI_start, ROI_end]]  # compute absolute distance between each time and time of jump
        min_pos_distances = [min(filter(lambda x: x >= 0, lst)) for lst in zip(*abs_distance_lists)]
        max_neg_distances = [max(filter(lambda x: x <= 0, lst)) for lst in zip(*abs_distance_lists)]
        # create a knot sequence that has higher density of knots after each jump
        knots_after_jump = [((x <= nPoints_closest) and (x % nPoints_between_closest == 0)) or (
                (nPoints_closest < x <= nPoints_around_jump) and (x % step_between_knots == 0)) for
                            x in min_pos_distances]  ##  ((x <= 2) and (x % 1 == 0)) or
        knots_before_jump = [((x >= -nPoints_closest) and (x % (nPoints_between_closest) == 0)) for x in
                             max_neg_distances]  # list on knots befor each jump - use this form if you don't want fine grid before the jump
        # knots_before_jump = [(x >= -1) for x in max_neg_distances]  # list on knots before each jump - add a fine grid
        knots_jump = [a or b for a, b in
                      zip(knots_after_jump, knots_before_jump)]  # logical sum of mininal and maximal distances
        # convert to numeric array again
        knot_indeces = [i + ROI_start for i, x in enumerate(knots_jump) if x]
        indeces_inner = knot_indeces.copy()
        # add additional coarse grid of knots between two jumps:
        for iKnot, timeKnot in enumerate(knot_indeces[:-1]):
            # add coarse grid knots between jumps
            if knot_indeces[iKnot + 1] - timeKnot > step_between_knots:
                # create evenly spaced points and drop start and end - those are already in the grid
                knots_between_jumps = np.rint(
                    np.linspace(timeKnot, knot_indeces[iKnot + 1], num=nPoints_between_jumps + 2)[1:-1]).astype(int)
                # add indeces to the list
                indeces_inner = indeces_inner + list(knots_between_jumps)
            # add copies of the closest points to the jump
        ## end loop over knots
        indeces_inner.sort()  # sort list in ascending order - this is done inplace
        degree = 3
        # define the Boor points to
        indeces_outer = [indeces_inner[0]] * 3 + [indeces_inner[-1]] * 3
        boor_indeces = np.insert(indeces_outer, degree,
                                 indeces_inner)  # create knots for which we want to build splines
        knots = times[boor_indeces]
        # save knots for the segment - including additional points at the edges
        knots_roi.append(knots)
        # build the collocation matrix using the defined knot structure
        coeffs = np.zeros(len(knots) - degree - 1)  # number of splines will depend on the knot order
        spl_ones = BSpline(knots, np.ones_like(coeffs), degree)
        splinest = [None] * len(coeffs)
        splineder = [None] * len(coeffs)  # the grid of indtividual splines is required to generate a collocation matrix
        for i in range(len(coeffs)):
            coeffs[i] = 1.
            splinest[i] = BSpline(knots, coeffs.copy(), degree,
                                  extrapolate=False)  # create a spline that only has one non-zero coeff
            coeffs[i] = 0.
        collocation_roi.append(collocm(splinest, ROI))
        # create inital values of beta to be used at the true value of parameters
    ##^ this loop stores the time intervals from which to draw collocation points and the data for piece-wise fitting # this to be used in params method of class ForwardModel
    return jump_indeces, times_roi, voltage_roi, knots_roi, collocation_roi, degree

####################################################################################################################
# Load the training protocol
#  load the voltage data:
volts = np.genfromtxt("protocols/protocol-staircaseramp.csv", skip_header=1, dtype=float, delimiter=',')
#  check when the voltage jumps
# read the times and valued of voltage clamp
volt_times, volts = np.genfromtxt("protocols/protocol-staircaseramp.csv", skip_header=1, dtype=float, delimiter=',').T
# interpolate with smaller time step (milliseconds)
volts_interpolated = sp.interpolate.interp1d(volt_times, volts, kind='previous') # this is the default protocol for fitting
del volt_times, volts
print('Loaded voltage protocol for model fitting.')
####################################################################################################################
# load the AP voltage protocol for validation
times_ap, voltage_ap = np.genfromtxt('protocols/ap.csv', delimiter=',', skip_header=1).T
times_ap_sec = times_ap/1000  # convert to s to match the other protocol and the V function
# if we want to use the AP protocol, we can use the following function to interpolate the voltage
volts_interpolated_ap = sp.interpolate.interp1d(times_ap_sec, voltage_ap, kind='previous')
del times_ap_sec, voltage_ap
print('Loaded voltage protocol for validation.')
####################################################################################################################
# main
if __name__ == '__main__':
    # test loading protocols and generating knots
    tlim = [0, 14899]
    times = np.linspace(*tlim, tlim[-1] - tlim[0], endpoint=False)
    # generate the segments with B-spline knots and intialise the betas for splines
    jump_indeces, times_roi, voltage_roi, knots_roi, collocation_roi, spline_order = generate_knots(times)
    nSegments = len(jump_indeces[:-1])
    print('Inner optimisation is split into ' + str(nSegments) + ' segments based on protocol steps.')
    # plot the staircase protocol and save into figures directory
    fig, ax = plt.subplots()
    ax.plot(times, V(times), 'k')
    ax.set_xlabel('Time, ms')
    ax.set_ylabel('Voltage, mV')
    ax.set_title('Staircase protocol for model fitting')
    ax = pretty_axis(ax, legendFlag=False)
    plt.tight_layout()
    plt.savefig('Figures/staircase_protocol.png')
    # plot voltage against time andd save into figures directory
    volts_interpolated = volts_interpolated_ap
    fig, ax = plt.subplots()
    ax.plot(times_ap, V(times_ap), 'k')
    ax.set_xlabel('Time, ms')
    ax.set_ylabel('Voltage, mV')
    ax.set_title('Action potential protocol for model validation')
    ax = pretty_axis(ax, legendFlag=False)
    plt.tight_layout()
    plt.savefig('Figures/ap_protocol.png')