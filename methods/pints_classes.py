from methods.generate_data import *
from methods.generate_training_set import *
from methods.nn_classes import *
import pints

class bsplineOutput(pints.ForwardModel):
    # this model outputs the discrepancy to be used in a rectangle quadrature scheme
    def simulate(self, parameters, support, times):
        # given times and return the simulated values
        coeffs_a, coeffs_r = np.split(parameters, 2)
        tck_a = (support, coeffs_a, spline_order)
        tck_r = (support, coeffs_r, spline_order)
        dot_a = sp.interpolate.splev(times, tck_a, der=1)
        dot_r = sp.interpolate.splev(times, tck_r, der=1)
        fun_a = sp.interpolate.splev(times, tck_a, der=0)
        fun_r = sp.interpolate.splev(times, tck_r, der=0)
        # the RHS must be put into an array
        dadr = two_state_model_log(times, [fun_a, fun_r], Thetas_ODE)
        rhs_theta = np.array(dadr)
        spline_surface = np.array([fun_a, fun_r])
        spline_deriv = np.array([dot_a, dot_r])
        # pack all required variables into the same array - will be the wrong orientation from pints preferred nTimes x nOutputs
        packed_output = np.concatenate((spline_surface, spline_deriv, rhs_theta), axis=0)
        return np.transpose(packed_output)

    def n_parameters(self):
        # Return the dimension of the parameter vector
        return nBsplineCoeffs

    def n_outputs(self):
        # Return the dimension of the output vector
        return nOutputs
####################################################################################################################
# define an error w.r.t B-spline parameters that assumes that it knows ODE parameters
class InnerCriterion(pints.ProblemErrorMeasure):
    # do I need to redefine custom init or can just drop this part?
    def __init__(self, problem, weights=None):
        super(InnerCriterion, self).__init__(problem)
        if weights is None:
            weights = [1] * self._n_outputs
        elif self._n_outputs != len(weights):
            raise ValueError(
                'Number of weights must match number of problem outputs.')
        # Check weights
        self._weights = np.asarray([float(w) for w in weights])

    # this function is the function of beta - bspline parameters
    def __call__(self, betas):
        # evaluate the integral at the value of B-spline coefficients
        model_output = self._problem.evaluate(
            betas)  # the output of the model with be an array of size nTimes x nOutputs
        x, x_dot, rhs = np.split(model_output, 3, axis=1)  # we split the array into states, state derivs, and RHSs
        # compute the data fit
        volts_for_model = self._values[:,
                          1]  # we need to make sure that voltage is read at the times within ROI so we pass it in as part of values
        d_y = Thetas_ODE[-1] * np.prod(x, axis=1) * (volts_for_model - EK) - self._values[:, 0]
        data_fit_cost = np.transpose(d_y) @ d_y
        # compute the gradient matching cost
        d_deriv = (x_dot - rhs) ** 2
        integral_quad = sp.integrate.simpson(y=d_deriv, even='avg', axis=0)
        gradient_match_cost = np.sum(integral_quad, axis=0)
        # not the most elegant implementation because it just grabs global lambda
        return data_fit_cost + lambd * gradient_match_cost
## this one is not really used within outer criterion, how to define one without the problem/model?
# # define a class that outputs only b-spline surfaces for all segments - we dont actually evaluate this,
# but I am not sure how to define a criterion without a problem
betas_segment = [] # placeholder for the beta values from all segments
knots = [] # placeholder for the knots from all segments
param_scaling_coeff = []
class PinnOutput(pints.ForwardModel):
    # this model outputs the discrepancy to be used in a rectangle quadrature scheme
    def simulate(self, parameters, times):
        # make stacked domain and pass it through the model
        # make a stacked domain from the theta instance and the times
        thetas_tensor = pt.tensor(parameters).unsqueeze(-1) * param_scaling_coeff
        stacked_domain = stack_inputs(times, thetas_tensor)
        # get the true current
        with pt.no_grad():
            pinn_output = pinn(stacked_domain)
        pinn_current_at_point = observation_tensors(times, pinn_output, stacked_domain, "cpu")
        pinn_current_at_point = pinn_current_at_point / param_scaling_coeff
        return pinn_current_at_point

    def n_parameters(self):
        # Return the dimension of the parameter vector
        # return len(Thetas_ODE) # this does not work beacuse the thetas are given in a different file
        return 9

    def n_outputs(self):
        # Return the dimension of the output vector
        # return len(state_names) # careful as this needs state_names to be a global variable
        return 1

# define an error w.r.t. the ODE parameters that assumes that it knows B-spline parameters - simply data fit
state_all_segments = [] # placeholder for the state values from all segments
current_pinn = [] # placeholder for the current values from the pinn model

class OuterCriterion(pints.ProblemErrorMeasure):
    # do I need to redefine custom init or can just drop this part?
    def __init__(self, problem, weights=None):
        super(OuterCriterion, self).__init__(problem)
        if weights is None:
            weights = [1] * self._n_outputs
        elif self._n_outputs != len(weights):
            raise ValueError(
                'Number of weights must match number of problem outputs.')
        # Check weights
        self._weights = np.asarray([float(w) for w in weights])

    # this function is the function of theta - ODE parameters
    def __call__(self, theta):
        # evaluate the integral at the value of ODE parameters
        # compute the data fit
        residuals_data = np.transpose(current_pinn) - self._values  # this part depends on theta_g
        data_fit_cost = np.sum((residuals_data) ** 2)
        return data_fit_cost

class OuterCriterionNoModel(pints.ProblemErrorMeasure):
    # do I need to redefine custom init or can just drop this part?
    def __init__(self, problem, weights=None):
        super(OuterCriterionNoModel, self).__init__(problem)
        if weights is None:
            weights = [1] * self._n_outputs
        elif self._n_outputs != len(weights):
            raise ValueError(
                'Number of weights must match number of problem outputs.')
        # Check weights
        self._weights = np.asarray([float(w) for w in weights])

    # this function is the function of theta - ODE parameters
    def __call__(self, theta):
        # evaluate the integral at the value of ODE parameters
        # state_all_segments must be updated globally elsewhere
        # compute the data fit
        current_model = observation_direct_input(state_all_segments, self._values[:, 1], theta)
        d_y = current_model - self._values[:, 0]  # this part depends on theta_g
        data_fit_cost = np.transpose(d_y) @ d_y
        return data_fit_cost
## define a class that will be used to define boundaries for the parameters of a two-state model
class BoundariesTwoStates(pints.Boundaries):
    """
    Boundary constraints on the parameters for a two state variables

    """

    def __init__(self):

        super(BoundariesTwoStates, self).__init__()

        # Limits on p1-p4 for a signle gative variable model
        self.lower_alpha = 1e-7  # Kylie: 1e-7
        self.upper_alpha = 1e3  # Kylie: 1e3
        self.lower_beta = 1e-7  # Kylie: 1e-7
        self.upper_beta = 0.4  # Kylie: 0.4
        self.lower_conductance = 1e-4  # arbitrary
        self.upper_conductance = self.lower_conductance*100000 # 10

        # Lower and upper bounds for all parameters
        self.lower = [
            self.lower_alpha,
            self.lower_beta,
            self.lower_alpha,
            self.lower_beta,
            self.lower_alpha,
            self.lower_beta,
            self.lower_alpha,
            self.lower_beta,
            self.lower_conductance
        ]
        self.upper = [
            self.upper_alpha,
            self.upper_beta,
            self.upper_alpha,
            self.upper_beta,
            self.upper_alpha,
            self.upper_beta,
            self.upper_alpha,
            self.upper_beta,
            self.upper_conductance
        ]

        self.lower = np.array(self.lower)
        self.upper = np.array(self.upper)

        # Limits on maximum reaction rates
        self.rmin = 1.67e-5
        self.rmax = 1000

        # Voltages used to calculate maximum rates
        self.vmin = -120
        self.vmax = 60

    def n_parameters(self):
        return 9

    def check(self, transformed_parameters):

        debug = False

        # # check if parameters are sampled in log space
        # if InLogScale:
        #     # Transform parameters back to decimal space
        #     parameters = np.exp(transformed_parameters)
        # else:
        #     # leave as is
        #     parameters = transformed_parameters

        # Transform parameters back to decimal space
        parameters = np.exp(transformed_parameters)

        # Check parameter boundaries
        if np.any(parameters < self.lower):
            if debug:
                print('Lower')
            return False
        if np.any(parameters > self.upper):
            if debug:
                print('Upper')
            return False

        # Check maximum rate constants
        p1, p2, p3, p4, p5, p6, p7, p8 = parameters[:8]
        # Check positive signed rates
        r = p1 * np.exp(p2 * self.vmax)
        if r < self.rmin or r > self.rmax:
            if debug:
                print('r1')
            return False
        r = p5 * np.exp(p6 * self.vmax)
        if r < self.rmin or r > self.rmax:
            if debug:
                print('r2')
            return False

        # Check negative signed rates
        r = p3 * np.exp(-p4 * self.vmin)
        if r < self.rmin or r > self.rmax:
            if debug:
                print('r3')
            return False
        r = p7 * np.exp(-p8 * self.vmin)
        if r < self.rmin or r > self.rmax:
            if debug:
                print('r4')
            return False
        return True

class BoundariesTwoStatesWithInit(pints.Boundaries):
    """
    Boundary constraints on the parameters for a two state variables

    """

    def __init__(self):

        super(BoundariesTwoStatesWithInit, self).__init__()

        # Limits on p1-p4 for a signle gative variable model
        self.lower_alpha = 1e-7  # Kylie: 1e-7
        self.upper_alpha = 1e3  # Kylie: 1e3
        self.lower_beta = 1e-7  # Kylie: 1e-7
        self.upper_beta = 0.4  # Kylie: 0.4
        self.upper_init = 1
        self.lower_init = 1e-7
        self.lower_conductance = 1e-4  # arbitrary
        self.upper_conductance = self.lower_conductance*100000 # 10

        # Lower and upper bounds for all parameters
        self.lower = [
            self.lower_alpha,
            self.lower_beta,
            self.lower_alpha,
            self.lower_beta,
            self.lower_alpha,
            self.lower_beta,
            self.lower_alpha,
            self.lower_beta,
            self.lower_conductance,
            self.lower_init,
            self.lower_init
        ]
        self.upper = [
            self.upper_alpha,
            self.upper_beta,
            self.upper_alpha,
            self.upper_beta,
            self.upper_alpha,
            self.upper_beta,
            self.upper_alpha,
            self.upper_beta,
            self.upper_conductance,
            self.upper_init,
            self.upper_init
        ]

        self.lower = np.array(self.lower)
        self.upper = np.array(self.upper)

        # Limits on maximum reaction rates
        self.rmin = 1.67e-5
        self.rmax = 1000

        # Voltages used to calculate maximum rates
        self.vmin = -120
        self.vmax = 60

    def n_parameters(self):
        return 11

    def check(self, transformed_parameters):

        debug = False

        # # check if parameters are sampled in log space
        # if InLogScale:
        #     # Transform parameters back to decimal space
        #     parameters = np.exp(transformed_parameters)
        # else:
        #     # leave as is
        #     parameters = transformed_parameters

        # Transform parameters back to decimal space
        parameters = np.exp(transformed_parameters)

        # Check parameter boundaries
        if np.any(parameters < self.lower):
            if debug:
                print('Lower')
            return False
        if np.any(parameters > self.upper):
            if debug:
                print('Upper')
            return False

        # Check maximum rate constants
        p1, p2, p3, p4, p5, p6, p7, p8 = parameters[:8]
        # Check positive signed rates
        r = p1 * np.exp(p2 * self.vmax)
        if r < self.rmin or r > self.rmax:
            if debug:
                print('r1')
            return False
        r = p5 * np.exp(p6 * self.vmax)
        if r < self.rmin or r > self.rmax:
            if debug:
                print('r2')
            return False

        # Check negative signed rates
        r = p3 * np.exp(-p4 * self.vmin)
        if r < self.rmin or r > self.rmax:
            if debug:
                print('r3')
            return False
        r = p7 * np.exp(-p8 * self.vmin)
        if r < self.rmin or r > self.rmax:
            if debug:
                print('r4')
            return False
        return True

# main
if __name__ == '__main__':
    # test the pints classes
    nOutputs = 2
    inLogScale = True
    lambd = 1000
    ####################################################################################################################
    # set up variables for the simulation
    figureFolder = direcory_names["figures"]
    # test generating data for multiple segments
    # select the model that we will use to generate the synthetic data
    load_protocols  # this module will load the voltage protocol ad give times of interest so we dont have to generate it again
    ## define the time interval on which the fitting will be done
    tlim = [300, 14899]
    times = np.linspace(*tlim, tlim[-1] - tlim[0], endpoint=False)
    del tlim
    # generate the segments with B-spline knots and intialise the betas for splines
    jump_indeces, times_roi, voltage_roi, knots_roi, collocation_roi, spline_order = generate_knots(times)
    nSegments = len(jump_indeces[:-1])
    print('Inner optimisation is split into ' + str(nSegments) + ' segments based on protocol steps.')
    state_names = ['a', 'r']  # how many states we have in the model that we are fitting
    nBsplineCoeffs = (len(knots_roi[0]) - spline_order - 1) * len(state_names)
    init_betas_roi = nSegments * [0.5 * np.ones(nBsplineCoeffs)]
    print('Number of B-spline coeffs per segment: ' + str(nBsplineCoeffs) + '.')
    ####################################################################################################################
    # parameters needed to generate synthetic data
    model_name = 'wang'
    if model_name.lower() not in available_models:
        raise ValueError(f'Unknown model name: {model_name}. Available models are: {available_models}.')
    elif model_name.lower() == 'hh':
        thetas_true = thetas_hh_baseline
    elif model_name.lower() == 'kemp':
        thetas_true = thetas_kemp
    elif model_name.lower() == 'wang':
        thetas_true = thetas_wang
    solution, current_model = generate_synthetic_data(model_name, thetas_true, times)
    true_states = solution.sol(times)
    nThetas = len(thetas_true)
    # set signal to noise ratio in decibels
    snr_db = 30
    snr = 10 ** (snr_db / 10)
    current_test = current_model(times, solution, thetas_true, snr=snr)
    # # test generating niseless data
    # current_test_noiseless = current_model(times, solution, thetas_true)
    ####################################################################################################################
    ## create multiple segments limited by time instances of jumps
    states_roi, states_known_roi, current_roi = split_generated_data_into_segments(solution, current_test, jump_indeces,
                                                                                   times)
    # create a current_true list out of current_roi - we will need it for the outer optimisation
    # use list comprehension but remember to remove the end point from each segment - we needed them for continuity in segments
    current_true = [item for sublist in current_roi for item in sublist[:-1]]
    # add the last point from the last segment
    current_true += [current_roi[-1][-1]]
    print('Produced synthetic data for the ' + model_name + ' model based on the pre-loaded voltage protocol.')
    ####################################################################################################################
    # set up boundaries for thetas
    ## rectangular boundaries of thetas from Clerx et.al. paper - they are the same for two gating variables + one for conductance
    theta_lower_boundary = [np.log(10e-5), np.log(10e-5), np.log(10e-5), np.log(10e-5), np.log(110e-5), np.log(10e-5),
                            np.log(10e-5), np.log(10e-5), np.log(10e-3)]
    theta_upper_boundary = [np.log(10e3), np.log(0.4), np.log(10e3), np.log(0.4), np.log(10e3), np.log(0.4),
                            np.log(10e3), np.log(0.4), np.log(10)]
    if inLogScale:
        # theta in log scale
        init_thetas = np.log(thetas_hh_baseline) # start around the true solution to see how long it takes to converge
        sigma0_thetas = 0.1 * np.ones(nThetas)
        boundaries_thetas = pints.RectangularBoundaries(theta_lower_boundary, theta_upper_boundary)
        boundaries_thetas_Michael = BoundariesTwoStates()
    else:
        # theta in decimal scale
        init_thetas = 0.001 * np.ones(nThetas)
        sigma0_thetas = 0.0005 * np.ones(nThetas)
        boundaries_thetas = pints.RectangularBoundaries(np.exp(theta_lower_boundary), np.exp(theta_upper_boundary))
    Thetas_ODE = init_thetas.copy()
    ####################################################################################################################
    # example of creating models and related optimiseers in pints
    model_bsplines_test = bsplineOutput()
    model_segments = PinnOutput()
    ## create the problem of comparing the modelled current with measured current
    voltage = V(times)  # must read voltage at the correct times to match the output
    values_to_match_output_ode = np.transpose(np.array([current_true, voltage]))
    # ^ we actually only need first two columns in this array but pints wants to have the same number of values and outputs
    problem_outer = pints.MultiOutputProblem(model=model_segments, times=times,
                                             values=current_true)
    ## associate the cost with it
    error_outer = OuterCriterion(problem=problem_outer)
    ####################################################################################################################
    # take 1: loosely based on ask-tell example from  pints
    convergence_threshold = 1e-8
    iter_for_convergence = 20
    max_iter = 2000
    # Create an outer optimisation object
    # optimiser_outer = pints.CMAES(x0=init_thetas,sigma0=sigma0_thetas, boundaries=boundaries_thetas) # with simple rectangular boundaries
    optimiser_outer = pints.CMAES(x0=init_thetas, sigma0=sigma0_thetas, boundaries=boundaries_thetas_Michael) # with boundaries accounting for the reaction rates
    optimiser_outer.set_population_size(min(len(Thetas_ODE)*7,30))

