import numpy as np
#===========================================================================
# Linear optimization problem class
#===========================================================================
class Parameters():

	#-----------------------------------------------------------------------
	# Initialize the problem
	#-----------------------------------------------------------------------
	def __init__(self):

		#-------------------------------------------------------------------
		# Default values of problem generator
		#-------------------------------------------------------------------
		self.Problem_Type 		= "LO"
		self.norm_x 			= 5
		self.norm_s 			= 5
		self.norm_y 			= 10
		self.norm_b 			= -1
		self.norm_c 			= -1
		self.norm_A				= 1

		self.decimals			= 3
		self.condition_number	= 2

		self.seed 				= 6261997

		self.has_interior		= True
		self.has_optimal		= True

		self.make_psd			= False
		self.do_print			= False
		self.qlsa_print			= False
		self.symmetry			= False

		self.time_limit 		= 100


		#-------------------------------------------------------------------
		# Default values of paramters
		#-------------------------------------------------------------------
		self.Method 			= "II-IPM"
		self.LS_Method 			= "LS"
		self.Is_Quantum 		= False
		self.Is_Simulator 		= True
		self.Is_Noisy			= False

		self.LS_Precision 		= 1e-1
		self.IR_LS_Precision 	= 1e-8

		self.IR_Precision 		= 1e-10
		self.LO_Precision 		= 1e-8
		self.Stop_Precision 	= 1e-16					# If the step-length becomes less than the algorithm stops
		self.Stop_Cond_Num		= 1e3

		self.IR_Verbosity 		= 2
		self.LO_Verbosity 		= 1

		self.num_ancillae 		= 3
		self.num_time_slices	= 1
		self.expansion_order	= 2
		self.HHL_Method 		= 1
		self.qlsa_precision  	= 1e0




		#-------------------------------------------------------------------
		# Default values of inexact_infeasible_IPM paramters
		#-------------------------------------------------------------------
		self.Beta_1 			= 0.1
		self.Beta_2 			= 1 - 5e-4

		self.Omega 				= 1e8
		self.Gamma 				= 0.5

		self.AlphaHatDec 		= 1 - 1e-3


		#-------------------------------------------------------------------
		# Default values of Iterative Refinement algorithm paramters
		#-------------------------------------------------------------------
		self.ScalFact 			= 1e2							# Scaling factor
		self.IncScalLim 		= 10  							# Incremental scaling limit


		self.LS_ScalFact 		= 1								# Scaling factor
		self.LS_IncScalLim 		= 2  							# Incremental scaling limit



# ===========================================================================
# Linear optimization problem with a known optimal solution
# ===========================================================================
def generate_lo_problem_with_opt(m, n, parameters):
    np.random.seed(parameters.seed)

    mask = [1 if ind < m else 0 for ind in range(n)]
    np.random.shuffle(mask)

    opt_x = np.random.rand(n)
    opt_x = np.multiply(opt_x, mask)
    opt_s = np.random.rand(n)
    opt_s = opt_s - np.multiply(opt_s, mask)
    opt_y = np.random.rand(m) - 0.5
    opt_x = (parameters.norm_x / np.linalg.norm(opt_x)) * opt_x
    opt_s = (parameters.norm_s / np.linalg.norm(opt_s)) * opt_s
    opt_y = (parameters.norm_y / np.linalg.norm(opt_y)) * opt_y

    A = np.random.rand(m, n) - 0.5
    A = np.matmul(A, A.T) if parameters.make_psd == True else A

    u, s, v = np.linalg.svd(A, full_matrices=False)
    s = np.linspace(parameters.norm_A, parameters.norm_A / parameters.condition_number, min(m, n))

    A = np.dot(u * s, v) if parameters.symmetry == False else np.dot(u * s, u.T)

    b = np.matmul(A, opt_x)
    c = np.matmul(A.T, opt_y) + opt_s

    if parameters.norm_b != -1:
        temp_norm_b = np.linalg.norm(b)
        coef = parameters.norm_b / temp_norm_b
        opt_x = coef * opt_x
        b = coef * b

    results = (A, b, c, opt_x, opt_y, opt_s)

    return results

# ===========================================================================
# Linear optimization problem with a known interior solution
# ===========================================================================
def generate_lo_problem_with_int(m, n, parameters):
    np.random.seed(parameters.seed)

    int_x = np.random.rand(n)
    int_s = np.random.rand(n)
    int_y = np.random.rand(m) - 0.5

    int_x = (parameters.norm_x / np.linalg.norm(int_x)) * int_x
    int_s = (parameters.norm_s / np.linalg.norm(int_s)) * int_s
    int_y = (parameters.norm_y / np.linalg.norm(int_y)) * int_y

    A = np.random.rand(m, n) - .5
    A = np.matmul(A, A.T) if parameters.make_psd == True else A

    u, s, v = np.linalg.svd(A, full_matrices=False)
    s = np.linspace(parameters.norm_A, parameters.norm_A / parameters.condition_number, min(m, n))

    A = np.dot(u * s, v) if parameters.symmetry == False else np.dot(u * s, u.T)

    b = np.matmul(A, int_x)
    c = np.matmul(A.T, int_y) + int_s

    if parameters.norm_b != -1:
        temp_norm_b = np.linalg.norm(b)
        coef = parameters.norm_b / temp_norm_b
        int_x = coef * int_x
        b = coef * b

    results = (A, b, c, int_x, int_y, int_s)

    return results


# ===========================================================================
# Linear optimization problem with known interior and optimal solutions
# ===========================================================================
def generate_lo_problem_with_int_and_opt(m, n, parameters):
    np.random.seed(parameters.seed)

    mask = [1 if ind < m else 0 for ind in range(n)]
    np.random.shuffle(mask)

    opt_x = np.random.rand(n)
    opt_x = np.multiply(opt_x, mask)
    opt_s = np.random.rand(n)
    opt_s = opt_s - np.multiply(opt_s, mask)
    opt_y = np.random.rand(m) - 0.5
    A = np.random.rand(m, n) - 0.5

    int_x = np.random.rand(n + 1)
    int_s = np.random.rand(n + 1)
    int_y = np.random.rand(m + 1) - 0.5
    int_y[m] = 0.5

    delta = np.dot((int_x[:n] - opt_x).T, (int_s[:n] - opt_s))
    de = max(0, -delta / int_x[n])
    int_s[n] = de if int_s[n] <= de else int_s[n]

    opt_x = np.append(opt_x, 0)
    opt_s = np.append(opt_s, (delta / int_x[n]) + int_s[n])
    opt_y = np.append(opt_y, 0)

    opt_x = (parameters.norm_x / np.linalg.norm(opt_x)) * opt_x
    opt_s = (parameters.norm_s / np.linalg.norm(opt_s)) * opt_s
    opt_y = (parameters.norm_y / np.linalg.norm(opt_y)) * opt_y

    int_x = (parameters.norm_x / np.linalg.norm(int_x)) * int_x
    int_s = (parameters.norm_s / np.linalg.norm(int_s)) * int_s
    int_y = (parameters.norm_y / np.linalg.norm(int_y)) * int_y

    ahat = (1 / int_x[n]) * (np.dot(A, (opt_x[:n] - int_x[:n])))
    dhat = (1 / int_y[m]) * (np.dot(A.T, (opt_y[:m] - int_y[:m])) + opt_s[:n] - int_s[:n])
    dlast = (1 / int_x[n]) * (np.dot(dhat, (opt_x[:n] - int_x[:n])))
    AA = np.block([[A, np.matrix(ahat).T], [np.matrix(dhat), np.matrix(dlast)]])

    AA = np.matmul(AA, AA.T) if parameters.make_psd == True else AA

    u, s, v = np.linalg.svd(A, full_matrices=False)
    s = np.linspace(parameters.norm_A, parameters.norm_A / parameters.condition_number, min(m, n))

    AA = np.dot(u * s, v) if parameters.symmetry == False else np.dot(u * s, u.T)

    b = np.matmul(AA, opt_x)
    c = np.matmul(AA.T, opt_y) + opt_s

    if parameters.norm_b != -1:
        temp_norm_b = np.linalg.norm(b)
        coef = parameters.norm_b / temp_norm_b
        opt_x = coef * opt_x
        int_x = coef * int_x
        b = coef * b

    results = (A, b, c, opt_x, opt_y, opt_s, int_x, int_y, int_s)

    return results
