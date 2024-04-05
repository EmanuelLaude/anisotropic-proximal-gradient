import numpy as np

import optim.function as fun
import optim.optimizer as optim

from abc import ABC, abstractmethod


##
# Optimizers for minimizing Loss(Ax - b) + <c, x> + nu/2 |x|^2
##

thres = 15.
fun.thres = thres

def sumexp(x):
    return np.sum(np.exp(x))

def logistic(y):
    v = y.copy()
    v[y <= thres] = np.log(1 + np.exp(y[y <= thres]))
    return v

class Parameters:
    def __init__(self, maxit, tol, alpha=0.5, method=None, pi=1.2, eps=1e-15, Wolfe = True, mem = 200, initialization_procedure=0,
                 sigma=1e-4, eta=0.9, gamma_init=0, force_use_gamma_init=False):
        self.maxit = maxit
        self.gamma_init = gamma_init
        self.tol = tol
        self.alpha = alpha
        self.method = method
        self.pi = pi
        self.eps = eps
        self.Wolfe = Wolfe
        self.mem = mem
        self.initialization_procedure = initialization_procedure
        self.sigma = sigma
        self.eta = eta
        self.force_use_gamma_init = force_use_gamma_init

class OptimizerBaseClass(ABC):
    def __init__(self, loss, linear_transform, b, c, nu, x_init, callback, params):
        self.x = np.copy(x_init)
        self.loss = loss
        self.linear_transform = linear_transform
        self.c = c
        self.b = b
        self.callback = callback
        self.nu = nu
        self.params = params

        self.diffable = fun.AdditiveComposite(
            (
                fun.AffineCompositeLoss(self.loss, linear_transform, self.b),
                fun.Linear(self.c),
                fun.NormPower(2, 2, nu)
             )
        )

    @abstractmethod
    def run(self):
        pass

class ExponentialProximalGradientMethod(OptimizerBaseClass):
    evals_per_iteration = 2
    evals_per_linesearch = 1

    def __init__(self, loss, linear_transform, b, c, nu, x_init, callback, params):
        super().__init__(loss, linear_transform, b, c, nu, x_init, callback, params)

        self.Aplus = np.zeros(linear_transform._A.shape)
        self.Aminus = np.zeros(linear_transform._A.shape)

        self.Aplus[linear_transform._A >= 0] = linear_transform._A[linear_transform._A >= 0]
        self.Aminus[linear_transform._A <= 0] = -linear_transform._A[linear_transform._A <= 0]

        self.Aplus = self.Aplus
        self.Aminus = self.Aminus

        self.cplus = np.zeros(c.shape)
        self.cminus = np.zeros(c.shape)

        self.cplus[c >= 0] = c[c >= 0]
        self.cminus[c <= 0] = -c[c <= 0]

        self.cplus = self.cplus + 1e-7
        self.cminus = self.cminus + 1e-7

        self.L = 0
        m, _ = linear_transform._A.shape
        for i in range(m):
            self.L = np.maximum(self.L, np.sum(np.abs(self.linear_transform._A[i, :])))

    def run(self):
        gamma_min = 1 / self.L

        gamma = gamma_min
        cum_num_backtracks = 0
        for k in range(self.params.maxit):
            grad = self.diffable.eval_gradient(self.x)
            res = np.linalg.norm(grad)

            if self.callback(k, cum_num_backtracks, gamma, self.x, res):
                break

            if res <= self.params.tol:
                break

            pred = np.dot(self.linear_transform._A, self.x) - self.b
            v = self.loss.eval_gradient(pred)

            delta_plus = np.log(np.dot(self.Aplus.T, v) + self.cplus + self.nu * logistic(self.x))
            delta_minus = np.log(np.dot(self.Aminus.T, v) + self.cminus + self.nu * logistic(-self.x))

            objective_value = self.diffable.eval(self.x)

            while True:
                gamma = np.maximum(gamma_min, gamma)
                x_new = self.x - gamma * (delta_plus - delta_minus) / 2
                if gamma == gamma_min or self.params.alpha <= 0.:
                    break

                cum_num_backtracks += 1

                if (self.diffable.eval(x_new) <= objective_value
                        + gamma * (2 * sumexp(0.5 * (delta_plus + delta_minus)) - sumexp(delta_plus) - sumexp(delta_minus))) + self.params.eps:
                    break

                gamma = gamma * self.params.alpha

            if self.params.alpha > 0.:
                gamma = gamma / self.params.alpha
            self.x = x_new


class OptimWrapper(OptimizerBaseClass):
    def __init__(self, loss, linear_transform, b, c, nu, x_init, callback, params):
        super().__init__(loss, linear_transform, b, c, nu, x_init, callback, params)

        self.proxable = fun.Zero()
        self.problem = optim.CompositeOptimizationProblem(x_init, self.diffable, self.proxable)


        if params.gamma_init <= 0.:
            gamma_init = 1.99 / self.diffable.get_Lip_gradient()

            if gamma_init <= 0.:
                gamma_init = 1.0
        else:
            gamma_init = params.gamma_init

        params_optim = optim.Parameters(
            alpha = params.alpha,
            Wolfe = params.Wolfe,
            mem = params.mem,
            sigma = params.sigma,
            eta = params.eta,
            epsilon=params.eps,
            maxit=params.maxit,
            tol=params.tol,
            pi=params.pi,
            initialization_procedure=params.initialization_procedure,
            gamma_init=gamma_init
        )
        self.algorithm = params.method(params_optim, self.problem, callback)

    def run(self):
        self.algorithm.run()
