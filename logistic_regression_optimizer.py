import numpy as np

import optim.function as fun
import optim.optimizer as optim

from abc import ABC, abstractmethod


##
# Optimizers for minimizing Loss(Ax - b) + <c, x> + nu/2 |x|^2
##

thres = 15.
fun.thres = thres

def phi(x):
    return np.sum(2 * np.log(1 + np.exp(x)) - x)

#def grad_symm_logistic(x):
#    return 2 * np.exp(x) / (1 + np.exp(x)) - 1

def conj_phi(x):
    return np.sum((x + 1) * np.log(x/2 + 0.5) + (1 - x) * np.log(0.5 - x/2))
    #return np.sum((x + K_min) * np.log(x + K_min) + (K_max - x) * np.log(K_max - x))

def grad_conj_phi(x):
    return np.log(x + 1) - np.log(1 - x)
    #return np.log(x + K_min) - np.log(K_max - x)

def euclidean_prox_logistic(v, rho):
    #Initial guess based on piecewise approximation.
    # if v < -2.5:
    #     x = v
    # elif v > 2.5 + 1 / rho:
    #     x = v - 1 / rho
    # else:
    #     x = (rho * v - 0.5) / (0.2 + rho)

    x = (rho * v[:] - 0.5) / (0.2 + rho)
    x[v > 2.5 + 1 / rho] = v[v > 2.5 + 1 / rho] - 1 / rho
    x[v < -2.5] = v[v < -2.5]

    #Newton iteration.
    l = v[:] - 1 / rho
    u = v[:] + 0.0

    for i in range(500):
        inv_ex = 1 / (1 + np.exp(-x[:]))
        f = inv_ex[:] + rho * (x[:] - v[:])
        g = inv_ex[:] * (1 - inv_ex[:]) + rho

        # if f < 0:
        #     l = x
        # else:
        #     u = x
        #     x = x - f / g
        #     x = np.minimum(x, u)
        #     x = np.maximum(x, l)

        l[f < 0] = x[f < 0]

        u[f >= 0] = x[f >= 0]
        x[f >= 0] = x[f >= 0] - f[f >= 0] / g[f >= 0]
        x[f >= 0] = np.minimum(x[f >= 0], u[f >= 0])
        x[f >= 0] = np.maximum(x[f >= 0], l[f >= 0])




    #Guarded method if not converged.
    for i in range(500):
        if np.amax(u - l) < 1e-15:
            break
        g_rho = 1 / (rho * (1 + np.exp(-x[:]))) + (x[:] - v[:])
        # if g_rho > 0:
        #     l = np.maximum(l, x - g_rho)
        #     u = x
        # else:
        #     u = np.minimum(u, x - g_rho)
        #     l = x

        l[g_rho > 0] = np.maximum(l[g_rho > 0], x[g_rho > 0] - g_rho[g_rho > 0])
        u[g_rho > 0] = x[g_rho > 0]
        u[g_rho <= 0] = np.minimum(u[g_rho <= 0], x[g_rho <= 0] - g_rho[g_rho <= 0])
        l[g_rho <= 0] = x[g_rho <= 0]

        x[:] = (u[:] + l[:]) / 2

    return x

class AnisotropicProxable(ABC):
    def __init__(self, weight):
        self.weight = weight

    @abstractmethod
    def eval(self, x):
        pass

    @abstractmethod
    def eval_anisotropic_prox(self, x, gamma):
        pass

class Ell1Norm(AnisotropicProxable):
    def __init__(self, weight):
        self.weight = weight

    def eval(self, x):
        return self.weight * np.sum(np.abs(x))

    def eval_anisotropic_prox(self, x, gamma):
        rho = grad_conj_phi(self.weight)
        z = np.zeros(x.shape)
        z[x >= gamma * rho] = x[x >= gamma * rho] - gamma * rho
        z[x <= -gamma * rho] = x[x <= -gamma * rho] + gamma * rho

        return z

class SquaredEll2Norm(AnisotropicProxable):
    def __init__(self, weight):
        self.weight = weight

    def eval(self, x):
        return self.weight * np.dot(x, x)

    def eval_anisotropic_prox(self, x, gamma):
        t = euclidean_prox_logistic(-x / gamma + 1 / (self.weight * gamma), self.weight / 2 * gamma)
        return x + t * gamma


class Parameters:
    def __init__(self, maxit, tol, alpha=0.5, method=None, pi=1.2, eps=1e-15, Wolfe = True, mem = 200, initialization_procedure=0,
                 sigma=1e-4,
                 eta=0.9, gamma_init=0.):
        self.maxit = maxit
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
        self.gamma_init = gamma_init

class OptimizerBaseClass(ABC):
    def __init__(self, loss, nu, reg, linear_transform, x_init, callback, params):
        self.x = np.copy(x_init)
        self.loss = loss
        self.nu = nu
        self.reg = reg
        self.linear_transform = linear_transform
        self.callback = callback
        self.params = params

        self.diffable = fun.AffineCompositeLoss(loss, A=linear_transform)

    @abstractmethod
    def run(self):
        pass

class AnisotropicProximalGradientMethod(OptimizerBaseClass):
    evals_per_iteration = 2
    evals_per_linesearch = 1

    def __init__(self, loss, nu, reg, linear_transform, x_init, callback, params):
        super().__init__(loss, nu, reg, linear_transform, x_init, callback, params)

        assert(reg == 1. or reg == 2.)

        if reg == 1.:
            self.proxable = Ell1Norm(nu)
        elif reg == 2.:
            self.proxable = SquaredEll2Norm(nu)

        self.L = 0
        m, _ = linear_transform._A.shape
        for i in range(m):
            self.L = np.maximum(self.L, np.linalg.norm(self.linear_transform._A[i, :], 2))

    def run(self):
        gamma_min = 1 / self.L

        gamma = gamma_min
        cum_num_backtracks = 0
        res = np.Inf
        for k in range(self.params.maxit):
            if self.callback(k, cum_num_backtracks, gamma, self.x, res):
                break

            if res <= self.params.tol:
                break


            grad = self.diffable.eval_gradient(self.x)
            prec_grad = grad_conj_phi(grad)

            value = self.diffable.eval(self.x)

            while True:
                gamma = np.maximum(gamma_min, gamma)
                x_new = self.proxable.eval_anisotropic_prox(self.x - gamma * prec_grad, gamma)
                if gamma == gamma_min or self.params.alpha <= 0.:
                    break

                cum_num_backtracks += 1

                if (self.diffable.eval(x_new) <= value
                        + gamma * (phi((x_new - self.x) / gamma + prec_grad) - phi(prec_grad)) + self.params.eps):
                    break

                gamma = gamma * self.params.alpha

            if self.params.alpha > 0.:
                gamma = gamma / self.params.alpha

            res = np.linalg.norm(self.x - x_new) / gamma
            self.x = x_new

class OptimWrapper(OptimizerBaseClass):
    def __init__(self, loss, nu, reg, linear_transform, x_init, callback, params):
        super().__init__(loss, nu, reg, linear_transform, x_init, callback, params)

        assert (reg == 1. or reg == 2.)

        if reg == 1.:
            self.proxable = fun.OneNorm(nu)
        elif reg == 2.:
            self.diffable = fun.AdditiveComposite(
                (
                    self.diffable,
                    fun.NormPower(2, 2, nu)
                )
            )
            self.proxable = fun.Zero()
        self.problem = optim.CompositeOptimizationProblem(x_init, self.diffable, self.proxable)


        if params.gamma_init <= 0.:
            gamma_init = 1.99 / self.diffable.get_Lip_gradient()
            if issubclass(params.method, optim.LineSearchDescentMethodBaseClass):
                gamma_init = 1.0
        else:
            gamma_init = params.gamma_init

        params_optim = optim.Parameters(
            alpha=params.alpha,
            Wolfe=params.Wolfe,
            mem=params.mem,
            sigma=params.sigma,
            eta=params.eta,
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

