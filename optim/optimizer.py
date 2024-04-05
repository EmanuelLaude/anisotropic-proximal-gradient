from abc import ABC, abstractmethod
import numpy as np


class OptimizationProblem(ABC):
    def __init__(self, x_init):
        self.x_init = x_init

    @abstractmethod
    def eval_objective(self, x):
        pass


class CompositeOptimizationProblem(OptimizationProblem):
    def __init__(self, x_init, diffable, proxable):
        super().__init__(x_init)
        self.diffable = diffable
        self.proxable = proxable

    def eval_objective(self, x):
        return self.diffable.eval(x) + self.proxable.eval(x)

class DiffableOptimizationProblem(OptimizationProblem):
    def __init__(self, x_init, diffable):
        super().__init__(x_init)
        self.diffable = diffable

    def eval_objective(self, x):
        return self.diffable.eval(x)

class Parameters:
    def __init__(self, pi = 1., epsilon = 1e-12, gamma_init = 1., maxit = 500, tol = 1e-5, initialization_procedure = 1
                 ,Gamma_init = 1., alpha = 0., Wolfe = True, mem = 200, sigma = 1e-4, eta = 0.9):
        self.maxit = maxit
        self.tol = tol
        self.gamma_init = gamma_init
        self.epsilon = epsilon
        self.pi = pi
        self.initialization_procedure = initialization_procedure
        self.Gamma_init = Gamma_init
        self.alpha = alpha

        # Quasi-Newton stuff
        self.mem = mem
        self.Wolfe = Wolfe
        self.sigma = sigma
        self.eta = eta




class Optimizer(ABC):
    def __init__(self, params, problem, callback = None):
        self.params = params
        self.problem = problem
        self.callback = callback

    @abstractmethod
    def run(self):
        pass

class LineSearchDescentMethodBaseClass(Optimizer):
    def __init__(self, params, problem, callback = None):
        super().__init__(params, problem, callback)
        self.iter = 0
        self.x = np.zeros(problem.x_init.shape)
        self.x[:] = problem.x_init[:]

    @abstractmethod
    def get_descent_direct(self):
        pass

    @abstractmethod
    def post_update(self, x):
        pass

    def run(self):
        sigma = self.params.sigma
        eta = self.params.eta

        cum_num_backtracks = 0
        gamma = self.params.gamma_init
        for k in range(self.params.maxit):
            fx, self.grad, d = self.get_descent_direct()

            res = np.linalg.norm(self.grad)

            if self.callback(k, cum_num_backtracks, gamma, self.x, res):
                break

            if res <= self.params.tol:
                break

            direc_deriv = np.dot(self.grad, d)

            gamma = self.params.gamma_init
            gamma_low = 0
            gamma_high = np.inf
            while True:
                cum_num_backtracks += 1

                fx_plus = self.problem.diffable.eval(self.x + gamma * d)
                grad_plus = self.problem.diffable.eval_gradient(self.x + gamma * d)

                if fx_plus > fx + sigma * gamma * direc_deriv + self.params.epsilon:
                    gamma_high = gamma
                    gamma = 0.5 * (gamma_low + gamma_high)
                elif self.params.Wolfe and np.dot(grad_plus, d) < eta * np.dot(self.grad, d) - self.params.epsilon:
                    gamma_low = gamma
                    if gamma_high == np.inf:
                        gamma = 2 * gamma_low
                    else:
                        gamma = 0.5 * (gamma_low + gamma_high)
                else:
                    break

            x = self.x + gamma * d

            self.post_update(x)
            self.iter = self.iter + 1
            self.x[:] = x[:]

class LineSearchGradientDescent(LineSearchDescentMethodBaseClass):
    evals_per_iteration = 2
    evals_per_linesearch = 1

    def __init__(self, params, problem, callback = None):
        super().__init__(params, problem, callback)

    def get_descent_direct(self):
        fx = self.problem.diffable.eval(self.x)
        grad = self.problem.diffable.eval_gradient(self.x)
        d = -grad
        return fx, grad, d

    def post_update(self, x):
        pass


class BFGS(LineSearchDescentMethodBaseClass):
    evals_per_iteration = 2
    evals_per_linesearch = 1

    def __init__(self, params, problem, callback = None):
        super().__init__(params, problem, callback)

        self.H = np.identity(self.x.shape[0])

    def get_descent_direct(self):
        fx = self.problem.diffable.eval(self.x)
        grad = self.problem.diffable.eval_gradient(self.x)
        d = -np.dot(self.H, grad)
        return fx, grad, d

    def post_update(self, x):
        s = x - self.x
        y = self.problem.diffable.eval_gradient(x) - self.grad

        rho = 1 / np.dot(y, s)

        V = np.identity(self.x.shape[0]) - rho * np.outer(y, s)
        self.H = np.dot(np.dot(V.T, self.H), V) + rho * np.outer(s, s)


class LBFGS(LineSearchDescentMethodBaseClass):
    evals_per_iteration = 2
    evals_per_linesearch = 1

    def __init__(self, params, problem, callback = None):
        super().__init__(params, problem, callback)

        self.S = []
        self.Y = []

    def get_descent_direct(self):
        fx = self.problem.diffable.eval(self.x)
        grad = self.problem.diffable.eval_gradient(self.x)

        q = grad

        alpha = [0.] * len(self.S)
        rho = [0.] * len(self.S)
        for j in reversed(range(len(self.S))):
            rho[j] = 1 / np.dot(self.Y[j], self.S[j])
            alpha[j] = rho[j] * np.dot(self.S[j], q)

            q = q - alpha[j] * self.Y[j]

        if self.iter > len(self.S):
            H = (np.dot(self.S[-1], self.Y[-1]) / np.dot(self.Y[-1], self.Y[-1])) * np.identity(self.x.shape[0])
        else:
            H = np.identity(self.x.shape[0])

        d = np.dot(H, q)
        for j in range(len(self.S)):
            beta = rho[j] * np.dot(self.Y[j], d)
            d = d + (alpha[j] - beta) * self.S[j]

        return fx, grad, -d

    def post_update(self, x):
        if len(self.S) >= self.params.mem:
            self.S.pop(0)
            self.Y.pop(0)

        self.S.append(x - self.x)
        self.Y.append(self.problem.diffable.eval_gradient(x) - self.grad)


class ProximalGradientDescent(Optimizer):
    evals_per_iteration = 2
    evals_per_linesearch = 0

    def __init__(self, params, problem, callback = None):
        super().__init__(params, problem, callback)

        self.x = np.zeros(problem.x_init.shape)
        self.x[:] = problem.x_init[:]


    def run(self):
        gamma = 1.99 / self.problem.diffable.get_Lip_gradient()

        for k in range(self.params.maxit):
            grad = self.problem.diffable.eval_gradient(self.x)
            res = np.linalg.norm(grad)

            if self.callback(k, 0, gamma, self.x, res):
                break

            if res <= self.params.tol:
                break

            self.x = self.problem.proxable.eval_prox(self.x - gamma * grad, gamma)

class LineSearchProximalGradientDescent(Optimizer):
    evals_per_iteration = 2
    evals_per_linesearch = 1

    def __init__(self, params, problem, callback = None):
        super().__init__(params, problem, callback)

        self.x = np.zeros(problem.x_init.shape)
        self.x[:] = problem.x_init[:]


    def run(self):
        cum_num_backtracks = 0

        gamma_min = 1.99 / self.problem.diffable.get_Lip_gradient()

        if self.params.gamma_init > 0:
            gamma = self.params.gamma_init
        else:
            gamma = self.params.gamma_min

        for k in range(self.params.maxit):
            grad = self.problem.diffable.eval_gradient(self.x)
            res = np.linalg.norm(grad)

            if self.callback(k, cum_num_backtracks, gamma, self.x, res):
                break

            if res <= self.params.tol:
                break

            value = self.problem.diffable.eval(self.x)

            while True:
                gamma = np.maximum(gamma_min, gamma)
                x_new = self.problem.proxable.eval_prox(self.x - gamma * grad, gamma)
                if gamma == gamma_min or self.params.alpha <= 0.:
                    break

                cum_num_backtracks += 1

                if self.problem.diffable.eval(x_new) <= value + np.dot(grad, x_new - self.x) + (0.5 / gamma) * np.dot(x_new - self.x, x_new - self.x) + self.params.epsilon:
                    break

                gamma = gamma * self.params.alpha

            if self.params.alpha > 0.:
                gamma = gamma / self.params.alpha
            self.x = x_new



class AdaptiveProximalGradientMethod(Optimizer):
    evals_per_iteration = 2
    evals_per_linesearch = 0

    def __init__(self, params, problem, callback = None):
        super().__init__(params, problem, callback)

        self.x = np.zeros(problem.x_init.shape)
        self.x[:] = problem.x_init[:]

        self.grad = problem.diffable.eval_gradient(self.x)

        self.gamma = self.params.gamma_init


    def run(self):
        if self.params.initialization_procedure == 0:
            x = self.problem.proxable.eval_prox(self.x - self.gamma * self.grad, self.gamma)
            gamma = self.gamma
        else:
            x_new = self.problem.proxable.eval_prox(self.x - self.params.gamma_init * self.grad, self.params.gamma_init)
            grad_x_new = self.problem.diffable.eval_gradient(x_new)
            L = np.linalg.norm(self.grad - grad_x_new) / np.linalg.norm(self.x - x_new)

            if self.params.pi - 2 * L < 0:
                self.gamma = self.params.gamma_init
            else:
                self.gamma = self.params.gamma_init * (self.params.pi * 2 * L) / (self.params.pi - 2 * L)
            gamma = self.params.gamma_init
            x = np.copy(x_new)
        res = np.Inf
        for k in range(self.params.maxit):
            if self.callback(k, 0, gamma, self.x, res / gamma):
                break

            grad = self.problem.diffable.eval_gradient(x)
            res = np.linalg.norm(x - self.x)

            if res / gamma <= self.params.tol:
                break

            ell = np.dot(grad - self.grad, x - self.x) / res ** 2
            L = np.linalg.norm(grad - self.grad) / res

            rho = gamma / self.gamma
            alpha = np.sqrt(1 / self.params.pi + rho)
            delta = gamma ** 2 * L ** 2 - (2 - self.params.pi) * gamma * ell + 1 - self.params.pi

            if delta <= 0.:
                beta = np.Inf
            else:
                beta = 1 / np.sqrt(2 * delta)

            self.gamma = gamma

            gamma = gamma * np.minimum(alpha, beta)

            self.x[:] = x[:]
            self.grad[:] = grad[:]

            x = self.problem.proxable.eval_prox(self.x - gamma * self.grad, gamma)

