from abc import ABC, abstractmethod
import numpy as np

from collections.abc import Iterable

##
# Global variables
##
counting_enabled = True
thres = 15.

##
# Interfaces
##
class Function(ABC):
    def __init__(self, weight = 1.):
        assert weight >= 0

        self._weight = weight

    def eval(self, x):
        return self._weight * self.eval_unweighted(x)

    @abstractmethod
    def eval_unweighted(self, x):
        pass

class Proxable(Function):
    def eval_prox(self, x, step_size, power = 2, norm = 2):
        return self.eval_prox_unweighted(x, self._weight * step_size, power, norm)

    @abstractmethod
    def eval_prox_unweighted(self, x, step_size, power = 2, norm = 2):
        pass

class Diffable(Function):
    def eval_gradient(self, x):
        return self._weight * self.eval_gradient_unweighted(x)

    def get_Lip_gradient(self):
        return self._weight * self.get_Lip_gradient_unweighted()

    @abstractmethod
    def eval_gradient_unweighted(self, x):
        pass

    def get_Lip_gradient_unweighted(self):
        return np.Inf



##
# Implementations
##
class NormPower(Diffable, Proxable):
    def __init__(self, power, norm = 2, weight = 1):
        super().__init__(weight)
        assert norm > 1. and power > 1.
        assert norm == power or norm == 2

        self._norm = norm
        self._power = power

    def eval_unweighted(self, x):
        if self._norm == self._power:
            return np.sum(np.power(np.abs(x), self._power)) / self._power
        elif self._norm == 2:
            return np.power(np.linalg.norm(x, 2), self._power) / self._power

    def eval_gradient_unweighted(self, x):
        if self._norm == self._power:
            return np.sign(x) * np.power(np.abs(x), self._power - 1)
        elif self._norm == 2:
            return np.power(np.linalg.norm(x, 2), self._power - 2) * x

    def eval_prox_unweighted(self, x, step_size, power = 2, norm = 2):
        assert power == 2 and norm == 2

        return x / (1. + step_size)

    def get_Lip_gradient_unweighted(self):
        if self._norm == 2. and self._power == 2.:
            return 1.

        return np.Inf


class PowerHingeLoss(Diffable):
    def __init__(self, power, weight = 1):
        super().__init__(weight)

        assert power > 1

        self._power = power

    def eval_unweighted(self, x):
        return np.sum(np.power(np.maximum(0., 1 + x), self._power)) / self._power

    def eval_gradient_unweighted(self, x):
        return np.power(np.maximum(0., 1 + x), self._power - 1)

class LogisticLoss(Diffable):
    def eval_unweighted(self, x):
        v = x.copy()
        v[x <= thres] = np.log(1 + np.exp(x[x <= thres]))
        return np.sum(v)

    def eval_gradient_unweighted(self, x):
        v = np.ones(x.shape)
        v[x <= thres] = np.exp(x[x <= thres]) / (1 + np.exp(x[x <= thres]))
        return v

    def get_Lip_gradient_unweighted(self):
        return 0.25

class SumExp(Diffable):
    def __init__(self, sigma, weight = 1):
        super().__init__(weight)
        self._sigma = sigma

    def eval_unweighted(self, x):
        return self._sigma * np.sum(np.exp(x / self._sigma))

    def eval_gradient_unweighted(self, x):
        return np.exp(x / self._sigma)


class LogSumExp(Diffable):
    def __init__(self, sigma, weight = 1):
        super().__init__(weight)
        self._sigma = sigma

    def eval_unweighted(self, x):
        rho = np.max(x / self._sigma)
        return self._sigma * (np.log(np.sum(np.exp(x / self._sigma - rho))) + rho)

    def eval_gradient_unweighted(self, x):
        rho = np.max(x / self._sigma)
        return np.exp(x / self._sigma - rho) / np.sum(np.exp(x / self._sigma - rho))

    def get_Lip_gradient_unweighted(self):
        return 1. / self._sigma

class Linear(Diffable):
    def __init__(self, c, weight = 1):
        super().__init__(weight)
        self.c = c

    def eval_unweighted(self, x):
        return np.dot(self.c, x)

    def eval_gradient_unweighted(self, x):
        return self.c

    def get_Lip_gradient_unweighted(self):
        return 0


def shrinkage(x, threshold):
    return np.maximum(0., np.abs(x) - threshold) * np.sign(x)


class Zero(Proxable):
    def eval_unweighted(self, x):
        return 0.

    def eval_prox_unweighted(self, x, step_size, power = 2, norm = 2):
        return x

def projection_simplex_sort(v, z=1):
    n_features = v.shape[0]
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u) - z
    ind = np.arange(n_features) + 1
    cond = u - cssv / ind > 0
    rho = ind[cond][-1]
    theta = cssv[cond][-1] / float(rho)
    w = np.maximum(v - theta, 0)
    return w

class IndicatorSimplex(Proxable):
    def eval_unweighted(self, x):
        if np.sum(x >= -1e10) == x.shape[0] and np.abs(np.sum(x) - 1) <= 1e-10:
            return 0
        return np.Inf

    def eval_prox_unweighted(self, x, step_size, power = 2, norm = 2):
        assert power == 2 and norm == 2

        return projection_simplex_sort(x)



class OneNorm(Proxable):
    def eval_unweighted(self, x):
        return np.sum(np.abs(x))

    def eval_prox_unweighted(self, x, step_size, power = 2, norm = 2):
        assert power == norm

        conj_power = power / (power - 1)

        threshold = np.power(step_size, conj_power - 1)

        return shrinkage(x, threshold)

class Indicator2NormBall(Proxable):
    def __init__(self, radius):
        super().__init__(1.)
        self._radius = radius


    def eval_unweighted(self, x):
        if np.linalg.norm(x, 2) <= self._radius + 1e-12:
            return 0
        return np.Inf

    def eval_prox_unweighted(self, x, step_size, power = 2, norm = 2):
        assert norm == 2


        if np.linalg.norm(x, 2) <= self._radius:
            return x
        else:
            return self._radius * (x / np.linalg.norm(x, 2))


class ElasticNet(Proxable):
    def __init__(self, alpha, weight = 1):
        super().__init__(weight)
        self._alpha = alpha

    def eval_unweighted(self, x):
        return self._alpha * 0.5 * np.dot(x, x) + np.sum(np.abs(x))

    def eval_prox_unweighted(self, x, step_size, power = 2, norm = 2):
        assert power == 2 and norm == 2

        threshold = step_size / (1 + step_size * self._alpha)

        z = x / (1 + step_size * self._alpha)

        return shrinkage(z, threshold)


class LinearTransform:
    def __init__(self, A):
        self._A = A
        self._num_calls = 0
        self._norm = np.linalg.norm(self._A, 2)

    def apply(self, x):
        if counting_enabled:
            self._num_calls += 1
        return np.dot(self. _A, x)

    def get_norm(self):
        return self._norm

    def apply_transpose(self, x):
        if counting_enabled:
            self._num_calls += 1
        return np.dot(self._A.T, x)

    def reset_num_calls(self):
        self._num_calls = 0

    def get_num_calls(self):
        return self._num_calls


class AffineCompositeLoss(Diffable):
    def __init__(self, loss, A = None, b = None, weight = 1):
        super().__init__(weight)

        self._A = A
        self._norm_A = A.get_norm()
        self._b = b
        self._loss = loss

    def eval_unweighted(self, x):
        p = x
        if not self._A is None:
            p = self._A.apply(x)

        if not self._b is None:
            p = p - self._b

        return self._loss.eval(p)

        return value

    def eval_gradient_unweighted(self, x):
        pred = x
        if not self._A is None:
            pred = self._A.apply(x)

        if not self._b is None:
            pred = pred - self._b

        grad = self._loss.eval_gradient(pred)

        if not self._A is None:
            grad = self._A.apply_transpose(grad)

        return grad

    def get_Lip_gradient_unweighted(self):
        return self._norm_A * self._norm_A * self._loss.get_Lip_gradient()

class AdditiveComposite(Diffable):
    def __init__(self, diffables, weight = 1):
        super().__init__(weight)
        self._diffables = diffables

    def eval_unweighted(self, x):
        value = 0.
        for diffable in self._diffables:
            value += diffable.eval(x)

        return value

    def eval_gradient_unweighted(self, x):
        grad = np.zeros(x.shape)
        for diffable in self._diffables:
            grad = grad + diffable.eval_gradient(x)

        return grad

    def get_Lip_gradient_unweighted(self):
        Lip = 0.0
        for diffable in self._diffables:
            Lip += diffable.get_Lip_gradient()

        return Lip


