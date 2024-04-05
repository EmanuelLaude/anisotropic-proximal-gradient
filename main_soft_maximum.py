import numpy as np

import matplotlib.pyplot as plt
import matplotlib

import optim.function as fun
import optim.optimizer as optim
import exponential_regression_optimizer as exp_optim
import benchmarks

import scipy.sparse as sp

class SoftMaximum(benchmarks.Benchmark):
    def setup(self):
        m, n = self.config["dims"]
        self.n = n
        #A = 10. * np.random.rand(m, n) - 5.
        self.b = 2 * np.random.rand(m) - 1

        A = np.random.rand(m, n) - 0.5

        u, s, v = np.linalg.svd(A, full_matrices=False)
        s = np.linspace(self.config["norm"], self.config["norm"] / self.config["cond"], min(m, n))

        A = np.dot(u * s, v)
        #A = sp.rand(m, n, density=config["density"], format="csr", random_state=42)
        #A = np.array(A.todense())
        #B = 2 * np.random.rand(m, n) - 1
        #A[A > 0] = B[A > 0]

        L = 0
        m, _ = A.shape
        for i in range(m):
            L = np.maximum(L, np.sum(np.abs(A[i, :])))

        print("|A|_{infty,1}", L/config["sigma"], "|A|_2^2", np.power(np.linalg.norm(A, 2), 2)/config["sigma"])

        self.linear_transform = fun.LinearTransform(A)

        self.fmin = np.Inf
        self.sol = np.zeros(self.n)

        self.loss = fun.LogSumExp(self.config["sigma"])
        self.objective = fun.AdditiveComposite(
            (
                fun.AffineCompositeLoss(self.loss, self.linear_transform, self.b),
                fun.NormPower(2, 2, self.config["nu"])
            )
        )


    def get_fmin(self, x_init):
        params = exp_optim.Parameters(
            method=optim.LBFGS,
            maxit=1000,
            tol=1e-15,
            mem=200,
            eps=1e-8)


        self.fmin = np.Inf

        def callback(k, cum_num_backtracks, gamma, x, res):
            self.sol = x
            objective_value = self.objective.eval(x)
            if objective_value < self.fmin:
                self.fmin = objective_value
            if k % 100 == 0:
                print("k", k, "i", cum_num_backtracks, "objective", objective_value, "gradnorm", res, "gamma", gamma, np.min(x), np.max(x))

        m, n = self.linear_transform._A.shape
        c = np.zeros(n)
        algorithm = exp_optim.OptimWrapper(self.loss, self.linear_transform, self.b, c, self.config["nu"], x_init, callback, params)
        algorithm.run()

        return (self.fmin, self.sol)

    def get_objective(self):
        return self.objective

    def get_linear_transforms(self):
        return [self.linear_transform]

    def get_filename(self):
        return ("results/" + self.name + "_dims_" + str(self.config["dims"])
                + "_nu_" + str(config["nu"])
                + "_sigma_" + str(config["sigma"])
                + "_norm_" + str(config["norm"])
                + "_cond_" + str(config["cond"])
                )


    def setup_optimizer(self, optimizer_config, x_init, callback):
        m, n = self.linear_transform._A.shape
        c = np.zeros(n)
        return optimizer_config["class"](self.loss, self.linear_transform, self.b, c,
                                         self.config["nu"], x_init, callback, optimizer_config["params"])


np.random.seed(1)

name = "soft_maximum"
configs = [
    {
        "dims": (3000, 1000),
        "nu": 0.0,
        "norm": 100,
        "cond": 150,
        #"density": 0.3,
        "sigma": 0.05,
        "verbose": 100,
        "seed": 100,
        "ftol": 1e-14,
        "init_proc": "np.zeros",
        "markevery": 200,
        "plotevery": 1,
        "maxcalls": 2000
    },
    # {
    #     "dims": (3000, 1000),
    #     "nu": 0.0,
    #     "norm": 100,
    #     "cond": 150,
    #     #"density": 0.9,
    #     "sigma": 0.05,
    #     "verbose": 100,
    #     "seed": 100,
    #     "ftol": 1e-14,
    #     "init_proc": "np.zeros",
    #     "markevery": 200,
    #     "plotevery": 1,
    #     "maxcalls": 2000
    # }
]
maxit = 3000
optimizer_configs = [
    {
        "name": "GD",
        "label": "GD",
        "marker": "P",
        "color": "darkgreen",
        "class": exp_optim.OptimWrapper,
        "evals_per_linesearch": optim.ProximalGradientDescent.evals_per_linesearch,
        "evals_per_iteration": optim.ProximalGradientDescent.evals_per_iteration,
        "params": exp_optim.Parameters(maxit=maxit,
                                       method=optim.ProximalGradientDescent,
                                       tol=1e-15,
                                       eps=1e-13)
    },
    {
        "name": "APG0.0",
        "label": "APG$_\pm$",
        "marker": "^",
        "color": "orange",
        "class": exp_optim.ExponentialProximalGradientMethod,
        "evals_per_linesearch": exp_optim.ExponentialProximalGradientMethod.evals_per_linesearch,
        "evals_per_iteration": exp_optim.ExponentialProximalGradientMethod.evals_per_iteration,
        "params": exp_optim.Parameters(maxit=maxit,
                                       tol=1e-15,
                                       alpha=0.0)
    }
]

exclude = {}
for config in configs:
    print(name + "_dims_" + str(config["dims"]) + "_nu_" + str(config["nu"])
          + "_norm_" + str(config["norm"])
          + "_cond_" + str(config["cond"]))
    benchmark = SoftMaximum(name, config, optimizer_configs, 1)
    benchmark.run()

    matplotlib.rcParams['mathtext.fontset'] = 'cm'
    fig, ax = plt.subplots(figsize=(5, 4))

    fig.suptitle("Soft maximum, $(m,n)="
                 + str(benchmark.linear_transform._A.shape) + "$", fontsize=12)
    ax.grid(True)
    benchmark.plot_suboptimality(markevery=config["markevery"], plotevery=config["plotevery"], calls_to_lin_trans=False, exclude=exclude)

    filename = benchmark.get_filename()
    suffix = "_objective.pdf"
    plt.savefig(filename + suffix, bbox_inches='tight')
    plt.close()
    #plt.show()

