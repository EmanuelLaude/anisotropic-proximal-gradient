import numpy as np

import matplotlib.pyplot as plt
import matplotlib
import scipy.sparse as sp

import optim.function as fun
import optim.optimizer as optim
import exponential_regression_optimizer as exp_optim
import benchmarks
import problem.problem_generator as gen


class RegularizedLinearPrograms(benchmarks.Benchmark):
    def setup(self):
        m, n = self.config["dims"]
        self.n = n

        params = gen.Parameters()
        params.seed = self.config["seed"]
        params.condition_number = self.config["cond"]
        params.norm_A = self.config["norm"]


        (E, d, e, opt_x, opt_y, opt_s) = gen.generate_lo_problem_with_opt(n, m, params)



        self.c, A, self.b = d, -E.T / params.norm_A, e / params.norm_A

        self.linear_transform = fun.LinearTransform(A)

        self.fmin = np.Inf
        self.sol = np.zeros(self.n)

        self.loss = fun.SumExp(weight=1, sigma=self.config["sigma"])
        self.objective = fun.AdditiveComposite(
            (
                fun.AffineCompositeLoss(self.loss, self.linear_transform, self.b),
                fun.Linear(self.c)
            )
        )




    def get_fmin(self, x_init):
        params = exp_optim.Parameters(
            method=optim.LBFGS,
            maxit=5000,
            tol=1e-12,
            mem=200,
            eps=1e-10)


        self.fmin = np.Inf

        def callback(k, cum_num_backtracks, gamma, x, res):
            self.sol = x
            objective_value = self.objective.eval(x)
            if objective_value < self.fmin:
                self.fmin = objective_value
            if k % 100 == 0:
                print("k", k, "i", cum_num_backtracks, "objective", objective_value, "gradnorm", res, "gamma", gamma, np.min(x), np.max(x))


        algorithm = exp_optim.OptimWrapper(self.loss, self.linear_transform, self.b, self.c, 0.0, x_init, callback, params)
        algorithm.run()

        return (self.fmin, self.sol)

    def get_objective(self):
        return self.objective

    def get_linear_transforms(self):
        return [self.linear_transform]

    def get_filename(self):
        return ("results/" + self.name + "_dims_" + str(self.config["dims"])
                + "_sigma_" + str(config["sigma"])
                + "_cond_" + str(config["cond"])
                + "_norm_" + str(config["sigma"]))


    def setup_optimizer(self, optimizer_config, x_init, callback):
        return optimizer_config["class"](self.loss, self.linear_transform, self.b, self.c,
                                         0.0, x_init, callback, optimizer_config["params"])


np.random.seed(1)

name = "regularized_lps"
configs = [
    {
        "dims": (6000, 1000),
        "cond": 10,
        "norm": 1.,
        "sigma": 1.,
        "verbose": 100,
        "seed": 100,
        "ftol": 1e-14,
        "init_proc": "np.zeros",
        "markevery": 100,
        "plotevery": 1,
        "maxcalls": 1400
    }
]
maxit = 2000
optimizer_configs = [
    {
        "name": "LS-GD",
        "label": "LS-GD$^{0.5}$",
        "marker": "X",
        "color": "red",
        "class": exp_optim.OptimWrapper,
        "evals_per_linesearch": optim.LineSearchProximalGradientDescent.evals_per_linesearch,
        "evals_per_iteration": optim.LineSearchProximalGradientDescent.evals_per_iteration,
        "params": exp_optim.Parameters(maxit=maxit,
                                       method=optim.LineSearchProximalGradientDescent,
                                       tol=1e-15,
                                       eps=1e-13,
                                       Wolfe=False,
                                       gamma_init=100
                                       )
    },
    {
        "name": "LS-APG0.5",
        "label": "LS-APG$^{0.5}_\pm$",
        "class": exp_optim.ExponentialProximalGradientMethod,
        "evals_per_linesearch": exp_optim.ExponentialProximalGradientMethod.evals_per_linesearch,
        "evals_per_iteration": exp_optim.ExponentialProximalGradientMethod.evals_per_iteration,
        "marker": "o",
        "color": "purple",
        "params": exp_optim.Parameters(maxit=maxit,
                                       tol=1e-15,
                                       alpha=0.5,
                                       eps=0.0)
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

exclude = { }

for config in configs:
    print(name + "_dims_" + str(config["dims"]) + "_sigma_" + str(config["sigma"]))
    benchmark = RegularizedLinearPrograms(name, config, optimizer_configs, 1)
    benchmark.run()

    matplotlib.rcParams['mathtext.fontset'] = 'cm'
    fig, ax = plt.subplots(figsize=(5, 4))

    fig.suptitle("Regularized LP, $(m,n)="
                 + str(benchmark.linear_transform._A.shape) + "$", fontsize=12)
    ax.grid(True)
    benchmark.plot_suboptimality(markevery=config["markevery"], plotevery=config["plotevery"], calls_to_lin_trans=True, exclude=exclude)

    filename = benchmark.get_filename()
    suffix = "_objective.pdf"
    plt.savefig(filename + suffix, bbox_inches='tight')
    plt.close()
    #plt.show()

