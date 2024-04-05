import numpy as np

import matplotlib.pyplot as plt
import matplotlib

from sklearn.datasets import load_svmlight_file

import optim.function as fun
import optim.optimizer as optim
import logistic_regression_optimizer as logistic_optim
import benchmarks

from libsvmdata import fetch_libsvm


class Ell1RegularizedLogisticRegression(benchmarks.Benchmark):
    def setup(self):
        if self.config["dataset"] == "mushrooms":
            data = load_svmlight_file("datasets/mushrooms")
            X, targets = data[0].toarray(), data[1]
        elif self.config["dataset"] == "leukemia":
            data = load_svmlight_file("datasets/leu.bz2")
            X, targets = data[0].toarray(), data[1]
        else:
            X, targets = fetch_libsvm(self.config["dataset"])
        labels = np.copy(targets)
        labels[targets == min(targets)] = -1
        labels[targets == max(targets)] = +1


        if hasattr(X, "nnz"):
            X = np.array(X.todense())
        X = np.column_stack((X, np.ones(X.shape[0])))
        A = -X * labels[:, None]
        self.linear_transform = fun.LinearTransform(A)

        m, self.n = A.shape

        self.fmin = np.Inf
        self.sol = np.zeros(self.n)

        self.loss = fun.LogisticLoss(weight = 1. / m)

        self.objective = fun.AdditiveComposite(
            (
                fun.AffineCompositeLoss(self.loss, self.linear_transform),
                fun.OneNorm(config["nu"])
            )
        )


    def get_fmin(self, x_init):
        params = logistic_optim.Parameters(
            method=optim.AdaptiveProximalGradientMethod,
            maxit=15000,
            tol=1e-15)


        self.fmin = np.Inf

        def callback(k, cum_num_backtracks, gamma, x, res):
            self.sol = x
            objective_value = self.objective.eval(x)
            if objective_value < self.fmin:
                self.fmin = objective_value
            if k % 100 == 0:
                print("k", k, "i", cum_num_backtracks, "objective", objective_value, "gradnorm", res, "gamma", gamma, np.min(x), np.max(x))


        algorithm = logistic_optim.OptimWrapper(self.loss, self.config["nu"], 1., self.linear_transform, x_init, callback, params)
        algorithm.run()

        return (self.fmin, self.sol)

    def get_objective(self):
        return self.objective

    def get_linear_transforms(self):
        return [self.linear_transform]

    def get_filename(self):
        return ("results/" + self.name
                + "_dataset_" + self.config["dataset"]
                + "_nu_" + str(self.config["nu"]))


    def setup_optimizer(self, optimizer_config, x_init, callback):
        return optimizer_config["class"](self.loss, self.config["nu"], 1.,
                                         self.linear_transform, x_init, callback, optimizer_config["params"])


np.random.seed(1)

#dataset = "covtype.binary"
#dataset = "rcv1.binary"
#dataset = "mushrooms"
name = "ell1_regularized_logistic_regression_simple"
configs = [
    {
        "dataset": "w8a",
        "nu": 1e-4,
        "verbose": 100,
        "seed": 100,
        "ftol": 1e-14,
        "init_proc": "np.zeros",
        "markevery": 100,
        "plotevery": 1,
        "maxcalls": 2000
    }
]
maxit = 2000
optimizer_configs = [
    {
        "name": "LS-PGD",
        "label": "LS-PGD$^{0.5}$",
        "marker": "X",
        "color": "red",
        "class": logistic_optim.OptimWrapper,
        "evals_per_linesearch": optim.LineSearchProximalGradientDescent.evals_per_linesearch,
        "evals_per_iteration": optim.LineSearchProximalGradientDescent.evals_per_iteration,
        "params": logistic_optim.Parameters(maxit=maxit,
                                            method=optim.LineSearchProximalGradientDescent,
                                            tol=1e-15,
                                            eps=1e-13
                                            )
    },
    {
        "name": "PGD",
        "label": "PGD",
        "marker": "P",
        "color": "darkgreen",
        "class": logistic_optim.OptimWrapper,
        "evals_per_linesearch": optim.ProximalGradientDescent.evals_per_linesearch,
        "evals_per_iteration": optim.ProximalGradientDescent.evals_per_iteration,
        "params": logistic_optim.Parameters(maxit=maxit,
                                            method=optim.ProximalGradientDescent,
                                            tol=1e-15,
                                            eps=1e-13)
    },
    {
        "name": "LS-APG0.5",
        "label": "LS-APG$^{0.5}_\pm$",
        "class": logistic_optim.AnisotropicProximalGradientMethod,
        "evals_per_linesearch": logistic_optim.AnisotropicProximalGradientMethod.evals_per_linesearch,
        "evals_per_iteration": logistic_optim.AnisotropicProximalGradientMethod.evals_per_iteration,
        "marker": "o",
        "color": "purple",
        "params": logistic_optim.Parameters(maxit=maxit,
                                            tol=1e-15,
                                            alpha=0.5,
                                            eps=0.0)
    },
    {
        "name": "APG0.0",
        "label": "APG$_\pm$",
        "marker": "^",
        "color": "orange",
        "class": logistic_optim.AnisotropicProximalGradientMethod,
        "evals_per_linesearch": logistic_optim.AnisotropicProximalGradientMethod.evals_per_linesearch,
        "evals_per_iteration": logistic_optim.AnisotropicProximalGradientMethod.evals_per_iteration,
        "params": logistic_optim.Parameters(maxit=maxit,
                                            tol=1e-15,
                                            alpha=0.0)
    }
]
exclude = {}
for config in configs:
    print("dataset", config["dataset"], "nu", config["nu"])
    benchmark = Ell1RegularizedLogisticRegression(name, config, optimizer_configs, 1)
    benchmark.run()

    matplotlib.rcParams['mathtext.fontset'] = 'cm'
    fig, ax = plt.subplots(figsize=(5, 4))

    fig.suptitle(config["dataset"] + ", $\\nu=" + ("" if benchmarks.fman10(config["nu"]) == 1 else str(benchmarks.fman10(config["nu"])) + " \\times") + "10^{" + str(benchmarks.fexp10(config["nu"])) + "}$, $(m,n)="
                 + str(benchmark.linear_transform._A.shape) + "$", fontsize=12)

    ax.grid(True)
    benchmark.plot_suboptimality(markevery=config["markevery"], plotevery=config["plotevery"], calls_to_lin_trans=True, exclude= exclude)

    filename = benchmark.get_filename()
    suffix = "_objective.pdf"
    plt.savefig(filename + suffix, bbox_inches='tight')
    plt.close()
    #plt.show()

