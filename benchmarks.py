from abc import ABC, abstractmethod

import numpy as np
import optim.function as fun

import matplotlib.pyplot as plt

from pathlib import Path

from math import floor, log10

def fexp10(f):
    return int(floor(log10(abs(f)))) if f != 0 else 0

def fman10(f):
    return f/10**fexp10(f)

class Benchmark(ABC):
    def __init__(self, name, config, optimizer_configs, num_runs):
        self.name = name
        self.config = config
        self.optimizer_configs = optimizer_configs
        self.num_runs = num_runs

        np.random.seed(self.config["seed"])

        self.setup()

    @abstractmethod
    def setup(self):
        pass

    @abstractmethod
    def setup_optimizer(self, optimizer_config, x_init, callback):
        pass

    @abstractmethod
    def get_fmin(self, x_init):
        pass

    @abstractmethod
    def get_filename(self):
        pass

    @abstractmethod
    def get_linear_transforms(self):
        pass

    @abstractmethod
    def get_objective(self):
        pass

    def run(self, overwrite_file=False):
        filename = self.get_filename()
        suffix = ".npz"

        if overwrite_file or not Path(filename + suffix).is_file():
            (self.fmin, sol) = self.get_fmin(np.zeros(self.get_linear_transforms()[0]._A.shape[1]))

            self.objective_values = dict()
            self.calls_linesearch = dict()
            self.calls_operator = dict()
        else:
            cache = np.load(filename + '.npz', allow_pickle=True)
            self.fmin = cache["fmin"]
            sol = cache["sol"]

            self.calls_operator = cache.get("calls_operator")
            self.calls_linesearch = cache["calls_linesearch"].item()
            self.objective_values = cache["objective_values"].item()


        fun.counting_enabled = True

        update_file = False

        for optimizer_config in self.optimizer_configs:
            if optimizer_config["name"] in self.objective_values:
                continue

            update_file = True
            np.random.seed(self.config["seed"])

            self.objective_values[optimizer_config["name"]] = [[] for _ in range(self.num_runs)]
            self.calls_linesearch[optimizer_config["name"]] = [[] for _ in range(self.num_runs)]
            self.calls_operator[optimizer_config["name"]] = [[] for _ in range(self.num_runs)]

            for run in range(self.num_runs):
                for transform in self.get_linear_transforms():
                    transform.reset_num_calls()

                x_init = eval(self.config["init_proc"] + "(" + str(self.get_linear_transforms()[0]._A.shape[1]) + ")")

                def callback(k, cum_num_backtracks, gamma, x, res):
                    fun.counting_enabled = False
                    objective_value = self.get_objective().eval(x)
                    fun.counting_enabled = True

                    self.objective_values[optimizer_config["name"]][run].append(objective_value)
                    self.calls_linesearch[optimizer_config["name"]][run].append(cum_num_backtracks)
                    self.calls_operator[optimizer_config["name"]][run].append(self.get_linear_transforms()[0].get_num_calls())

                    if k % self.config["verbose"] == 0:
                        print("    k", k, "i", cum_num_backtracks, "objective", objective_value, "gradnorm", res,
                              "gamma", gamma, "|x-x*|", np.linalg.norm(x - sol, 1) / self.get_linear_transforms()[0]._A.shape[1], np.min(x), np.max(x))

                    #calls = k * optimizer_config["class"].evals_per_iteration + i * optimizer_config["class"].evals_per_linesearch
                    if objective_value - self.fmin < self.config["ftol"]: #or calls > self.config["maxcalls"]:
                        return True

                    return False

                optimizer = self.setup_optimizer(optimizer_config, x_init, callback)
                print(optimizer_config["name"] + str(optimizer.params.__dict__))

                optimizer.run()

        if update_file:
            np.savez(filename,
                 fmin=self.fmin,
                 sol=sol,
                 objective_values=self.objective_values,
                 calls_linesearch=self.calls_linesearch,
                 calls_operator=self.calls_operator
                 )


    def linspace_values(self, x, y, interval):
        values = np.arange(0, x[-1], interval) * 0.

        j = 0
        for i in range(0, x[-1], interval):
            while True:
                if x[j] > i:
                    break
                j = j + 1

            # linearly interpolate the values at j-1 and j to obtain the value at i
            values[int(i / interval)] = (
                    y[j - 1]
                    + (i - x[j - 1])
                    * (y[j] - y[j - 1]) / (x[j] - x[j - 1])
            )
        return values

    def plot_mean_stdev(self, xvals, yvals, label, marker, color, refval=0., plotstdev=True, markevery=20,
                        plotevery=250):

        # compute new array with linspaced xvals with shortest length
        xvals_linspace = np.arange(0, xvals[0][-1], plotevery)
        for i in range(1, len(xvals)):
            arange = np.arange(0, xvals[i][-1], plotevery)
            if len(xvals_linspace) > len(arange):
                xvals_linspace = arange

        yvals_mean = np.zeros(len(xvals_linspace))

        for i in range(len(xvals)):
            y_values_interp = self.linspace_values(xvals[i],
                                                   yvals[i], plotevery)
            yvals_mean += y_values_interp[0:len(xvals_linspace)]

        yvals_mean = yvals_mean / len(xvals)

        plt.semilogy(xvals_linspace, yvals_mean - refval,
                     label=label,
                     marker=marker,
                     markevery=markevery,
                     color=color)

        if len(xvals) > 1 and plotstdev:
            yvals_stdev = np.zeros(len(xvals_linspace))

            for i in range(len(xvals)):
                y_values_interp = self.linspace_values(xvals[i],
                                                       yvals[i], plotevery)

                yvals_stdev += (yvals_mean - y_values_interp[0:len(xvals_linspace)]) ** 2

            yvals_stdev = np.sqrt(yvals_stdev / len(xvals))

            plt.fill_between(xvals_linspace,
                             yvals_mean - refval - yvals_stdev,
                             yvals_mean - refval + yvals_stdev,
                             alpha=0.5, facecolor=color,
                             edgecolor='white')

    def plot_suboptimality(self, markevery, plotevery, calls_to_lin_trans=True, exclude = None):
        self.plot(self.objective_values, self.fmin, "$\\varphi(x) - \\varphi(x^\\star)$", markevery, plotevery, calls_to_lin_trans, exclude)


    def plot(self, yvals, refval, ylabel, markevery, plotevery, calls_to_lin_trans=True, exclude = None):
        for optimizer_config in self.optimizer_configs:
            if not exclude is None:
                if optimizer_config["name"] in exclude:
                    continue

            if self.num_runs == 1:
                calls = self.calls_linesearch[optimizer_config["name"]][0]

                iters = np.arange(0, len(calls))

                if calls_to_lin_trans:
                    xvals = (np.array(iters) * optimizer_config["evals_per_iteration"]
                                + np.array(calls) * optimizer_config["evals_per_linesearch"])

                    plt.semilogy(xvals[xvals <= self.config["maxcalls"]], np.array(yvals[optimizer_config["name"]][0])[xvals <= self.config["maxcalls"]] - refval,
                                 label=optimizer_config["label"],
                                 marker=optimizer_config["marker"],
                                 markevery=markevery,
                                 color=optimizer_config["color"])
                    plt.xlabel("number of calls to $A,A^\\top$")
                else:
                    plt.semilogy(iters,
                                 np.array(yvals[optimizer_config["name"]][0]) - refval,
                                 label=optimizer_config["label"],
                                 marker=optimizer_config["marker"],
                                 markevery=markevery,
                                 color=optimizer_config["color"])
                    plt.xlabel("iteration $k$")
            else:
                xvals = []
                for calls in self.calls_linesearch[optimizer_config["name"]]:
                    iters = np.arange(0, len(calls))
                    xvals.append((np.array(iters) * optimizer_config["evals_per_iteration"]
                             + np.array(calls) * optimizer_config["evals_per_linesearch"]))

                self.plot_mean_stdev(xvals, yvals[optimizer_config["name"]],
                                     label = optimizer_config["label"],
                                     marker = optimizer_config["marker"],
                                     color = optimizer_config["color"],
                                     refval = refval,
                                     plotstdev= True,
                                     markevery = markevery,
                                     plotevery = plotevery)

                plt.xlabel("number of calls to $A,A^\\top$")

            #plt.ylabel(ylabel)


        plt.tight_layout()
        plt.legend()
