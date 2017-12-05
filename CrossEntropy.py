import matplotlib.pyplot as plt
import numpy as np

from Animate import animate
from Benchmarks import objective_functions

class CrossEntropy:
    """
    Optimize a function using the Cross Entropy method. Adapted from:
    http://www.cleveralgorithms.com/nature-inspired/probabilistic/cross_entropy.html
    """

    CURVE_DENSITY = 40


    def __init__(self, dimension: int, bounds: np.ndarray, max_iterations: int, num_samples: int, num_update: int, learning_rate: float, **kwargs):
        """
        Initialize parameters for Cross Entropy method of optimization.

        :param dimension: number of dimensions to optimize.
        :param bounds: list of tuples of upper/lower bounds. List is the same size as dimension.
        :param max_iterations: maximum number of iterations in performing optimization.
        :param num_samples: number of samples in Monte Carlo simulation.
        :param num_update: number of samples to consider when updating distribution.
        :param learning_rate: speed of convergence.
        """

        self.dimension = dimension
        self.bounds = bounds.astype(np.float64)
        self.max_iterations = max_iterations
        self.num_samples = num_samples
        self.num_update = num_update
        self.learning_rate = learning_rate

        # initialize results to empty, will be set after optimization
        self.bests = []
        self.averages = []

        # kwargs for compatibility of **params in main function
        self.kwargs = kwargs


    def optimize(self, objective_function: callable, do_print: bool = False) -> (np.ndarray, list):
        """
        Optimizes given objective function.

        :param objective_function: function to optimize
        :param do_print: toggle output on and off
        :return: best solution, list of individuals by iteration
        """

        # Randomly initialize means within given bounds
        means = self.bounds[:, 0] + np.multiply(
            np.random.rand(self.dimension),
            self.bounds[:, 1] - self.bounds[:, 0]
        )

        # Initialize standard deviations as range of bounds
        standard_deviations = self.bounds[:, 1] - self.bounds[:, 0]
        standard_deviations = standard_deviations.astype(np.float64)

        # Initialize current best to None
        best_individual = None
        best_cost = None

        # Intialize stuff for plotting
        best_by_iteration = []
        average_by_iteration = []

        samples_by_iteration = []

        # Iterate until convergence or until max iterations is reached
        for iteration in range(self.max_iterations):

            # Generate samples from a Gaussian distribution between bounds
            samples = np.asarray(
                [[max(min(np.random.normal(means[i], standard_deviations[i]), self.bounds[i, 1]), self.bounds[i, 0]) for i in range(self.dimension)] for _ in range(self.num_samples)]
            )
            sample_costs = objective_function(samples)

            # Sort samples and sample costs by index
            sorted_indices = sample_costs.argsort()
            samples = samples[sorted_indices]
            sample_costs = sample_costs[sorted_indices]

            # Update best individual if we have discovered a new best
            if best_individual is None or best_cost > sample_costs[0]:
                best_individual, best_cost = samples[0], sample_costs[0]

            # Select individuals to influence distribution
            selected_individuals = samples[:self.num_update]

            # Update distribution parameters
            for i in range(self.dimension):
                means[i] = self.learning_rate * means[i] + ((1.0-self.learning_rate) * np.mean(selected_individuals[:, i]))
                standard_deviations[i] = self.learning_rate * standard_deviations[i] + ((1.0-self.learning_rate) * np.std(selected_individuals[:, i]))

            # Display status of algorithm
            if do_print:
                print("Iteration {iteration}\t\tBest cost: {fitness}\t\tBest individual: {individual}".format(
                    iteration=str.zfill(str(iteration), 3),
                    fitness=repr(best_cost),
                    individual=", ".join(list(map(lambda x : str(x), best_individual)))
                ))

            samples_by_iteration.append(samples)
            best_by_iteration.append(best_cost)
            average_by_iteration.append(np.mean(sample_costs))

        self.bests.append(best_by_iteration)
        self.averages.append(average_by_iteration)

        return best_individual, samples_by_iteration


    def plot_reps(self, title, random_search_avg: list = None, random_search_best: list = None) -> None:
        """
        Plot all reps with global best solutions.

        :param title: suptitle of plot
        :param random_search_avg: average fitness value for random search across generations
        :param random_search_best: best fitness value for random search by generation
        :return: void
        """

        is_random_search = not (random_search_avg is None or random_search_best is None)

        plt.subplots(1, 2, figsize=(16, 8))
        plt.suptitle(title)
        plt.subplot(1, 2, 1)
        plt.title("Average Cost by Iteration")
        plt.xlabel("Iteration")
        plt.ylabel("Average Cost")

        num_reps = len(self.averages)

        # plot each rep
        for rep in range(num_reps):
            plt.plot(range(len(self.averages[rep])), self.averages[rep], label="Rep %d Average" % (rep + 1))

        if is_random_search:
            plt.plot(range(len(random_search_avg)), random_search_avg, label="Random Search Average")
            # plt.plot(range(len(random_search_best)), random_search_best, label="Random Search Best")

        plt.legend(loc="upper left")

        plt.subplot(1, 2, 2)
        plt.title("Lowest Cost by Iteration")
        plt.xlabel("Iteration")
        plt.ylabel("Best Cost")

        # plot each rep
        for rep in range(num_reps):
            plt.plot(range(len(self.bests[rep])), self.bests[rep], label="Rep %d Best" % (rep + 1))

        if is_random_search:
            plt.plot(range(len(random_search_best)), random_search_best, label="Random Search Best")

        plt.legend(loc="upper left")
        plt.show()


if __name__ == "__main__":

    from RandomSearch import RandomSearch
    objective_function = "Easom"

    params = {
        "dimension": 2,
        #"bounds": np.ones((10000, 1))*np.asarray([[0., 100.]]),
        "bounds": objective_functions[objective_function]["bounds"],
        "max_iterations": 30,
        "num_samples": 500,
        "num_update": 10,
        "learning_rate": 0.5,
        "num_reps": 5
    }

    CE = CrossEntropy(**params)
    best_individual, _ = CE.optimize(objective_functions[objective_function]["f"], True)

    RS = RandomSearch(**params)

    RS_best, RS_samples_by_iteration, RS_best_by_iteration, RS_average_by_iteration = RS.optimize(objective_functions[objective_function]["f"])
    CE_bests, CE_averages = [], []

    CE_best, CE_samples_by_iteration = None, None
    for rep in range(params["num_reps"]):
        CE_best, CE_samples_by_iteration = CE.optimize(objective_functions[objective_function]["f"], do_print=False)

    animate(objective_functions[objective_function]["f"], CE_samples_by_iteration, params["bounds"], CrossEntropy.CURVE_DENSITY)
    CE.plot_reps("Cross Entropy on %s Function (α=%.2f)" % (objective_function, params["learning_rate"]), RS_average_by_iteration, RS_best_by_iteration)
