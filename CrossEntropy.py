import numpy as np

from Animate import animate
from Benchmarks import Ackley, Easom, Rastrigin, Cross_in_Tray

class CrossEntropy:
    """
    Optimize a function using the Cross Entropy method. Adapted from:
    http://www.cleveralgorithms.com/nature-inspired/probabilistic/cross_entropy.html
    """

    CURVE_DENSITY = 40


    def __init__(self, dimension: int, bounds: np.ndarray, max_iterations: int, num_samples: int, num_update: int, learning_rate: float):
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


    def optimize(self, objective_function: callable) -> (np.ndarray, list):
        """
        Optimizes given objective function.

        :param objective_function: function to optimize
        :param plot: whether to plot
        :return: best solution, list of individuals by iteration
        """

        # Randomly initialize means within given bounds
        means = self.bounds[:, 0] + np.multiply(
            np.random.rand(self.dimension),
            self.bounds[:, 1] - self.bounds[:, 0]
        )

        # Initialize standard deviations as range of bounds
        standard_deviations = self.bounds[:, 1] - self.bounds[:, 0]

        # Initialize current best to None
        best_individual = None
        best_cost = None

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

            selected_individuals = samples[:self.num_update]

            for i in range(self.dimension):
                means[i] = self.learning_rate * means[i] + ((1.0-self.learning_rate) * np.mean(selected_individuals[:, i]))
                standard_deviations[i] = self.learning_rate * standard_deviations[i] + ((1.0-self.learning_rate) * np.std(selected_individuals[:, i]))

            print("Iteration {iteration}\t\tBest cost: {fitness}\t\tBest individual: {individual}".format(
                iteration=str.zfill(str(iteration), 3),
                fitness=repr(best_cost),
                individual=repr(best_individual)
            ))

            samples_by_iteration.append(samples)

        return best_individual, samples_by_iteration


if __name__ == "__main__":

    CE = CrossEntropy(
        dimension = 2,
        bounds = np.asarray([[-5, 5], [-5, 5]]),
        max_iterations = 100,
        num_samples = 50,
        num_update = 5,
        learning_rate = 0.7
    )

    #objective_function = lambda x : np.sum((np.power(x, 2)), axis=1); CE.bounds = np.asarray([[-5, 5], [-5, 5]])
    #objective_function = Rastrigin; CE.bounds = np.asarray([[-5, 5], [-5, 5]])
    #objective_function = Ackley; CE.bounds = np.asarray([[-5.12, 5.12], [-5.12, 5.12]])
    #objective_function = Easom; CE.bounds = np.asarray([[-100, 100], [-100, 100]])
    objective_function = Cross_in_Tray; CE.bounds = np.asarray([[-10, 10], [-10, 10]])
    best, samples_by_iteration = CE.optimize(objective_function)
    animate(objective_function, samples_by_iteration, CE.bounds, CrossEntropy.CURVE_DENSITY)
