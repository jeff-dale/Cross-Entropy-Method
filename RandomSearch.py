import numpy as np


class RandomSearch:


    def __init__(self, dimension: int, bounds: np.ndarray, max_iterations: int, num_samples: int, **kwargs):
        """
        Initialize parameters for Random Search benchmark.

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

        # unused, just for compatibility with initializing CE and RS with same **params
        self.kwargs = kwargs


    def optimize(self, objective_function: callable, do_print: bool = False) -> (np.ndarray, list):
        """
        Optimizes given objective function.

        :param objective_function: function to optimize
        :return: best solution, list of individuals by iteration
        """

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
                [[max(min(self.bounds[i, 0] + np.random.rand()*(self.bounds[i, 1] - self.bounds[i, 0]), self.bounds[i, 1]), self.bounds[i, 0]) for
                  i in range(self.dimension)] for _ in range(self.num_samples)]
            )
            sample_costs = objective_function(samples)

            # Sort samples and sample costs by index
            sorted_indices = sample_costs.argsort()
            samples = samples[sorted_indices]
            sample_costs = sample_costs[sorted_indices]

            # Update best individual if we have discovered a new best
            if best_individual is None or best_cost > sample_costs[0]:
                best_individual, best_cost = samples[0], sample_costs[0]

            # Display status of algorithm
            if do_print:
                print("Iteration {iteration}\t\tBest cost: {fitness}\t\tBest individual: {individual}".format(
                    iteration=str.zfill(str(iteration), 3),
                    fitness=repr(best_cost),
                    individual=repr(best_individual)
                ))

            samples_by_iteration.append(samples)
            best_by_iteration.append(best_cost)
            average_by_iteration.append(np.mean(sample_costs))


        return best_individual, samples_by_iteration, best_by_iteration, average_by_iteration
