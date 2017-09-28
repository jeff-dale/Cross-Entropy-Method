from matplotlib import animation
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np


def animate(objective_function: callable, samples_by_iteration, bounds: np.ndarray, curve_density: int):

    # Precalculate surface points for plotting
    xs = np.linspace(bounds[0][0], bounds[0][1], curve_density).reshape(
        (1, curve_density))
    ys = np.linspace(bounds[1][0], bounds[1][1], curve_density).reshape(
        (1, curve_density))
    X, Y = np.meshgrid(xs, ys)
    Z = objective_function(np.vstack((np.ravel(X), np.ravel(Y))).T).reshape(X.shape)

    fig = plt.figure()
    samples = fig.add_subplot(111, projection="3d")
    samples.plot_surface(X, Y, Z, color=(1, 1, 0, 0.5))
    individuals = samples.scatter(samples_by_iteration[0][:, 0], samples_by_iteration[0][:, 1], objective_function(samples_by_iteration[0]), c=(0, 0, 0), s=100)


    def init():
        plt.suptitle("Iteration 0")
        return samples,

    def iterate(i):
        nonlocal individuals
        plt.suptitle("Iteration " + str(i))
        individuals.remove()
        individuals = samples.scatter(samples_by_iteration[i][:, 0], samples_by_iteration[i][:, 1], objective_function(samples_by_iteration[i]), c=(0, 0, 0), s=100)
        return samples,

    anim = animation.FuncAnimation(fig, iterate, init_func=init, frames=len(samples_by_iteration), interval=60, blit=False)
    plt.show()
