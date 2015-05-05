__author__ = 'fiodar'

from wave_equation import *
from numpy import exp
import matplotlib.pyplot as plt


a = 1.
f = lambda x, t: exp(-x**2*t) * (2*a*t - 4 * a * t**2 * x**2 - x**2)
x_bounds = [0, 1]
t_bounds = [0, 1]
x_conditions = [lambda t: 1 + 0*t, lambda t: exp(-t)]
t_conditions = [lambda x: 1 + 0*x, lambda x: -x**2]

x = np.linspace(x_bounds[0], x_bounds[1], 100)
t = np.linspace(t_bounds[0], t_bounds[1], 50)

u = wave_equation_solve_1d(a, f, x_bounds, t_bounds, x_conditions, t_conditions, 10, 10)

plt.plot(x, u[-1])
plt.show()