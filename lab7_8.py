__author__ = 'fiodar'

from wave_equation import *
from numpy import exp, sin, cos, pi
import matplotlib.pyplot as plt
import matplotlib.animation as animation

a = 1.
f = lambda x, t: exp(-x**2*t) * (2*a*t - 4 * a * t**2 * x**2 - x**2)
f_simple = lambda x, t: 0*x+0*t
x_conditions = [lambda t: 0 + 0*t, lambda t: 0 + 0*exp(-t)]
x_simple_conditions = [lambda t: 0*sin(t), lambda t: 0*t]
t_conditions = [lambda x: 1 + 0*x, lambda x: -x**2]
t_simple_conditions = [lambda x: 1 + 0*x, lambda x: 0*x]

x_bounds = [0, 1]
t_bounds = [0, 10]
fps = 30
x = np.linspace(x_bounds[0], x_bounds[1], 500)
t = np.linspace(t_bounds[0], t_bounds[1], fps*t_bounds[1])

u = wave_equation_solve_1d(a, f_simple, x, t, x_conditions, t_conditions)

# plotting
fig = plt.figure()
ax = fig.gca(xlim=(0, 1), ylim=(-2, 4))
ax.set_xlabel("x")
ax.set_ylabel("u")
l, = ax.plot([], [], 'b-', lw=2)


def animate_wave(time, data, line):
    x, u = data
    line.set_data(x, u[time])
    text = ax.text(0.01, 3, "$t=%.2f$" % (float(time)/fps), fontsize=16)
    return line, text

def init():
    l.set_data([], [])
    return l,

anim = animation.FuncAnimation(fig, animate_wave, frames=len(t),
                               init_func=init, fargs=([x, u], l), blit=True, interval=t_bounds[1]*100/fps)
# anim.save('solution.mp4', writer='ffmpeg', fps=fps)

plt.show()