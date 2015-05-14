__author__ = 'fiodar'

from wave_equation import *
from numpy import exp, sin, cos, pi
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.widgets import Slider, Button

a = 1.
f = lambda x, t: exp(-x**2*t) * (2*a*t - 4 * a * t**2 * x**2 - x**2)
f_simple = lambda x, t: 0 * x + 0 * t
x_conditions = [lambda t: 1 + 0 * t, lambda t: 0 + exp(-t)]
x_simple_conditions = [lambda t: 0 * sin(t), lambda t: 0 * t]
t_conditions = [lambda x: 1 + 0 * x, lambda x: -x ** 2]
t_simple_conditions = [lambda x: sin(pi * x), lambda x: 0 * x]

x_bounds = [0, 1]
t_bounds = [0, 10]
x = np.linspace(x_bounds[0], x_bounds[1], 200)
t = np.linspace(t_bounds[0], t_bounds[1], 100)

u = wave_equation_solve_1d(a, f, x, t, x_conditions, t_conditions)


# plotting
fig = plt.figure()
plt.subplots_adjust(bottom=0.25)
ax = fig.gca(xlim=(0, 1), ylim=(-3, 4))
ax.set_xlabel("x")
ax.set_ylabel("u")
l, = ax.plot(x, u[0], 'b-', lw=2)


def animate_wave(time, data, line):
    x, u = data
    stime.set_val(t_bounds[1] * float(time) / len(t))
    line.set_data(x, u[time])
    return line,


def init():
    l.set_data([], [])
    return l,


axtime = plt.axes([0.2, 0.1, 0.65, 0.03])
stime = Slider(axtime, 'time', t_bounds[0], t_bounds[1] * (1 - 1. / len(t)), valinit=t_bounds[0])


def update(val):
    time = stime.val
    l.set_data(x, u[int(time / t_bounds[1] * len(t))])


stime.on_changed(update)



# playax = plt.axes([0.69, 0.025, 0.1, 0.04])
# button_play = Button(playax, 'Play', color='w', hovercolor='0.975')
#
# anim = animation.FuncAnimation(fig, animate_wave, frames=len(t),
# init_func=init, fargs=([x, u], l), blit=True, interval=t_bounds[1]*100/fps)
#
# def play(event):
# anim._start()
#
# def stop(event):
#     l.set_data(x, u[int(stime.val/t_bounds[1]*len(t))])
#
# button_play.on_clicked(play)

resetax = plt.axes([0.8, 0.025, 0.1, 0.04])
button_reset = Button(resetax, 'Reset', color='w', hovercolor='0.975')


def reset(event):
    stime.reset()


button_reset.on_clicked(reset)

# anim = animation.FuncAnimation(fig, animate_wave, frames=len(t),
#                                init_func=init, fargs=([x, u], l), blit=True, interval=t_bounds[1]*100/fps)
# anim.save('solution.mp4', writer='ffmpeg', fps=fps)

plt.show()