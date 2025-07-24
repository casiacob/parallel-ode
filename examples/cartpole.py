import jax.numpy as jnp
from jax import random, jit
from parode.sequential import seq_integrate
from parode.parallel import par_integrate_rk4
from parode.parareal import parareal_integrate
from parode.integrators import rk4
import matplotlib.pyplot as plt

from jax import config

config.update("jax_enable_x64", True)
config.update("jax_platform_name", "cuda")

t0 = 0.0
tf = 5.0
dt = 1e-5
t_eval = jnp.arange(t0, tf + dt, dt)
nb_steps = len(t_eval) - 1
x0 = jnp.array([0.0, jnp.pi/3, 0.0, 0.0])


def cartpole(x):
    # https://underactuated.mit.edu/acrobot.html#cart_pole

    gravity = 9.81
    pole_length = 0.5
    cart_mass = 10.0
    pole_mass = 1.0
    total_mass = cart_mass + pole_mass

    cart_position, pole_position, cart_velocity, pole_velocity = x

    sth = jnp.sin(pole_position)
    cth = jnp.cos(pole_position)

    cart_acceleration = (
                                pole_mass * sth * (pole_length * pole_velocity ** 2 + gravity * cth)
                        ) / (cart_mass + pole_mass * sth ** 2)

    pole_acceleration = (
                                - pole_mass * pole_length * pole_velocity ** 2 * cth * sth
                                - total_mass * gravity * sth
                        ) / (pole_length * cart_mass + pole_length * pole_mass * sth ** 2)

    return jnp.hstack(
        (cart_velocity, pole_velocity, cart_acceleration, pole_acceleration)
    )


_seq = lambda _x0: seq_integrate(cartpole, _x0, dt, nb_steps, rk4)
_par = lambda _x0: par_integrate_rk4(cartpole, _x0, nb_steps, dt)
_parareal = lambda _x0: parareal_integrate(cartpole, _x0, 60, dt, nb_steps, rk4, rk4)

_jitted_seq = jit(_seq)
_jitted_par = jit(_par)
_jitted_parareal = jit(_parareal)

X_rk4_seq = _jitted_seq(x0)
X_rk4_par = _jitted_par(x0)
X_G, t_G = _jitted_parareal(x0)


plt.plot(t_eval, X_rk4_seq)
plt.plot(t_eval, X_rk4_par)
plt.plot(t_G, X_G, marker='o')
plt.show()
