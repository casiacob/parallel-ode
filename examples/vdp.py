import jax.numpy as jnp
from parode.sequential import seq_integrate
from parode.parallel import par_integrate
from parode.parareal import parareal_integrate
from parode.integrators import euler, rk4
import matplotlib.pyplot as plt

from jax import config

config.update("jax_enable_x64", True)
config.update("jax_platform_name", "cuda")


t0 = 0.0
tf = 10.0
dt = 1e-2
t_eval = jnp.arange(t0, tf + dt, dt)
nb_steps = len(t_eval) - 1


def vdp(z):
    x, y = z
    mu = 1.0
    return jnp.hstack((y, mu * (1 - x**2) * y - x))


z0 = jnp.array((0.0, 1.0))
Z_rk4_seq = seq_integrate(vdp, z0, dt, nb_steps, rk4)

Z_rk4_par = par_integrate(vdp, z0, dt, nb_steps, rk4)

X_G, t_G = parareal_integrate(vdp, z0, dt, nb_steps, rk4, rk4)

plt.plot(t_eval, Z_rk4_seq)
plt.plot(t_eval, Z_rk4_par)
plt.plot(t_G, X_G, marker="o")
plt.show()
