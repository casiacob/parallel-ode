import jax.numpy as jnp
from jax import random
from parode.sequential import seq_integrate
from parode.parallel import par_integrate
from parode.parareal import parareal_integrate
from parode.integrators import euler, rk4
import matplotlib.pyplot as plt

from jax import config

config.update("jax_enable_x64", True)
config.update("jax_platform_name", "cuda")

n_oscillators = 100
K = 2 * jnp.sqrt(2 * jnp.pi)/jnp.pi
t0 = 0.0
tf = 10.0
dt = 1e-2
t_eval = jnp.arange(t0, tf + dt, dt)
nb_steps = len(t_eval) - 1

key = random.PRNGKey(42)
omega = random.normal(key, (n_oscillators,))
theta0 = random.uniform(key, (n_oscillators,), minval=0, maxval=2 * jnp.pi)

def kuramoto(theta):
    theta_diff = theta[:, None] - theta[None, :]
    coupling = jnp.sum(jnp.sin(theta_diff), axis=1)
    return omega + (K / n_oscillators) * coupling


theta_rk4_seq = seq_integrate(kuramoto, theta0, dt, nb_steps, rk4)
theta_rk4_par = par_integrate(kuramoto, theta0, dt, nb_steps, rk4)

theta_G, t_G = parareal_integrate(kuramoto, theta0, dt, nb_steps, rk4, rk4)

plt.plot(t_eval, theta_rk4_seq)
plt.plot(t_eval, theta_rk4_par)
plt.plot(t_G, theta_G, marker="o")
plt.show()