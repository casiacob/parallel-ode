import jax.numpy as jnp
from jax import random, jit
from parode.sequential import seq_integrate
from parode.parallel import par_integrate
from parode.parareal import parareal_integrate
from parode.integrators import rk4
import time
import pandas as pd

from jax import config

config.update("jax_enable_x64", True)
config.update("jax_platform_name", "cuda")

K = 2 * jnp.sqrt(2 * jnp.pi)/jnp.pi
t0 = 0.0
tf = 10.0
dt = 1e-2
t_eval = jnp.arange(t0, tf + dt, dt)
nb_steps = len(t_eval) - 1


n_oscillators = [10, 100, 1000, 10000]
mean_seq_elapsed = []
mean_par_elapsed = []
mean_parareal_elapsed = []

key = random.PRNGKey(465)


for N in n_oscillators:
    omega = random.normal(key, (N,))
    theta0 = random.uniform(key, (N,), minval=0, maxval=2 * jnp.pi)


    def kuramoto(theta):
        theta_diff = theta[:, None] - theta[None, :]
        coupling = jnp.sum(jnp.sin(theta_diff), axis=1)
        return omega + (K / N) * coupling


    _seq = lambda _theta0: seq_integrate(kuramoto, _theta0, dt, nb_steps, rk4)
    _par = lambda _theta0: par_integrate(kuramoto, _theta0, dt, nb_steps, rk4)
    _parareal = lambda _theta0: parareal_integrate(kuramoto, _theta0, dt, nb_steps, rk4, rk4)

    _jitted_seq = jit(_seq)
    _jitted_par = jit(_par)
    _jitted_parareal = jit(_parareal)

    _jitted_seq(theta0)
    _jitted_par(theta0)
    _jitted_parareal(theta0)

    seq_elapsed = []
    par_elapsed = []
    parareal_elapsed = []
    for i in range(10):
        seq_start = time.time()
        theta_rk4_seq = _jitted_seq(theta0)
        theta_rk4_seq.block_until_ready()
        seq_end = time.time()
        seq_elapsed.append(seq_end - seq_start)

        par_start = time.time()
        theta_rk4_par = _jitted_par(theta0)
        theta_rk4_par.block_until_ready()
        par_end = time.time()
        par_elapsed.append(par_end - par_start)

        parareal_start = time.time()
        theta_G, t_G = _jitted_parareal(theta0)
        theta_G.block_until_ready()
        parareal_end = time.time()
        parareal_elapsed.append(parareal_end - parareal_start)

    mean_seq_elapsed.append(jnp.mean(jnp.array(seq_elapsed)))
    mean_par_elapsed.append(jnp.mean(jnp.array(par_elapsed)))
    mean_parareal_elapsed.append(jnp.mean(jnp.array(parareal_elapsed)))


mean_seq_elapsed_array = jnp.array(mean_seq_elapsed)
mean_par_elapsed_array = jnp.array(mean_par_elapsed)
mean_parareal_elapsed_array = jnp.array(mean_parareal_elapsed)

df_mean_seq = pd.DataFrame(mean_seq_elapsed)
df_mean_par = pd.DataFrame(mean_par_elapsed)
df_mean_parareal = pd.DataFrame(mean_parareal_elapsed)

df_mean_seq.to_csv("mean_seq_elapsed.csv")
df_mean_par.to_csv("mean_par_elapsed.csv")
df_mean_parareal.to_csv("mean_parareal_elapsed.csv")




