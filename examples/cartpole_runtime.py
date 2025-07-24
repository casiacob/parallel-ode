import jax.numpy as jnp
from jax import random, jit
from parode.sequential import seq_integrate
from parode.parallel import par_integrate_rk4
from parode.parareal import parareal_integrate
from parode.integrators import rk4
import time
import pandas as pd

from jax import config

config.update("jax_enable_x64", True)
config.update("jax_platform_name", "cuda")

t0 = 0.0
tf = 5.0
x0 = jnp.array([0.0, jnp.pi/3, 0.0, 0.0])

mean_seq_elapsed = []
mean_par_elapsed = []
mean_parareal_elapsed = []

sampling_periods = [1e-2, 1e-3, 1e-4, 1e-5, 1e-6]

for dt in sampling_periods:
    t_eval = jnp.arange(t0, tf + dt, dt)
    nb_steps = len(t_eval) - 1


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

    _jitted_seq(x0)
    _jitted_par(x0)
    _jitted_parareal(x0)

    seq_elapsed = []
    par_elapsed = []
    parareal_elapsed = []
    for i in range(10):
        seq_start = time.time()
        X_rk4_seq = _jitted_seq(x0)
        X_rk4_seq.block_until_ready()
        seq_end = time.time()
        seq_elapsed.append(seq_end - seq_start)

        par_start = time.time()
        X_rk4_par = _jitted_par(x0)
        X_rk4_par.block_until_ready()
        par_end = time.time()
        par_elapsed.append(par_end - par_start)

        parareal_start = time.time()
        X_G, t_G = _jitted_parareal(x0)
        X_G.block_until_ready()
        parareal_end = time.time()
        parareal_elapsed.append(parareal_end - parareal_start)

    mean_seq_elapsed.append(jnp.mean(jnp.array(seq_elapsed)))
    mean_par_elapsed.append(jnp.mean(jnp.array(par_elapsed)))
    mean_parareal_elapsed.append(jnp.mean(jnp.array(parareal_elapsed)))


df_mean_seq = pd.DataFrame(mean_seq_elapsed)
df_mean_par = pd.DataFrame(mean_par_elapsed)
df_mean_parareal = pd.DataFrame(mean_parareal_elapsed)

df_mean_seq.to_csv("mean_seq_elapsed.csv")
df_mean_par.to_csv("mean_par_elapsed.csv")
df_mean_parareal.to_csv("mean_parareal_elapsed.csv")




