import jax.numpy as jnp
from jax import jit
import jax
from parode.implicit_parallel import implicit_parallel_integrate
from parode.parareal import parareal_integrate
from parode.sequential import seq_integrate
import matplotlib.pyplot as plt
import time

from jax import config

config.update("jax_enable_x64", True)
config.update("jax_platform_name", "cuda")

y0 = jnp.array([1.0])
t0 = 0.0
tf = 4.0

Ts = [1e-1, 1e-2, 1e-3, 1e-4]
steps = []
avg_par_elapsed = []
avg_seq_elapsed = []
avg_parareal_elapsed = []
par_iterations = []
pr_iterations = []
tolerance = 1e-12


def dahlquist(y):
    a = -1.0
    return a * y


for i in range(len(Ts)):
    dt = Ts[i]
    t_eval = jnp.arange(t0, tf + dt, dt)
    nb_steps = len(t_eval) - 1
    coarse_solver_steps = jnp.int64(jnp.sqrt(nb_steps))
    fine_solver_steps = jnp.int64(nb_steps / coarse_solver_steps)
    initial_guess = jnp.zeros((nb_steps, y0.shape[0]))
    steps.append(nb_steps)
    annon_seq_integrate = lambda x0, h: seq_integrate(
        dahlquist, x0, h, nb_steps, method="bdf"
    )
    annon_par_integrate = lambda x0, h: implicit_parallel_integrate(
        dahlquist, x0, initial_guess, h, tol=tolerance, method="bdf"
    )
    annon_parareal_integrate = lambda x0, h: parareal_integrate(
        dahlquist,
        x0,
        coarse_solver_steps,
        fine_solver_steps,
        dt,
        tol=tolerance,
        coarse_solver="bdf",
        fine_solver="bdf",
    )
    _jitted_seq = jit(annon_seq_integrate)
    _jitted_par = jit(annon_par_integrate)
    _jitted_parareal = jit(annon_parareal_integrate)

    s_seq = _jitted_seq(y0, dt)
    s_par, par_nb_iterations = _jitted_par(y0, dt)
    s_parareal, pr_nb_iterations = _jitted_parareal(y0, dt)

    plt.plot(t_eval, s_seq)
    plt.plot(t_eval, s_par)
    t_eval_parareal = jnp.linspace(0, nb_steps * dt, coarse_solver_steps + 1)
    plt.plot(t_eval_parareal, s_parareal, marker="o")
    plt.show()

    par_iterations.append(par_nb_iterations)
    pr_iterations.append(pr_nb_iterations)

    par_times = []
    seq_times = []
    parareal_times = []

    for j in range(10):
        start = time.time()
        seq_sol = _jitted_seq(y0, dt)
        jax.block_until_ready(seq_sol)
        end = time.time()
        seq_elapsed = end - start
        seq_times.append(seq_elapsed)

        start = time.time()
        par_sol, _ = _jitted_par(y0, dt)
        jax.block_until_ready(par_sol)
        end = time.time()
        par_elapsed = end - start
        par_times.append(par_elapsed)

        start = time.time()
        parareal_sol, _ = _jitted_parareal(y0, dt)
        jax.block_until_ready(parareal_sol)
        end = time.time()
        parareal_elapsed = end - start
        parareal_times.append(parareal_elapsed)

    avg_seq_elapsed.append(jnp.mean(jnp.array(seq_times)))
    avg_par_elapsed.append(jnp.mean(jnp.array(par_times)))
    avg_parareal_elapsed.append(jnp.mean(jnp.array(parareal_times)))


plt.plot(steps, avg_seq_elapsed, marker="o", label="sequential")
plt.plot(steps, avg_par_elapsed, marker="o", label="parallel")
plt.plot(steps, avg_parareal_elapsed, marker="o", label="parareal")
plt.xscale("log")
plt.yscale("log")
plt.xlabel("steps")
plt.ylabel("runtime [s]")
plt.grid(True, which="both", ls="--")
plt.legend()
plt.show()


plt.plot(Ts, par_iterations, marker="o", label="parallel")
plt.plot(Ts, pr_iterations, marker="o", label="parareal")
plt.xscale("log")
plt.xlabel("T_s")
plt.ylabel("iterations")
plt.legend(loc="upper left")
plt.show()
