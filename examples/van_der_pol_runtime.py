import jax.numpy as jnp
from jax import jit
import jax
from parode.explicit_parallel import explicit_par_integrate
from parode.parareal import parareal_integrate
from parode.sequential import seq_integrate
import time
import matplotlib.pyplot as plt
from jax import config

config.update("jax_enable_x64", True)
config.update("jax_platform_name", "cuda")

s0 = jnp.array([0.0, 1.0])
t0 = 0.0
tf = 10.0

Ts = [1e-2, 1e-3, 1e-4, 1e-5]
steps = []
avg_par_elapsed = []
avg_seq_elapsed = []
avg_parareal_elapsed = []


def vdp(state):
    x, y = state
    mu = 1.0
    return jnp.hstack((y, mu * (1 - x**2) * y - x))


for i in range(len(Ts)):
    dt = Ts[i]
    t_eval = jnp.arange(t0, tf + dt, dt)
    nb_steps = len(t_eval) - 1
    coarse_solver_steps = jnp.int64(jnp.sqrt(nb_steps))
    fine_solver_steps = jnp.int64(nb_steps / coarse_solver_steps)
    initial_guess = jnp.ones((nb_steps, s0.shape[0]))
    steps.append(nb_steps)
    annon_seq_integrate = lambda x0, h: seq_integrate(
        vdp, x0, h, nb_steps, method="rk4"
    )
    annon_par_integrate = lambda x0, h: explicit_par_integrate(
        vdp, x0, initial_guess, h, max_iter = 10, method="rk4"
    )
    annon_parareal_integrate = lambda x0, h: parareal_integrate(
        vdp,
        x0,
        coarse_solver_steps,
        fine_solver_steps,
        dt,
        max_iter=10,
        coarse_solver="rk4",
        fine_solver="rk4",
    )
    _jitted_seq = jit(annon_seq_integrate)
    _jitted_par = jit(annon_par_integrate)
    _jitted_parareal = jit(annon_parareal_integrate)

    s_seq = _jitted_seq(s0, dt)
    s_par, par_status, nb_iterations = _jitted_par(s0, dt)
    s_parareal = _jitted_parareal(s0, dt)

    plt.plot(t_eval, s_seq)
    t_eval_parareal = jnp.linspace(0, nb_steps * dt, coarse_solver_steps + 1)
    plt.plot(t_eval_parareal, s_parareal, marker="o")
    plt.plot(t_eval, s_par)
    plt.show()

    par_times = []
    seq_times = []
    parareal_times = []

    for j in range(10):
        start = time.time()
        seq_sol = _jitted_seq(s0, dt)
        jax.block_until_ready(seq_sol)
        end = time.time()
        seq_elapsed = end - start
        seq_times.append(seq_elapsed)

        start = time.time()
        par_sol, par_residual, par_iterations = _jitted_par(s0, dt)
        jax.block_until_ready(par_sol)
        end = time.time()
        par_elapsed = end - start
        par_times.append(par_elapsed)

        start = time.time()
        parareal_sol = _jitted_parareal(s0, dt)
        jax.block_until_ready(parareal_sol)
        end = time.time()
        parareal_elapsed = end - start
        parareal_times.append(parareal_elapsed)

    avg_seq_elapsed.append(jnp.mean(jnp.array(seq_times)))
    avg_par_elapsed.append(jnp.mean(jnp.array(par_times)))
    avg_parareal_elapsed.append(jnp.mean(jnp.array(parareal_times)))


plt.plot(steps, avg_seq_elapsed, marker='o', label='sequential')
plt.plot(steps, avg_par_elapsed, marker='o', label='parallel')
plt.plot(steps, avg_parareal_elapsed, marker='o', label='parareal')
plt.xscale("log")
plt.yscale("log")
plt.xlabel('steps')
plt.ylabel('runtime [s]')
plt.grid(True, which="both", ls="--")
plt.legend()
plt.show()

