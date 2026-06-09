import jax.numpy as jnp
from jax import lax, vmap
from parode.sequential import seq_integrate
from typing import Callable
from jax import debug


def parareal_integrate(
    ode: Callable,
    x0: jnp.ndarray,
    nb_steps_G: int,
    nb_steps_F: int,
    dt: float,
    tol: float,
    coarse_solver: str,
    fine_solver: str,
):
    dt_G = nb_steps_F * dt
    X_init = seq_integrate(ode, x0, dt_G, nb_steps_G, coarse_solver)

    def body(carry, inp):
        U = carry
        F, G = inp
        U = seq_integrate(ode, U, dt_G, nb_steps=1, method=coarse_solver)[1] + F - G
        return U, U

    def while_body(val):
        X_G, iteration, _ = val

        X_dense = vmap(seq_integrate, in_axes=(None, 0, None, None, None))(
            ode, X_G[:-1], dt, nb_steps_F, fine_solver
        )
        X_F = X_dense[:, -1, :]

        res_inf = jnp.max(jnp.abs(X_G[1:] - X_F))

        X_G_old_dense = vmap(seq_integrate, in_axes=(None, 0, None, None, None))(
            ode, X_G[:-1], dt_G, 1, coarse_solver
        )
        X_G_old = X_G_old_dense[:, -1, :]

        _, X_G_new = lax.scan(body, x0, (X_F, X_G_old))
        X_G_new = jnp.vstack((x0, X_G_new))
        iteration += 1
        return X_G_new, iteration, res_inf

    def while_cond(val):
        _, iteration, res = val
        exit_condition = res < tol
        return jnp.logical_not(exit_condition)

    X, nb_iterations, error = lax.while_loop(
        while_cond,
        while_body,
        (X_init, 0, 1000.0),
    )
    return X
