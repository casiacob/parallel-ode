import jax.numpy as jnp
from jax import lax, vmap
from parode.sequential import seq_integrate
from typing import Callable


def parareal_integrate(
    ode: Callable,
    x0: jnp.ndarray,
    dt: float,
    nb_steps: int,
    coarse_solver: Callable,
    fine_solver: Callable,
):
    nb_steps_G = 20
    nb_steps_F = int(nb_steps / nb_steps_G)
    dt_G = nb_steps_F * dt
    X_init = seq_integrate(ode, x0, dt_G, nb_steps_G, coarse_solver)

    def body(carry, inp):
        U = carry
        F, G = inp
        U = coarse_solver(ode, U, dt_G) + F - G
        return U, U

    def while_body(val):
        X_G, _ = val
        X_F = vmap(seq_integrate, in_axes=(None, 0, None, None, None))(
            ode, X_G[:-1], dt, nb_steps_F, fine_solver
        )[:, -1, :]
        X_F = jnp.vstack((x0, X_F))
        _, X_G_new = lax.scan(body, x0, (X_F[:-1], X_G[:-1]))
        X_G_new = jnp.vstack((x0, X_G_new))
        diff = jnp.max(jnp.abs(X_G - X_G_new))
        return X_G_new, diff

    def while_cond(val):
        _, diff = val
        return jnp.logical_not(diff < 1e-4)

    X, _ = lax.while_loop(
        while_cond,
        while_body,
        (
            X_init,
            jnp.array(1e3),
        ),
    )
    t_eval = jnp.arange(0, (nb_steps + 1) * dt, dt_G)
    return X, t_eval
