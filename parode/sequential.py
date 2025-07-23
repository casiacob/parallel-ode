import jax.numpy as jnp
from typing import Callable
from jax import lax


def seq_integrate(
    ode: Callable, x0: jnp.ndarray, dt: float, nb_steps: int, method: Callable
) -> jnp.ndarray:
    def integrate_body(carry, inp):
        carry = method(ode, carry, dt)
        return carry, carry

    _, x = lax.scan(integrate_body, x0, xs=None, length=nb_steps)
    return jnp.vstack((x0, x))
