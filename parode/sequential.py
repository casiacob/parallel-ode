import jax.numpy as jnp
from typing import Callable
from jax import lax
from parode.sequential_integrators import integrators


def seq_integrate(
    ode: Callable, x0: jnp.ndarray, dt: float, nb_steps: int, method: str
) -> jnp.ndarray:
    integrator = integrators[method]

    def integrate_body(carry, inp):
        carry = integrator(ode, carry, dt)
        return carry, carry

    _, x = lax.scan(integrate_body, x0, xs=None, length=nb_steps)
    return jnp.vstack((x0, x))
