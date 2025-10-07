import jax.numpy as jnp
from typing import Callable
import optimistix as optx
from optimistix import Newton

def euler(ode: Callable, x: jnp.ndarray, dt: float) -> jnp.ndarray:
    return x + ode(x) * dt


def rk4(ode: Callable, x: jnp.ndarray, dt: float) -> jnp.ndarray:
    k1 = ode(x) * dt
    k2 = ode(x + 0.5 * k1) * dt
    k3 = ode(x + 0.5 * k2) * dt
    k4 = ode(x + k3) * dt
    return x + (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0

def implicit_euler(ode: Callable, x: jnp.ndarray, dt: float) -> jnp.ndarray:
    def fn(y, args):
        return x + ode(y) * dt
    sol = optx.fixed_point(fn, Newton(rtol=1e-5, atol=1e-5), x)
    return sol.value

integrators = {
    "euler": euler,
    "rk4": rk4,
    "bdf": implicit_euler
}
