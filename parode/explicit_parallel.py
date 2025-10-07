import jax.numpy as jnp
from jax import vmap, jacfwd, lax
# from jax import debug


def rk4_step(ode, x, dt):
    k1 = ode(x) * dt
    k2 = ode(x + 0.5 * k1) * dt
    k3 = ode(x + 0.5 * k2) * dt
    k4 = ode(x + k3) * dt
    return (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0


def euler_step(ode, x, dt):
    return ode(x) * dt


step_functions = {
    "euler": euler_step,
    "rk4": rk4_step,
}


def affine_terms(X, x0, ode, dt, step):
    X_prev = jnp.vstack((x0, X[:-1]))
    return -X + X_prev + vmap(step, in_axes=(None, 0, None))(ode, X_prev, dt)


def linear_terms(X, ode, dt, step):
    J = vmap(jacfwd(step, 1), in_axes=(None, 0, None))(ode, X[:-1], dt)
    I = jnp.eye(X.shape[1])
    return I + J


def combine_fc(elem1, elem2):
    Fij, cij = elem1
    Fjk, cjk = elem2

    Fik = Fjk @ Fij
    cik = Fjk @ cij + cjk
    return Fik, cik


def par_scan(elems):
    return lax.associative_scan(vmap(combine_fc), elems, reverse=False)


def par_init(F, c, x0):
    tF0 = jnp.zeros_like(F[0])
    tc0 = F[0] @ x0 + c[0]
    tF = jnp.vstack((tF0.reshape(1, tF0.shape[0], tF0.shape[1]), F[1:]))
    tc = jnp.vstack((tc0, c[1:]))
    elems = (tF, tc)
    return elems


def newton_step(X, x0, ode, dt, step):
    c = affine_terms(X, x0, ode, dt, step)
    F = linear_terms(X, ode, dt, step)
    elems = par_init(F, c[1:], c[0])
    elems = par_scan(elems)
    return jnp.vstack((c[0], elems[1])), jnp.max(jnp.abs(c))


def explicit_par_integrate(ode, x0, initial_guess, dt, max_iter, method):
    step = step_functions[method]

    def while_body(val):
        X, _, iteration = val
        X_N, err = newton_step(X, x0, ode, dt, step)
        # debug.print('{x}', x=err)
        X_new = X + X_N
        iteration += 1
        return X_new, err, iteration

    def while_cond(val):
        _, err, iteration = val
        exit_condition = iteration > max_iter
        return jnp.logical_not(exit_condition)

    sol, residual, nb_iterations = lax.while_loop(
        while_cond, while_body, (initial_guess, jnp.array(1e3), 0)
    )
    return jnp.vstack((x0, sol)), residual, nb_iterations
