import jax.numpy as jnp
from jax import vmap, jacfwd, lax, random
from jax import debug


def implicit_euler_step(ode, xk, xkp1, dt):
    return ode(xkp1) * dt


def implicit_trapezoidal_step(ode, xk, xkp1, dt):
    return 0.5 * (ode(xk) * dt + ode(xkp1) * dt)


step_functions = {
    "bdf": implicit_euler_step,
    "trapezoidal": implicit_trapezoidal_step,
}


def linear_and_affine_terms(X, x0, ode, dt, step):
    I = jnp.eye(x0.shape[0])
    X_prev = jnp.vstack((x0, X[:-1]))
    h = X - X_prev - vmap(step, in_axes=(None, 0, 0, None))(ode, X_prev, X, dt)
    step_jac_k = vmap(jacfwd(step, 2), in_axes=(None, 0, 0, None))(ode, X_prev, X, dt)
    step_jac_km1 = vmap(jacfwd(step, 1), in_axes=(None, 0, 0, None))(ode, X_prev, X, dt)
    K = I - step_jac_k
    R = -I - step_jac_km1[1:]
    K_inv = vmap(jnp.linalg.solve, in_axes=(0, None))(K, I)
    linear_terms = -vmap(lambda x, y: x @ y)(K_inv[1:], R)
    affine_terms = -vmap(lambda x, y: x @ y)(K_inv, h)
    return linear_terms, affine_terms


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
    F, c = linear_and_affine_terms(X, x0, ode, dt, step)
    elems = par_init(F, c[1:], c[0])
    elems = par_scan(elems)
    return jnp.vstack((c[0], elems[1])), jnp.max(jnp.abs(c))


def implicit_parallel_integrate(ode, x0, initial_guess, dt, max_iter, method):
    step = step_functions[method]

    def while_body(val):
        X, _, iteration = val
        X_N, err = newton_step(X, x0, ode, dt, step)
        debug.print('{y} {x}',y=iteration, x=err)
        X_new = X + X_N
        iteration += 1
        return X_new, err, iteration

    def while_cond(val):
        _, err, iteration = val
        exit_condition = iteration > max_iter
        return jnp.logical_not(exit_condition)

    sol, _, nb_iterations = lax.while_loop(
        while_cond, while_body, (initial_guess, jnp.array(1e3), 0)
    )
    return jnp.vstack((x0, sol))
