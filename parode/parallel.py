import jax.numpy as jnp
from jax import vmap, jacfwd, lax


def affine_terms(X, x0, ode, method, dt):
    X_prev = jnp.vstack((x0, X[:-1]))
    return X - vmap(method, in_axes=(None, 0, None))(ode, X_prev, dt)


def linear_terms(X, ode, method, dt):
    return vmap(jacfwd(method, 1), in_axes=(None, 0, None))(ode, X[:-1], dt)


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


def newton_step(X, x0, ode, dt, method):
    c = affine_terms(X, x0, ode, method, dt)
    F = linear_terms(X, ode, method, dt)
    elems = par_init(F, c[1:], c[0])
    elems = par_scan(elems)
    u = elems[1]
    u = jnp.vstack((c[0], u))
    return u, jnp.max(jnp.abs(c))


def par_integrate(ode, x0, dt, nb_steps, method):
    X0 = jnp.kron(jnp.ones((nb_steps, 1)), x0)

    def while_body(val):
        X, _, it_cnt = val
        u_N, err = newton_step(X, x0, ode, dt, method)
        X_new = X - u_N
        it_cnt += 1
        return X_new, err, it_cnt

    def while_cond(val):
        _, err, it_cnt = val
        exit_condition = err < 1e-4
        return jnp.logical_not(exit_condition)

    sol, _, _ = lax.while_loop(
        while_cond, while_body, (X0, jnp.array(1e3), jnp.array(0))
    )
    return jnp.vstack((x0, sol))
