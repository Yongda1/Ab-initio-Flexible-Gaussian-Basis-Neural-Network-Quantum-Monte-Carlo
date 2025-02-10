import jax.numpy as jnp


def comput_S(e_trial: float, e_est: float, branchcut: float, v2: jnp.array, tau: float, eloc: jnp.array, nelec: int):
    """
    :param e_trial: E_trial = E_est - feedback * log(jnp.mean(weights)).real
    :param e_est: E_est is the mean value. E_est = E_local * weight
    :param branchcut: 
    :param v2: V2 = |\nabla log\psi|**2
    :param tau: taus is the step length.
    :param eloc: E_local is the current energy of walkers applied in the calculation.
    :param nelec: number of electrons.
    :return: S matrix.
    """
    v2 = jnp.sum(v2, axis=-1)
    eloc = jnp.real(eloc)
    e_est = jnp.real(e_est)
    e_trial = jnp.real(e_trial)
    e_cut = e_est-eloc
    e_cut = jnp.min(jnp.array([jnp.abs(e_cut[0]), branchcut]))*jnp.sign(e_cut)
    denominator = 1 + (v2 * tau/nelec) ** 2
    return e_trial - e_est + e_cut/denominator