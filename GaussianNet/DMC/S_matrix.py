import jax
import jax.numpy as jnp


def comput_S(e_trial: float,
             e_est: float,
             branchcut: jnp.ndarray,
             v2: jnp.ndarray,
             tau: float,
             eloc: jnp.ndarray,
             nelec: int):
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
    #jax.debug.print("v2:{}", v2)
    eloc = jnp.real(eloc)
    e_est = jnp.real(e_est)
    e_trial = jnp.real(e_trial)
    e_cut = e_est-eloc
    #jax.debug.print("eloc:{}", eloc)
    #jax.debug.print("branchcut:{}", branchcut)
    e_cut = jnp.min(jnp.array([jnp.abs(e_cut), branchcut]))*jnp.sign(e_cut)
    #jax.debug.print("e_cut:{}", e_cut)
    denominator = 1 + (v2 * tau/nelec) ** 2
    return e_trial - e_est + e_cut/denominator

def comput_S_new(tau: float,
                 nelec: int):

    def compute_kernel(v2: jnp.array, eloc: jnp.array, e_trial: float, e_est: float, branchcut: float,):
        e_est = jnp.real(e_est)
        e_trial = jnp.real(e_trial)
        #v2 = jnp.sum(v2, axis=-1)
        eloc = jnp.real(eloc)
        e_cut = e_est-eloc
        e_cut = jnp.min(jnp.array([jnp.abs(e_cut), branchcut]))*jnp.sign(e_cut)
        denominator = 1 + (v2 * tau/nelec) ** 2
        return e_trial - e_est + e_cut/denominator
    return compute_kernel


def comput_S_normal(e_trial: float, eloc: jnp.array):
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
    return e_trial.real - eloc.real