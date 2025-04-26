from AIQMC.wavefunction import networks as nn
import chex
import jax
import jax.numpy as jnp
from AIQMC.DMC.drift_diffusion import propose_drift_diffusion
from AIQMC.DMC.S_matrix import comput_S
from AIQMC.hamiltonian import hamiltonian
from AIQMC.DMC.total_energy import calculate_total_energy


def dmc_propagate(signed_network,
                  log_network,
                  logabs_f,
                  nelectrons: int, 
                  natoms: int, 
                  ndim: int,
                  batch_size: int,
                  tstep: float,
                  nsteps: int,
                  charges: jnp.array,
                  nspins: jnp.array,):
    """we start to constuct the one loop dmc progagation process."""

    drift_diffusion = propose_drift_diffusion(
        logabs_f=logabs_f,
        tstep=tstep,
        ndim=ndim,
        nelectrons=nelectrons,
        batch_size=batch_size)
    drift_diffusion_pmap = jax.pmap(drift_diffusion)

    localenergy = hamiltonian.local_energy(f=signed_network,
                                           charges=charges,
                                           nspins=nspins,
                                           use_scan=False,
                                           complex_output=True,
                                           laplacian_method='default')

    total_e = calculate_total_energy(local_energy=localenergy)
    total_e_parallel = jax.pmap(total_e)

    def dmc_propagate_run(params: nn.ParamTree,
                          key: chex.PRNGKey,
                          data: nn.FermiNetData,
                          weights: jnp.array,
                          branchcut_start: jnp.array,
                          e_trial: jnp.array,
                          e_est: jnp.array,):

        new_data, newkey, tdamp, grad_eff_old, grad_new_eff = drift_diffusion_pmap(params, key, data)
        eloc_old, variance_old = total_e_parallel(params,  key, data)
        eloc_new, variance_new = total_e_parallel(params,  key, new_data)
        #jax.debug.print("eloc_old:{}", eloc_old)
        #jax.debug.print("eloc_new:{}", eloc_new)
        S_old = comput_S(e_trial=e_trial, e_est=e_est, branchcut=branchcut_start, v2=jnp.square(grad_eff_old), tau=tstep,
                         eloc=eloc_old, nelec=nelectrons)
        S_new = comput_S(e_trial=e_trial, e_est=e_est, branchcut=branchcut_start, v2=jnp.square(grad_new_eff), tau=tstep,
                         eloc=eloc_new, nelec=nelectrons)

        wmult = jnp.exp(tstep * tdamp * (0.5 * S_new + 0.5 * S_old))
        weights = wmult * weights
        return eloc_new, weights, new_data
    return dmc_propagate_run