from GaussianNet.wavefunction import networks
import chex
import jax
import jax.numpy as jnp
from GaussianNet.DMC.drift_diffusion import propose_drift_diffusion
from GaussianNet.DMC.S_matrix import comput_S
from GaussianNet.hamiltonian import hamiltonian
from GaussianNet.DMC.total_energy import calculate_total_energy


def dmc_propagate(signed_network,
                  log_network,
                  nelectrons: int, 
                  natoms: int, 
                  ndim: int,
                  tstep: float,
                  charges: jnp.array,
                  nspins: jnp.array,):
    """we start to constuct the one loop dmc progagation process."""

    drift_diffusion = propose_drift_diffusion(f=signed_network,
                                              tstep=0.02,
                                              ndim=3,
                                              nelectrons=6,)
    drift_diffusion_parallel = jax.pmap(
        jax.vmap(drift_diffusion,
                 in_axes=(networks.GaussianNetData(positions=0, spins=0, atoms=0, charges=0), None, 0)),
        in_axes=(networks.GaussianNetData(positions=0, spins=0, atoms=0, charges=0), 0, 0))

    def generate_batch_key(batch_size: int):
        def get_keys(key: chex.PRNGKey):
            keys = jax.random.split(key, num=batch_size)
            return keys
        return get_keys

    localenergy = hamiltonian.local_energy(f=signed_network,
                                           charges=charges,
                                           nspins=nspins,
                                           use_scan=False,
                                           complex_output=True,
                                           laplacian_method='default')

    total_e = calculate_total_energy(local_energy=localenergy)
    total_e_parallel = jax.pmap(total_e)

    def dmc_propagate_run(params: networks.ParamTree,
                          key: chex.PRNGKey,
                          data: networks.GaussianNetData,
                          weights: jnp.array,
                          branchcut_start: jnp.array,
                          e_trial: jnp.array,
                          e_est: jnp.array,):
        generate_mc_keys = generate_batch_key(6)
        mc_keys = jax.pmap(generate_mc_keys)(key)

        eloc_old, variance_old = total_e_parallel(params, key, data)
        """to be continued...21.5.2025."""
        data, mc_keys, grad_eff_old, grad_new_eff = drift_diffusion_parallel(data, params, mc_keys)

        eloc_new, variance_new = total_e_parallel(params,  key, data)

        grad_eff_old = jnp.reshape(grad_eff_old, (6, -1))
        grad_new_eff = jnp.reshape(grad_new_eff, (6, -1))
        #jax.debug.print("eloc_old:{}", eloc_old)
        #jax.debug.print("eloc_new:{}", eloc_new)

        S_old = comput_S(e_trial=e_trial,
                         e_est=e_est,
                         branchcut=branchcut_start,
                         v2=jnp.square(jnp.abs(grad_eff_old)),
                         tau=tstep,
                         eloc=eloc_old,
                         nelec=nelectrons)

        S_new = comput_S(e_trial=e_trial,
                         e_est=e_est,
                         branchcut=branchcut_start,
                         v2=jnp.square(jnp.abs(grad_new_eff)),
                         tau=tstep,
                         eloc=eloc_new,
                         nelec=nelectrons)

        wmult = jnp.exp(tstep * (0.5 * S_new + 0.5 * S_old))
        weights = wmult * weights
        next_key_parallel = generate_batch_key(1)
        next_key = jax.pmap(next_key_parallel)(key)
        #jax.debug.print("next_key:{}", next_key)
        return eloc_new, weights, data, next_key[0]
    return dmc_propagate_run