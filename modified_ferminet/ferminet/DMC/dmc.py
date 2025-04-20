from modified_ferminet.ferminet import networks as nn
import chex
import jax
import jax.numpy as jnp
from modified_ferminet.ferminet.DMC.drift_diffusion import propose_drift_diffusion
from modified_ferminet.ferminet.DMC.Tmoves import compute_tmoves
from modified_ferminet.ferminet.DMC.pseudopotential import pseudopotential
from modified_ferminet.ferminet.DMC.S_matrix import comput_S
from modified_ferminet.ferminet.DMC.energy import pphamiltonian
from modified_ferminet.ferminet.DMC.total_energy import calculate_total_energy


def dmc_propagate(signed_network,
                  log_network,
                  logabs_f,
                  list_l: int, 
                  nelectrons: int, 
                  natoms: int, 
                  ndim: int,
                  batch_size: int,
                  tstep: float,
                  nsteps: int,
                  charges: jnp.array,
                  spins: jnp.array,
                  Rn_local: jnp.array,
                  Local_coes: jnp.array,
                  Local_exps: jnp.array,
                  Rn_non_local: jnp.array, 
                  Non_local_coes: jnp.array, 
                  Non_local_exps: jnp.array):
    """we start to constuct the one loop dmc progagation process."""
    tmoves = compute_tmoves(list_l=list_l,
                            tstep=tstep,
                            nelectrons=nelectrons,
                            natoms=natoms,
                            ndim=ndim,
                            lognetwork=log_network,
                            Rn_non_local=Rn_non_local,
                            Non_local_coes=Non_local_coes,
                            Non_local_exps=Non_local_exps)

    tmoves_pmap = jax.pmap(
        jax.vmap(tmoves, in_axes=(nn.FermiNetData(positions=0, spins=0, atoms=0, charges=0), None, None)))

    drift_diffusion = propose_drift_diffusion(
        logabs_f=logabs_f,
        tstep=tstep,
        ndim=ndim,
        nelectrons=nelectrons,
        batch_size=batch_size)
    drift_diffusion_pmap = jax.pmap(drift_diffusion)

    localenergy = pphamiltonian.local_energy(f=signed_network,
                                             lognetwork=log_network,
                                             charges=charges,
                                             rn_local=Rn_local,
                                             local_coes=Local_coes,
                                             local_exps=Local_exps,
                                             rn_non_local=Rn_non_local,
                                             non_local_coes=Non_local_coes,
                                             non_local_exps=Non_local_exps,
                                             natoms=natoms,
                                             nelectrons=nelectrons,
                                             ndim=ndim,
                                             list_l=2,
                                             use_scan=False)

    total_e = calculate_total_energy(local_energy=localenergy)
    total_e_parallel = jax.pmap(total_e)

    def dmc_propagate_run(params: nn.ParamTree,
                          key: chex.PRNGKey,
                          data: nn.FermiNetData,
                          weights: jnp.array,
                          branchcut_start: jnp.array,
                          e_trial: jnp.array,
                          e_est: jnp.array,):
        pos, acceptance = tmoves_pmap(data, params, key)
        t_move_data = nn.FermiNetData(**(dict(data) | {'positions': pos}))
        new_data, newkey, tdamp, grad_eff_old, grad_new_eff = drift_diffusion_pmap(params, key, t_move_data)
        eloc_old, variance_old = total_e_parallel(params,  key, data)
        eloc_new, variance_new = total_e_parallel(params,  key, new_data)
        jax.debug.print("eloc_old:{}", eloc_old)
        jax.debug.print("eloc_new:{}", eloc_new)
        S_old = comput_S(e_trial=e_trial, e_est=e_est, branchcut=branchcut_start, v2=jnp.square(grad_eff_old), tau=tstep,
                         eloc=eloc_old, nelec=nelectrons)
        S_new = comput_S(e_trial=e_trial, e_est=e_est, branchcut=branchcut_start, v2=jnp.square(grad_new_eff), tau=tstep,
                         eloc=eloc_new, nelec=nelectrons)

        wmult = jnp.exp(tstep * tdamp * (0.5 * S_new + 0.5 * S_old))
        weights = wmult * weights
        return eloc_new, weights, new_data
    return dmc_propagate_run