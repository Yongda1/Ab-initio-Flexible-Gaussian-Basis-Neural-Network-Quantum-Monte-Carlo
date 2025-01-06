"""Evaluates the Hamiltonian on a wavefunction."""

from typing import Any, Callable, Sequence, Union, Tuple, Optional
from AIQMCrelease1.utils import utils
import jax
import jax.numpy as jnp
import numpy as np
from typing_extensions import Protocol
from AIQMCrelease1.wavefunction import nn
import chex
from AIQMCrelease1.pseudopotential import pp_energy_test
from AIQMCrelease1.pseudopotential import pseudopotential
from AIQMCrelease1.main import main_adam
import kfac_jax




Array = Union[jnp.ndarray, np.ndarray]


class LocalEnergy(Protocol):
    def __call__(self, params: nn.ParamTree, key: chex.PRNGKey, data: nn.AINetData) -> Tuple[jnp.array, Optional[jnp.array]]:
        """Returns the local energy of a Hamiltonian at a configuration."""


class LocalEnergyDMC(Protocol):
    def __call__(self, params: nn.ParamTree, key:chex.PRNGKey, pos: jnp.array, atoms: jnp.array, charges: jnp.array) -> Tuple[jnp.array, Optional[jnp.array]]:
        """Returns the DMC local energy of a Hamiltonian at a configurations. Only change the parallel stragety."""


class MakeLocalEnergy(Protocol):
    def __call__(self, f: nn.AINetLike, **kwargs: Any) -> LocalEnergy:
        """Builds the LocalEnergy function."""


KineticEnergy = Callable[[nn.ParamTree, nn.AINetData], jnp.array]
KineticEnergy_DMC = Callable[[nn.ParamTree, jnp.array, jnp.array, jnp.array], jnp.array]

def local_kinetic_energy(f: nn.AINetLike) -> KineticEnergy:
    """Create the function for the local kinetic energy, -1/2 \nabla^2 ln|f|.
    29.08.2024 here our codes will be completely different from other codes due to the introduction of angular functions.
    I need take some notes on my slides.
    29.08.2024 I dont understand angular functions, complex number? how to calculate kinetic energy by real number? Why is it a real number?"""
    phase_f = utils.select_output(f, 0)
    logabs_f = utils.select_output(f, 1)
    second_grad_value = jax.jacfwd(jax.jacrev(logabs_f, argnums=1), argnums=1)
    angle_grad_hessian = jax.jacfwd(jax.jacrev(phase_f, argnums=1), argnums=1)
    grad_f = jax.value_and_grad(phase_f, argnums=1)
    grad_angle = jax.value_and_grad(logabs_f, argnums=1)

    def _lapl_over_f(params, data):
        hessian_value_logabs = second_grad_value(params, data.positions, data.atoms, data.charges)
        hessian_value_angle_f = angle_grad_hessian(params, data.positions, data.atoms, data.charges)
        value_angle, first_derivative_angle = grad_angle(params, data.positions, data.atoms, data.charges)
        value_f, first_derivative_f = grad_f(params, data.positions, data.atoms, data.charges)
        kinetic_energy = jnp.sum(hessian_value_logabs) + 1.j * jnp.sum(hessian_value_angle_f) +\
                         jnp.sum(jnp.square(first_derivative_f)) - jnp.sum(jnp.square(first_derivative_angle)) + \
                         jnp.sum(1.j * 2 * first_derivative_angle * first_derivative_f)
        return kinetic_energy

    return _lapl_over_f


"""we solve the VMC part first.26.12.2024."""
def local_kinetic_energy_DMC(f: nn.AINetLike) -> KineticEnergy_DMC:
    """Create the function for the local kinetic energy, -1/2 \nabla^2 ln|f|.
    29.08.2024 here our codes will be completely different from other codes due to the introduction of angular functions.
    I need take some notes on my slides.
    29.08.2024 I dont understand angular functions, complex number? how to calculate kinetic energy by real number? Why is it a real number?"""
    phase_f = utils.select_output(f, 0)
    logabs_f = utils.select_output(f, 1)

    def _lapl_over_f_DMC(params, pos: jnp.array, atoms: jnp.array, charges: jnp.array):
        second_grad_value = jax.jacfwd(jax.jacrev(logabs_f, argnums=1), argnums=1)
        hessian_value_logabs = second_grad_value(params, pos, atoms, charges)
        angle_grad_hessian = jax.jacfwd(jax.jacrev(phase_f, argnums=1), argnums=1)
        hessian_value_angle_f = angle_grad_hessian(params, pos, atoms, charges)
        value_angle, first_derivative_angle = jax.value_and_grad(phase_f, argnums=1)(params, pos, atoms, charges)
        value_f, first_derivative_f = jax.value_and_grad(logabs_f, argnums=1)(params, pos, atoms, charges)
        third_term = jnp.sum(1.j * 2 * first_derivative_angle * first_derivative_f)
        second_term = jnp.sum(jnp.square(first_derivative_f)) - jnp.sum(jnp.square(first_derivative_angle))
        first_term = jnp.sum(hessian_value_logabs) + 1.j * np.sum(hessian_value_angle_f)
        kinetic_energy = first_term + second_term + third_term
        return kinetic_energy

    return _lapl_over_f_DMC

"""all electron interaction calculation."""


def potential_electron_electron(r_ee: jnp.array) -> jnp.array:
    """electron-electron interaction energy."""
    r_ee = r_ee[jnp.triu_indices_from(r_ee[..., 0], 1)]
    return (1.0 / r_ee).sum()



def potential_electron_nuclear(r_ae: jnp.array, charges: jnp.array) -> jnp.array:
    """electron-atom interaction energy"""
    return -jnp.sum(charges / r_ae[..., 0])


def potential_nuclear_nuclear(atoms: jnp.array, charges: jnp.array) -> jnp.array:
    """atom-atom interaction energy
    to be continued 3.11.2024."""
    r_aa = jnp.linalg.norm(atoms[None, ...] - atoms[:, None], axis=-1)
    return jnp.sum(
        jnp.triu((charges[None, ...] * charges[..., None]) / r_aa, k=1))


def local_energy(signed_network: nn.AINetLike,
                 abslognetwork: nn.AINetLike,
                 Rn_local: jnp.array,
                 Local_coes: jnp.array,
                 Local_exps: jnp.array,
                 Rn_non_local: jnp.array,
                 Non_local_coes: jnp.array,
                 Non_local_exps: jnp.array,
                 batch_size: int,
                 natoms: int,
                 nelectrons: int,
                 ndim: int) -> LocalEnergy:
    """create the function to evaluate the local energy.
    default is complex number.
    f: signednetwork
    we leave the interface for pseudopotential calculation to be continued.
    28.10.2024, we still keep batch_atoms and batch_charges as the inputs.
    Because the optimizer do not accept the extra input, i.e. batch_atoms and batch_charges. We have to remove them.
    """
    lap_over_f = local_kinetic_energy(signed_network)
    """we can control the pseudopotential energy parallel here."""
    get_local_part_energy_test = pseudopotential.local_pp_energy(nelectrons=nelectrons,
                                                                 natoms=natoms,
                                                                 ndim=ndim,
                                                                 rn_local=Rn_local,
                                                                 local_coefficient=Local_coes,
                                                                 local_exponent=Local_exps)

    get_non_local_coe_test = pseudopotential.get_non_v_l(ndim=ndim,
                                                         nelectrons=nelectrons,
                                                         natoms=natoms,
                                                         rn_non_local=Rn_non_local,
                                                         non_local_coefficient=Non_local_coes,
                                                         non_local_exponent=Non_local_exps)

    generate_points_information_test = pseudopotential.get_P_l(nelectrons=nelectrons,
                                                               natoms=natoms,
                                                               ndim=ndim,
                                                               log_network_inner=abslognetwork)



    #total_energy_function_test_parallel = jax.pmap(jax.vmap(total_energy_function_test,
    #                                                        in_axes=(
    #                                                            nn.AINetData(positions=0, atoms=0, charges=0),
    #                                                            None,)))

    def _e_l(params: nn.ParamTree, key: chex.PRNGKey, data: nn.AINetData) -> Tuple[jnp.array, Optional[jnp.array]]:
        """after we change the parallel, we also need rewrite this part. we will solve this later, 31.10.2024.
        This is the version for test. Be careful of the input. we need change the last three variabls to data.3.11.2024.
        We leave it to monday."""

        ee = jnp.reshape(data.positions, [1, -1, ndim]) - jnp.reshape(data.positions, [-1, 1, ndim])
        ae = jnp.reshape(data.positions, [-1, 1, ndim]) - data.atoms[None, ...]
        r_ae = jnp.linalg.norm(ae, axis=-1)
        r_ee = jnp.linalg.norm(ee, axis=-1)
        r_ee = jnp.reshape(r_ee, (nelectrons, nelectrons, 1))
        r_ae = jnp.reshape(r_ae, (nelectrons, natoms, 1))
        kinetic = lap_over_f(params, data)
        potential_ee = potential_electron_electron(r_ee)
        """we need improve this function.make it be static as well."""
        total_energy_function_test = pp_energy_test.total_energy_pseudopotential(
            get_local_pp_energy=get_local_part_energy_test,
            get_nonlocal_pp_coes=get_non_local_coe_test,
            get_P_l=generate_points_information_test,
            log_network=abslognetwork,
            nelectrons=nelectrons,
            natoms=natoms,
            ndim=ndim,
            list_l=2,
            batch_size=4,
            key=key)
        pp_energy_value = total_energy_function_test(data, params)
        """maybe we could generate the integration points here. Just before the pp energy calculation."""
        potential_ae = potential_electron_nuclear(r_ae, charges=data.charges)
        potential_aa = potential_nuclear_nuclear(charges=data.charges, atoms=data.atoms)
        total_energy = kinetic + potential_ee + potential_aa + potential_ae
        energy_mat = None
        return total_energy, energy_mat

    return _e_l


def local_energy_dmc(f: nn.AINetLike) -> LocalEnergyDMC:
    lap_over_f_DMC = local_kinetic_energy_DMC(f)

    def _e_l_dmc(params: nn.ParamTree, key: chex.PRNGKey, pos: jnp.array, atoms: jnp.array, charges: jnp.array) -> Tuple[jnp.array, Optional[jnp.array]]:
        """after we change the parallel, we also need rewrite this part. we will solve this later, 31.10.2024.
        This is the version for test. Be careful of the input. we need change the last three variabls to data.3.11.2024.
        We leave it to monday."""
        """we already created the correct batch version. We do not need reshape the array, data.atoms and data.charges."""
        ndim = 3
        ee = jnp.reshape(pos, [1, -1, ndim]) - jnp.reshape(pos, [-1, 1, ndim])
        ae = jnp.reshape(pos, [-1, 1, ndim]) - atoms[None, ...]
        r_ae = jnp.linalg.norm(ae, axis=-1)
        r_ee = jnp.linalg.norm(ee, axis=-1)
        r_ee = jnp.reshape(r_ee, (4, 4, 1))
        r_ae = jnp.reshape(r_ae, (4, 2, 1))
        kinetic = lap_over_f_DMC(params, pos, atoms, charges)
        potential_ee = potential_electron_electron(r_ee)
        potential_ae = potential_electron_nuclear(r_ae, charges=charges)
        potential_aa = potential_nuclear_nuclear(charges=charges, atoms=atoms)
        total_energy = kinetic + potential_ee + potential_aa + potential_ae
        return total_energy

    return _e_l_dmc




structure = jnp.array([[10, 0, 0],
                       [0, 10, 0],
                       [0, 0, 10]])
Symbol = ['C', 'O', 'O']
atoms = jnp.array([[1.33, 1.0, 1.0], [0.0, 1.0, 1.0], [2.66, 1.0, 1.0]])
charges = jnp.array([4.0, 6.0, 6.0])
spins = jnp.array([1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0])
signed_network, data, batch_params, lognetwork = main_adam.main(atoms=atoms,
                                                                charges=charges,
                                                                spins=spins,
                                                                nelectrons=16,
                                                                natoms=3,
                                                                ndim=3,
                                                                batch_size=4,
                                                                iterations=1,
                                                                structure=structure)
key = jax.random.PRNGKey(1)
sharded_key = kfac_jax.utils.make_different_rng_key_on_all_devices(key)
jax.debug.print("sharded_key:{}", sharded_key)
sharded_key, subkeys = kfac_jax.utils.p_split(sharded_key)
symbol = ['Ge', 'Si', 'O']
Rn_local = jnp.array([[1.0, 3.0, 2.0, 2.0], [1.0, 3.0, 2.0, 0.0], [1.0, 3.0, 2.0, 0.0]])
Rn_non_local = jnp.array([[[2.0, 2.0], [2.0, 2.0], [2.0, 2.0]],
                          [[2.0, 2.0], [2.0, 2.0], [0.0, 0.0]],
                          [[2.0, 0.0], [0.0, 0.0], [0.0, 0.0]]])
Local_coes = jnp.array([[4.0,      5.9158506497680, -12.033712959815, 1.283543489065],
                        [4.000000, 20.673264,       -14.818174,       0],
                        [6.000000, 73.85984,        -47.87600,        0]])
Local_exps = jnp.array([[1.478962662442, 3.188905647765, 1.927438978253, 1.545539235916],
                        [5.168316,       8.861690,       3.933474,       0],
                        [12.30997,       14.76962,       13.71419,       0]])
Non_local_coes = jnp.array([[[43.265429324814, -1.909339873965], [35.263014141211, 0.963439928853], [2.339019442484, 0.541380654081]],
                            [[14.832760,       26.349664],       [7.621400,        10.331583],      [0,              0]],
                            [[85.86406,         0],               [0,               0],              [0,              0]]])
Non_local_exps = jnp.array([[[2.894473589836, 1.550339816290], [2.986528872039, 1.283381203893], [1.043001142249, 0.554562729807]],
                            [[9.447023,       2.553812],       [3.660001,       1.903653],       [0,              0]],
                            [[13.65512,        0],             [0,              0],              [0,              0]]])


localenergy = local_energy(signed_network=signed_network,
                           abslognetwork=lognetwork,
                           Rn_local=Rn_local,
                           Local_coes=Local_coes,
                           Local_exps=Local_exps,
                           Rn_non_local=Rn_non_local,
                           Non_local_coes=Non_local_coes,
                           Non_local_exps=Non_local_exps,
                           batch_size=4,
                           natoms=3,
                           nelectrons=16,
                           ndim=3)

batch_local_energy = jax.pmap(jax.vmap(localenergy, in_axes=(None, None, nn.AINetData(positions=0, atoms=0, charges=0)), out_axes=(0, 0)))
output = batch_local_energy(batch_params, subkeys, data)
#jax.debug.print("output:{}", output)
