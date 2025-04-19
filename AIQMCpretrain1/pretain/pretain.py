from typing import Callable, Mapping, Sequence, Tuple, Union, Optional

import jax
import optax
from absl import logging
import chex
#from AIQMCpretrain1.wavefunction_Ynlm import nn
from AIQMCpretrain1.wavefunction import networks as nn
from AIQMCpretrain1.VMC import mcmc
from AIQMCpretrain1 import constants
from AIQMCpretrain1.utils import scf
import jax.numpy as jnp
import kfac_jax
import pyscf
from AIQMCpretrain1.utils import system
import numpy as np

def get_hf(molecule: Optional[Sequence[system.Atom]] = None,
           nspins: Optional[Tuple[int, int]] = None,
           basis: Optional[str] = 'sto-3g',
           ecp: Optional[Mapping[str, str]] = None,
           core_electrons: Optional[Mapping[str, int]] = None,
           pyscf_mol: Optional[pyscf.gto.Mole] = None,
           restricted: Optional[bool] = False,
           states: int = 0,
           excitation_type: str = 'ordered') -> scf.Scf:
  """Returns an Scf object with the Hartree-Fock solution to the system.

  Args:
    molecule: the molecule in internal format.
    nspins: tuple with number of spin up and spin down electrons.
    basis: basis set to use in Hartree-Fock calculation.
    ecp: dictionary of the ECP to use for different atoms.
    core_electrons: dictionary of the number of core electrons excluded by the
      pseudopotential/effective core potential.
    pyscf_mol: pyscf Mole object defining the molecule. If supplied,
      molecule, nspins and basis are ignored.
    restricted: If true, perform a restricted Hartree-Fock calculation,
      otherwise perform an unrestricted Hartree-Fock calculation.
    states: Number of excited states.  If nonzero, compute all single and double
      excitations of the Hartree-Fock solution and return coefficients for the
      lowest ones.
    excitation_type: The way to construct different states for excited state
      pretraining. One of 'ordered' or 'random'. 'Ordered' tends to work better,
      but 'random' is necessary for some systems, especially double excitaitons.
  """
  if pyscf_mol:
    scf_approx = scf.Scf(pyscf_mol=pyscf_mol,
                         restricted=restricted)
  else:
    scf_approx = scf.Scf(molecule,
                         nelectrons=nspins,
                         basis=basis,
                         ecp=ecp,
                         core_electrons=core_electrons,
                         restricted=restricted)
  scf_approx.run(excitations=max(states - 1, 0),
                 excitation_type=excitation_type)
  return scf_approx


def eval_orbitals(scf_approx: scf.Scf, pos: Union[np.ndarray, jnp.ndarray],
                  nspins: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
  """Evaluates SCF orbitals from PySCF at a set of positions.

  Args:
    scf_approx: an scf.Scf object that contains the result of a PySCF
      calculation.
    pos: an array of electron positions to evaluate the orbitals at, of shape
      (..., nelec*3), where the leading dimensions are arbitrary, nelec is the
      number of electrons and the spin up electrons are ordered before the spin
      down electrons.
    nspins: tuple with number of spin up and spin down electrons.

  Returns:
    tuple with matrices of orbitals for spin up and spin down electrons, with
    the same leading dimensions as in pos.
  """
  if not isinstance(pos, np.ndarray):  # works even with JAX array
    try:
      pos = pos.copy()
    except AttributeError as exc:
      raise ValueError('Input must be either NumPy or JAX array.') from exc
  leading_dims = pos.shape[:-1]
  # split into separate electrons
  pos = np.reshape(pos, [-1, 3])  # (batch*nelec, 3)
  mos = scf_approx.eval_mos(pos)  # (batch*nelec, nbasis), (batch*nelec, nbasis)
  # Reshape into (batch, nelec, nbasis) for each spin channel.
  mos = [np.reshape(mo, leading_dims + (sum(nspins), -1)) for mo in mos]
  # Return (using Aufbau principle) the matrices for the occupied alpha and
  # beta orbitals. Number of alpha electrons given by nspins[0].
  alpha_spin = mos[0][..., :nspins[0], :nspins[0]]
  beta_spin = mos[1][..., nspins[0]:, :nspins[1]]
  return alpha_spin, beta_spin


def make_pretrain_step(
        batch_orbitals: nn.OrbitalFnLike,
        batch_network: nn.LogAINetLike,
        optimizer_update: optax.TransformUpdateFn,
        electrons: Tuple[int, int],
        batch_size: int = 0,
        full_det: bool = True,
        scf_fraction: float = 0.0,
):
    """creates function for performing one step of Hartree_Fock pretrain."""
    if scf_fraction > 1 or scf_fraction < 0:
        raise ValueError('scf_fraction must be in between 0 and 1, inclusive. ')
    scf_network = lambda fn, x: fn(x, electrons)[1]

    if scf_fraction < 1e-6:
        def mcmc_network(full_params, pos, spins, atoms, charges):
            return batch_network(full_params['AINet'], pos, spins, atoms, charges)
    elif scf_fraction > 0.999999:
        def mcmc_network(full_params, pos, spins, atoms, charges):
            del spins, atoms, charges
            return scf_network(full_params['scf'].eval_slater, pos)
    else:
        def mcmc_network(full_params, pos, spins, atoms, charges):
            log_ferminet = batch_network(full_params['AINet'], pos, spins, atoms, charges)
            log_scf = scf_network(full_params['scf'].eval_slater, pos)
            return (1 - scf_fraction) * log_ferminet + scf_fraction * log_scf
    '''
    mcmc_step = VMCmcstep.main_monte_carlo(mcmc_network,  tstep=0.02, ndim=3, nelectrons=6, nsteps=1,
                                           batch_size=int(batch_size / (1 * 1)))
    '''

    mcmc_step = mcmc.make_mcmc_step(
        mcmc_network, batch_per_device=batch_size, steps=1)

    def loss_fn(params: nn.ParamTree, data: nn.AINetData, scf_approx: scf.Scf):
        pos = data.positions
        spins = data.spins
        scf_orbitals = scf_approx.eval_orbitals
        net_orbitals = batch_orbitals

        target = scf_orbitals(pos, electrons)
        orbitals = net_orbitals(params, pos, spins, data.atoms, data.charges)
        """to be continued... 17.4.2025."""
        cnorm = lambda x, y: (x -y) * jnp.conj(x - y)
        if full_det:
            dims = target[0].shape[:-2]  # (batch) or (batch, states).
            na = target[0].shape[-2]
            nb = target[1].shape[-2]
            target = jnp.concatenate(
                (
                    jnp.concatenate(
                        (target[0], jnp.zeros(dims + (na, nb))), axis=-1),
                    jnp.concatenate(
                        (jnp.zeros(dims + (nb, na)), target[1]), axis=-1),
                ),
                axis=-2,
            )
            result = jnp.mean(cnorm(target[:, None, ...], orbitals[0])).real
        else:
            result = jnp.array([
                jnp.mean(cnorm(t[:, None, ...], o)).real
                for t, o in zip(target, orbitals)
            ]).sum()
        #jax.debug.print("result:{}", result)
        return constants.pmean(result)

    def pretain_step(data, params, state, key, scf_approx):
        val_and_grad = jax.value_and_grad(loss_fn, argnums=0)
        loss_val, search_direction = val_and_grad(params, data, scf_approx)
        jax.debug.print("loss_val:{}", loss_val)
        search_direction = constants.pmean(search_direction)
        updates, state = optimizer_update(search_direction, state, params)
        params = optax.apply_updates(params, updates)
        full_params = {'AINet': params, 'scf': scf_approx}
        #jax.debug.print("data:{}", data)
        #jax.debug.print("full_params:{}", full_params)
        data, pmove = mcmc_step(full_params, data, key, width=0.02)
        return data, params, state, loss_val, pmove

    return pretain_step

    


def pretain_hartree_fock(*,
                         params: nn.ParamTree,
                         positions: jnp.array,
                         spins: jnp.array,
                         atoms: jnp.array,
                         charges: jnp.array,
                         batch_network: nn.AINetLike,
                         batch_orbitals: nn.OrbitalFnLike,
                         sharded_key: chex.PRNGKey,
                         electrons: Tuple[int, int],
                         scf_approx: scf.Scf,
                         iterations: int = 1000,
                         batch_size: int = 0,
                         logger: Optional[Callable[[int, float], None]] = None,
                         scf_fraction: float = 0.0,
                         states: int = 0,
                         ):
    """pretrain the orbitatls to get good initial guess."""
    optimizer = optax.adam(3.e-4)
    opt_state_pt = constants.pmap(optimizer.init)(params)
    pretain_step = make_pretrain_step(batch_orbitals,
                                      batch_network,
                                      optimizer.update,
                                      electrons=electrons,
                                      batch_size=batch_size,
                                      full_det=True,
                                      scf_fraction=scf_fraction,)
    pretain_step = constants.pmap(pretain_step)
    
    batch_spins = jnp.tile(spins[None], [positions.shape[1], 1])
    pmap_spins = kfac_jax.utils.replicate_all_local_devices(batch_spins)
    data = nn.AINetData(positions=positions, spins=pmap_spins, atoms=atoms, charges=charges)

    for t in range(iterations):
        sharded_key, subkeys = kfac_jax.utils.p_split(sharded_key)
        data, params, opt_state_pt, loss, pmove = pretain_step(data, params, opt_state_pt, subkeys, scf_approx)
        logging.info('Pretain iter %05d: %g %g', t, loss[0], pmove[0])
        #jax.debug.print("loss:{}", loss)
        if logger:
            logger(t, loss[0])
    return params, data.positions