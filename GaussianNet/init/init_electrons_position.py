import jax
import jax.numpy as jnp
import numpy as np
from typing import Sequence, Mapping, Tuple
from GaussianNet.tools.utils import system
from absl import logging

def _assign_spin_configuration(nalpha: int, nbeta: int, batch_size: int = 1) -> jnp.ndarray:
    spins = jnp.concatenate((jnp.ones(nalpha), -jnp.ones(nbeta)))
    return jnp.tile(spins[None], reps=(batch_size, 1))

def init_electrons(  # pylint: disable=dangerous-default-value
    key,
    molecule: Sequence[system.Atom],
    electrons: Sequence[int],
    batch_size: int,
    init_width: float,
    core_electrons: Mapping[str, int] = {},
    max_iter: int = 10000,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    niter = 0
    total_electrons = sum(atom.charge - core_electrons.get(atom.symbol, 0)
                          for atom in molecule)
    if total_electrons != sum(electrons):
        if len(molecule) == 1:
            atomic_spin_configs = [electrons]
        else:
            raise NotImplementedError('No initialization policy yet '
                                      'exists for charged molecules.')
    else:
        atomic_spin_configs = [
            (atom.element.nalpha - core_electrons.get(atom.symbol, 0) // 2,
             atom.element.nbeta - core_electrons.get(atom.symbol, 0) // 2)
            for atom in molecule
        ]
        assert sum(sum(x) for x in atomic_spin_configs) == sum(electrons)
        while (
                tuple(sum(x) for x in zip(*atomic_spin_configs)) != electrons
                and niter < max_iter
        ):
            i = np.random.randint(len(atomic_spin_configs))
            nalpha, nbeta = atomic_spin_configs[i]
            atomic_spin_configs[i] = nbeta, nalpha
            niter += 1

    if tuple(sum(x) for x in zip(*atomic_spin_configs)) == electrons:
        # Assign each electron to an atom initially.
        electron_positions = []
        for i in range(2):
            for j in range(len(molecule)):
                atom_position = jnp.asarray(molecule[j].coords)
                electron_positions.append(
                    jnp.tile(atom_position, atomic_spin_configs[j][i]))
        electron_positions = jnp.concatenate(electron_positions)
    else:
        logging.warning(
            'Failed to find a valid initial electron configuration after %i'
            ' iterations. Initializing all electrons from a Gaussian distribution'
            ' centred on the origin. This might require increasing the number of'
            ' iterations used for pretraining and MCMC burn-in. Consider'
            ' implementing a custom initialisation.',
            niter,
        )
        electron_positions = jnp.zeros(shape=(3 * sum(electrons),))

    # Create a batch of configurations with a Gaussian distribution about each atom.
    key, subkey = jax.random.split(key)
    electron_positions += (
            jax.random.normal(subkey, shape=(batch_size, electron_positions.size))
            * init_width
    )

    electron_spins = _assign_spin_configuration(
        electrons[0], electrons[1], batch_size
    )

    return electron_positions, electron_spins
