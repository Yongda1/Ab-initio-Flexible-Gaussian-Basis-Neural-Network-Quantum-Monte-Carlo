import jax
import jax.numpy as jnp
import numpy as np

def init_electrons(key, structure: jnp.array, atoms: jnp.array, charges: jnp.array,
                   electrons: jnp.array, batch_size: int, init_width: float) -> Tuple[jnp.array, jnp.array]:
    """Initializes electron positions around each atom.
    structure: the crystal structure, (lattice parameters, cell size).
    atoms: positions of the atoms.
    electrons: the array of alpha and beta electrons, i.e. spin configurations.
    batch_size: total number of Monte Carlo configurations to generate across all devices.
    init_width: width of atom-centred Gaussian used to generate initial electron configurations.
    This function needs be finished."""
    electrons_positions_batch = []
    for _ in range(batch_size):
        for i in range(len(atoms)):
            electrons_positions_batch.append(np.tile(atoms[i], int(charges[i])))

    """the following line has some problems. But it is still working now. We can make it better later."""
    electrons_positions_batch = np.hstack(np.array(electrons_positions_batch))
    electrons_positions_batch = jnp.reshape(jnp.array(electrons_positions_batch), (batch_size, -1))
    key, subkey = jax.random.split(key, num=2)
    electrons_positions_batch += (jax.random.normal(subkey, shape=electrons_positions_batch.shape) * init_width)
    "we need think about this. We need assign the spin configurations to electrons.12.08.2024."
    spins_no_batch = electrons
    #jax.debug.print("spins_no_batch:{}", spins_no_batch[None, ...])
    #spins_batch = jnp.repeat(spins_no_batch[None, ...], batch_size, axis=0)
    #jax.debug.print("spins_batch:{}", spins_batch)
    return electrons_positions_batch, spins_no_batch