import functools
import jax
import kfac_jax


# Axis name we pmap over.
PMAP_AXIS_NAME = 'qmc_pmap_axis'

# Shortcut for jax.pmap over PMAP_AXIS_NAME. Prefer this if pmapping any
# function which does communications or reductions.
pmap = functools.partial(jax.pmap, axis_name=PMAP_AXIS_NAME)
psum = functools.partial(kfac_jax.utils.psum_if_pmap, axis_name=PMAP_AXIS_NAME)
pmean = functools.partial(
    kfac_jax.utils.pmean_if_pmap, axis_name=PMAP_AXIS_NAME)

all_gather = functools.partial(kfac_jax.utils.wrap_if_pmap(jax.lax.all_gather),
                               axis_name=PMAP_AXIS_NAME)