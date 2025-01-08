import functools
import jax
import kfac_jax

PMAP_AXIS_NAME = 'qmc_pmap_axis'
pmap = functools.partial(jax.pmap, axis_name=PMAP_AXIS_NAME)
psum = functools.partial(kfac_jax.utils.psum_if_pmap, axis_name=PMAP_AXIS_NAME)
pmean = functools.partial(kfac_jax.utils.pmean_if_pmap, axis_name=PMAP_AXIS_NAME)
all_gather = functools.partial(kfac_jax.utils.wrap_if_pmap(jax.lax.all_gather), axis_name=PMAP_AXIS_NAME)