from AIQMC.wavefunction import networks
from AIQMC.hamiltonian import hamiltonian
from AIQMC.tools.utils import utils
import jax.numpy as jnp
import chex
import jax



def limdrift(g, tau, acyrus):
    v2 = jnp.sum(g**2)
    taueff = (jnp.sqrt(1 + 2 * tau * acyrus * v2) - 1)/ (acyrus * v2)
    return g * taueff


def walkers_accept(x1, x2, acceptance, key, nelectrons: int, batch_size: int):
    key, subkey = jax.random.split(key)
    rnd = jax.random.uniform(subkey, shape=acceptance.shape, minval=0, maxval=1.0)
    cond = acceptance > rnd
    cond = jnp.reshape(cond, (batch_size, nelectrons, 1))
    x_new = jnp.where(cond, x2, x1)
    tdamp = jnp.sum(x_new) / jnp.sum(x1)
    return x_new, subkey, tdamp


def propose_drift_diffusion(logabs_f: networks.LogFermiNetLike,
                            tstep: float,
                            ndim: int,
                            nelectrons: int,
                            batch_size: int):
    def drift_diffusion(params, key: chex.PRNGKey, data: networks.FermiNetData):
        key, subkey = jax.random.split(key)
        x1 = data.positions
        grad_value = jax.grad(logabs_f, argnums=1)
        atoms = data.atoms[0]
        charges = data.charges[0]
        spins = data.spins[0]

        def grad_f_closure(x):
            return grad_value(params, x, spins, atoms, charges)

        # grad_test = jax.vmap(grad_value, in_axes=(None, 0, 0, 0, 0))(params, data.positions, data.spins, data.atoms, data.charges)
        # jax.debug.print("grad_test:{}", grad_test)
        grad_f = jax.vmap(grad_f_closure, in_axes=0)
        # jax.debug.print("x1:{}", x1)
        grad = grad_f(x1)
        # jax.debug.print("grad:{}", grad)
        initial_configuration = jnp.reshape(x1, (batch_size, nelectrons, ndim))
        x1 = jnp.reshape(jnp.reshape(x1, (batch_size, -1, ndim)), (batch_size, 1, -1))
        x1 = jnp.reshape(jnp.repeat(x1, nelectrons, axis=1), (batch_size, nelectrons, nelectrons, ndim))
        gauss = jnp.sqrt(tstep) * jax.random.normal(key=key, shape=(jnp.shape(grad)))

        grad_eff = limdrift(grad, tstep, 0.25)
        grad_eff_old = grad_eff

        g = grad_eff * tstep + gauss
        g = jnp.reshape(g, (batch_size, nelectrons, ndim))
        order = jnp.arange(0, nelectrons, step=1)
        order = jnp.repeat(order[None, ...], batch_size, axis=0)
        """maybe some thing is wrong here. 8.4.2025."""

        def change_configurations(order: jnp.array, g: jnp.array):
            z = jnp.zeros((nelectrons, ndim))
            temp = z.at[order].add(g[order])
            return temp

        change_configurations_parallel = jax.vmap(jax.vmap(change_configurations, in_axes=(0, None)), in_axes=(0, 0),
                                                  out_axes=0)
        # jax.debug.print("g:{}", g)
        z = change_configurations_parallel(order, g)
        # jax.debug.print("z:{}", z)
        x2 = x1 + z
        changed_configuration = g + initial_configuration
        x2 = jnp.reshape(x2, (batch_size, nelectrons, -1))
        grad_new = jax.vmap(jax.vmap(grad_f_closure, in_axes=0, out_axes=0), in_axes=0)(x2)
        grad_new_eff = limdrift(grad_new, tstep, 0.25)
        grad_eff = jnp.repeat(grad_eff, nelectrons, axis=0)
        grad_eff = jnp.reshape(grad_eff, (batch_size, nelectrons, -1))
        gauss = jnp.sqrt(tstep) * jax.random.normal(key=key, shape=(jnp.shape(grad_eff)))
        forward = gauss ** 2
        backward = (gauss + (grad_eff + grad_new_eff)) ** 2 # the drift-diffusion must be wrong. 27.4.2025.
        t_probability = jnp.exp((forward - backward) / (2 * tstep))
        t_probability = jnp.reshape(t_probability, (batch_size, nelectrons, nelectrons, ndim))
        t_probability = jnp.sum(t_probability, axis=-1)

        def return_t(t_pro_inner: jnp.array):
            return jnp.diagonal(t_pro_inner)

        return_t_parallel = jax.vmap(return_t, in_axes=0)
        t_pro = return_t_parallel(t_probability)
        logabs_f_vmap = jax.vmap(jax.vmap(logabs_f, in_axes=(None, 0, None, None, None,)),
                                 in_axes=(None, 0, None, None, None))
        wave_x2 = logabs_f_vmap(params, x2, spins, atoms, charges)
        x1 = jnp.reshape(x1, (batch_size, nelectrons, -1))
        wave_x1 = logabs_f_vmap(params, x1, spins, atoms, charges)
        wfratio = wave_x2 / wave_x1
        ratio = jnp.abs(wfratio) ** 2 * t_pro
        ratio *= jnp.sign(wfratio)
        #jax.debug.print("ratio:{}", ratio)
        acceptance = ratio
        final_configuration, newkey, tdamp = walkers_accept(initial_configuration,
                                                     changed_configuration,
                                                     acceptance,
                                                     key,
                                                     nelectrons,
                                                     batch_size)

        final_configuration = jnp.reshape(final_configuration, (batch_size, -1))
        new_data = networks.FermiNetData(**(dict(data) | {'positions': final_configuration}))

        grad_new_s = grad_f(new_data.positions)
        grad_new_eff_s = limdrift(grad_new_s, tstep, 0.25)

        return new_data, newkey, tdamp, grad_eff_old, grad_new_eff_s

    return drift_diffusion

def propose_drift_diffusion_new(f: networks.FermiNetLike,
                                tstep: float,
                                ndim: int,
                                nelectrons: int,
                                batch_size: int,
                                complex_output: bool):
    phase_f = utils.select_output(f, 0)
    logabs_f = utils.select_output(f, 1)

    def drift_diffusion_new(params, key: chex.PRNGKey, data: networks.FermiNetData):
        grad_f = jax.grad(logabs_f, argnums=1)
        def grad_f_closure(x):
            return grad_f(params, x, data.spins, data.atoms, data.charges)

        primal, dgrad_f = jax.linearize(grad_f_closure, data.positions)

        if complex_output:
            grad_phase = jax.grad(phase_f, argnums=1)

            def grad_phase_closure(x):
                return grad_phase(params, x, data.spins, data.atoms, data.charges)

            phase_primal, dgrad_phase = jax.linearize(
                grad_phase_closure, data.positions)

        O_old = primal + 1.j * phase_primal
        O_old = jnp.reshape(O_old, (nelectrons, ndim))
        x1 = data.positions
        x1 = jnp.reshape(x1, (nelectrons, ndim))
        x_new = jnp.zeros_like(x1)
        for i in range(len(x1)):
            key_inner, key_new_inner = jax.random.split(key)
            gauss = jnp.sqrt(tstep) * jax.random.normal(key=key_new_inner, shape=(jnp.shape(x1[i])))
            O_eff = limdrift(O_old[i], tstep, 0.25)
            temp = O_eff + gauss + x1[i]
            x2 = x1.at[i].set(temp)
            x_2_temp = jnp.reshape(x2, (-1))
            x_1_temp = jnp.reshape(x1, (-1))
            wave_x1_mag = logabs_f(params, x_1_temp, data.spins, data.atoms, data.charges)
            wave_x2_mag = logabs_f(params, x_2_temp, data.spins, data.atoms, data.charges)
            wave_x1_phase = phase_f(params, x_1_temp, data.spins, data.atoms, data.charges)
            wave_x2_phase = phase_f(params, x_2_temp, data.spins, data.atoms, data.charges)
            ratio = ((wave_x2_mag + 1.j*wave_x2_phase) / (wave_x1_mag + 1.j * wave_x1_phase)).real ** 2
            forward = jnp.sum(gauss**2)
            primal_x2, dgrad_f_x2 = jax.linearize(grad_f_closure, x_2_temp)
            phase_primal_x2, dgrad_phase_x2 = jax.linearize(grad_phase_closure, x_2_temp)
            O_new = primal_x2 + 1.j * phase_primal_x2
            O_new = jnp.reshape(O_new, (nelectrons, ndim))
            O_new_eff = limdrift(O_new[i], tstep, 0.25)
            backward = jnp.sum((gauss + O_eff + O_new_eff)**2)
            t_pro = jnp.exp(1/(2 * tstep) * (forward - backward))
            ratio_total = jnp.abs(ratio) * t_pro
            ratio_total = ratio_total * jnp.sign(ratio)
            rnd = jax.random.uniform(key, shape=ratio_total.shape, minval=0, maxval=1.0)
            cond = ratio_total > rnd
            x_new = x_new.at[i].set(jnp.where(cond, x2[i], x1[i]))


        x_new = jnp.reshape(x_new, (-1))
        new_data = networks.FermiNetData(**(dict(data) | {'positions': x_new}))
        new_key, new_key2 = jax.random.split(key)
        return new_data, new_key2
    return drift_diffusion_new

"""the following lines are for debuging."""