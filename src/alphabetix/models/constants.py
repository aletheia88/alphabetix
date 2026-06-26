import jax.numpy as jnp


class Constants:
    # leaky-integrate-fire model parameters
    spiking_threshold: jnp.float32 = -50.0  # mV
    reset_voltage: jnp.float32 = -60.0  # mV
    exc_reversal_potential: jnp.float32 = 0.0  # mV
    inh_reversal_potential: jnp.float32 = -70.0  # mV
    leaky_reversal_potential: jnp.float32 = -70.0  # mV
    membrane_capacitance: jnp.float32 = 200.0  # pF

    # short-term plasticity parameters (Mongillo et al., 2008)
    u_total: jnp.float32 = 0.3
    x_max: jnp.float32 = 1.0
    tau_f: jnp.float32 = 1600  # msec
    tau_d: jnp.float32 = 50  # msec

    tau_refractory: float = 2.0  # msec
