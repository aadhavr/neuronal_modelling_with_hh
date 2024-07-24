import matplotlib.pyplot as plt
import numpy as np
import math

def alpha_n(v: float) -> float:
    """Rate constant alpha for potassium channels."""
    v = v * 1000  # Converts voltage to mV
    return 0.01 * (-v - 55) / (math.exp((-v - 55) / 10.0) - 1) * 1000  # Reactions converted from ms to seconds

def beta_n(v: float) -> float:
    """Rate constant beta for potassium channels."""
    v = v * 1000
    return 0.125 * math.exp((-v - 65) / 80.0) * 1000

def alpha_m(v: float) -> float:
    """Rate constant alpha for sodium channels activation gate."""
    v = v * 1000
    return 0.1 * (-v - 40) / (math.exp((-v - 40) / 10.0) - 1) * 1000

def beta_m(v: float) -> float:
    """Rate constant beta for sodium channels activation gate."""
    v = v * 1000
    return 4 * math.exp((-v - 65) / 18.0) * 1000

def alpha_h(v: float) -> float:
    """Rate constant alpha for sodium channels inactivation gate."""
    v = v * 1000
    return 0.07 * math.exp((-v - 65) / 20.0) * 1000

def beta_h(v: float) -> float:
    """Rate constant beta for sodium channels inactivation gate."""
    v = v * 1000
    return 1 / (math.exp((-v - 35) / 10.0) + 1) * 1000

def initialize_variables(v_init: float):
    """Initialize gating variables n, m, and h based on the initial membrane potential."""
    n = alpha_n(v_init) / (alpha_n(v_init) + beta_n(v_init))
    m = alpha_m(v_init) / (alpha_m(v_init) + beta_m(v_init))
    h = alpha_h(v_init) / (alpha_h(v_init) + beta_h(v_init))
    return n, m, h

def update_gating_variables(v: float, n: float, m: float, h: float, dt: float):
    """Update gating variables n, m, and h based on the current membrane potential and time step."""
    dn = (alpha_n(v) * (1 - n) - beta_n(v) * n) * dt
    n += dn
    dm = (alpha_m(v) * (1 - m) - beta_m(v) * m) * dt
    m += dm
    dh = (alpha_h(v) * (1 - h) - beta_h(v) * h) * dt
    h += dh
    return n, m, h

def compute_currents(v: float, n: float, m: float, h: float, Gbar_k: float, Gbar_na: float, Gleak: float, Ek: float, Ena: float, Eleak: float):
    """Compute ionic currents through potassium, sodium, and leak channels."""
    Gk = Gbar_k * n**4
    i_k = Gk * (v - Ek)
    Gna = Gbar_na * m**3 * h
    i_na = Gna * (v - Ena)
    i_leak = Gleak * (v - Eleak)
    return i_k, i_na, i_leak

def run_simulation(dt: float, Cm: float, v_init: float, current_magnitude: float, Gbar_k: float, Gbar_na: float, Gleak: float, Ek: float, Ena: float, Eleak: float):
    """Run the neuronal simulation and return the membrane potential over time."""
    n, m, h = initialize_variables(v_init)
    i_inj = np.concatenate((np.zeros([round(0.2/dt), 1]),
                            current_magnitude * np.ones([round(0.3/dt), 1]),
                            np.zeros([round(0.5/dt), 1])))
    v_out = np.zeros(np.size(i_inj))

    for t in range(np.size(v_out)):
        if t == 0:
            v_out[t] = v_init
        else:
            n, m, h = update_gating_variables(v_out[t-1], n, m, h, dt)
            i_k, i_na, i_leak = compute_currents(v_out[t-1], n, m, h, Gbar_k, Gbar_na, Gleak, Ek, Ena, Eleak)
            i_cap = i_inj[t] - i_leak - i_k - i_na
            dv = i_cap / Cm * dt
            v_out[t] = v_out[t-1] + dv

    return v_out, dt

def plot_results(v_out, dt):
    """Plot the results of the simulation."""
    t_vec = np.linspace(0, dt * np.size(v_out), np.size(v_out))
    plt.plot(t_vec, v_out)
    plt.xlabel('Time (s)')
    plt.ylabel('Membrane Potential (V)')
    plt.show()

def main():
    dt = 10E-6  # Time step in 10 microseconds
    Cm = 100E-12  # Membrane capacitance of 100 picofarads
    v_init = -70E-3  # Initial membrane potential of -70 microvolts
    current_magnitude = 200E-12  # Injected current step of 200 picoamps
    Gbar_k = 1E-6  # Max potassium conductance
    Gbar_na = 7E-6  # Max sodium conductance
    Gleak = 5E-9  # Conductance of 5ns
    Ek = -80E-3  # Reversal for HH potassium current
    Ena = 40E-3  # Reversal for HH sodium current
    Eleak = -70E-3  # Reverse potential of -70mV

    v_out, dt = run_simulation(dt, Cm, v_init, current_magnitude, Gbar_k, Gbar_na, Gleak, Ek, Ena, Eleak)
    plot_results(v_out, dt)

if __name__ == '__main__':
    main()
