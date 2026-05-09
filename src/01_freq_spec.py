import numpy as np
import matplotlib.pyplot as plt
import time

# Qiskit
import qiskit
from qiskit import QuantumCircuit, transpile, pulse
from qiskit.circuit import Measure, Delay, Parameter
from qiskit.quantum_info import Operator
from qiskit.providers import QubitProperties
from qiskit.transpiler import Target, InstructionProperties

# Qiskit dynamics
import qiskit_dynamics
from qiskit_dynamics import Solver, DynamicsBackend

# Qiskit experiments
import qiskit_experiments
from qiskit_experiments.framework import ParallelExperiment
from qiskit_experiments.library import QubitSpectroscopy, Rabi, T1, T2Ramsey
from qiskit_experiments.calibration_management import Calibrations

# Qiskit traditional backend
import qiskit_aer as aer
from qiskit_aer.primitives import Estimator
from qiskit_ibm_runtime.fake_provider import FakeValenciaV2, FakeArmonk

import sympy


print("---- 1. Construct a qubit (Solver) ----")
qubit_freq = 5.0e9      # 5 GHz qubit frequency
drive_strength = 20e6   # 20 MHz drive strength
dt = 0.222e-9           # sampling period (~0.222 ns, IBM standard)

Z = Operator.from_label('Z')
X = Operator.from_label('X')

# Drift Hamiltonian (qubit's own energy)
static_hamiltonian = -0.5 * 2 * np.pi * qubit_freq * Z

solver = Solver(
    static_hamiltonian=static_hamiltonian,
    hamiltonian_operators=[2 * np.pi * drive_strength * X],
    hamiltonian_channels=["d0"],
    channel_carrier_freqs={"d0": qubit_freq},
    dt=dt,
    # === KEY FIX 1: Rotating frame ===
    # Move into the rotating frame of the drift Hamiltonian.
    # This eliminates the fast oscillations at the qubit frequency,
    # avoiding the sympy ComplexInfinity issue and making
    # simulation 10-100x faster.
    rotating_frame=static_hamiltonian,
    # === KEY FIX 2: Rotating Wave Approximation ===
    # Drop counter-rotating terms above 2*qubit_freq.
    # Standard practice for qubit simulations.

)

print("---- 2. Initialize Dynamics backend and assign kwargs ----")
pulse_backend = DynamicsBackend(
    solver=solver,
    subsystem_dims=[2],
    solver_options={
        "method": "RK45",   # If you installed jax+diffrax, change to "jax_odeint" for extra speed
        "atol": 1e-8,
        "rtol": 1e-8,
    },
)

# --- Build the target (instruction set) ---
target = Target(num_qubits=1, dt=dt)
target.qubit_properties = [QubitProperties(frequency=qubit_freq)]

# Measurement schedule: triggers ADC acquisition for 400 samples
with pulse.build(name="measure") as meas_sched:
    pulse.acquire(400, pulse.AcquireChannel(0), pulse.MemorySlot(0))

target.add_instruction(
    Measure(),
    {(0,): InstructionProperties(duration=400 * dt, calibration=meas_sched)}
)
target.add_instruction(Delay(Parameter('t')), {(0,): InstructionProperties()})

pulse_backend._target = target
pulse_backend._dt = dt
print("---- 3. Frequency Sweep ----")
freq_range = 3e7
frequencies = np.linspace(qubit_freq - freq_range/2,
                          qubit_freq + freq_range/2,
                          30)

start_time = time.perf_counter()
spec_exp = QubitSpectroscopy(physical_qubits=[0], frequencies=frequencies)

# Override defaults to avoid the GaussianSquare width=0 / sympy ComplexInfinity bug.
# Use a long pulse with a substantial flat-top width so the normalization
# denominator stays well away from 1 - exp(0) = 0.
spec_exp.set_experiment_options(
    amp=0.05,        # weak drive (avoid saturation)
    duration=2400*dt,   # samples (about 533 ns at dt=0.222 ns)
    sigma=256*dt,       # sample units, default
    width=1600*dt,      # ← KEY: non-zero flat top, avoids the bug
)

exp_data = spec_exp.run(backend=pulse_backend).block_for_results()
end_time = time.perf_counter()
print("---- 4. Finished ----")
if len(exp_data.data()) > 0:
    fig_data = exp_data.figure(0)
    # In qiskit-experiments 0.7+, figure() returns a FigureData wrapper
    fig = fig_data.figure if hasattr(fig_data, 'figure') else fig_data
    fig.savefig("spectroscopy_result.png", dpi=150, bbox_inches='tight')
    plt.show()

    fit_results = exp_data.analysis_results(0)
    freq = fit_results.value.params['freq']
    print(f"Resonance Frequency: {freq / 1e9:.5f} GHz")
    print(f"Time used: {(end_time - start_time):.2f} s")
else:
    print("no data")