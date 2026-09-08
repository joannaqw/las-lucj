from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error                                                                                                                                         
from qiskit_aer.primitives import SamplerV2 as Sampler
from qiskit import QuantumCircuit, transpile
noise_model = NoiseModel()
cx_depolarizing_prob = 0.2 
circ = QuantumCircuit(2)
circ.h(0)
circ.cx(0, 1)
circ.measure_all()

# Transpile for simulator
simulator = AerSimulator()
transpiled= transpile(circ, simulator)

noise_model.add_all_qubit_quantum_error(
    depolarizing_error(cx_depolarizing_prob, 2), ["cx"]
)
 
noisy_estimator = Sampler(
    options=dict(backend_options=dict(noise_model=noise_model))
)
job = noisy_estimator.run([transpiled],shots=300)
result = job.result()[0]          # PubResult for the first circuit
meas = result.data.meas           # BitArray
counts = meas.get_counts()        # dict like {'00': 12, '11': 288}
print(counts)

