# python file to test if the env meet the requirements

import qiskit
import qiskit_experiments
from qiskit_ibm_runtime.fake_provider import FakeValenciaV2

print('----- start testing env -----')
print(f"Qiskit Version: {qiskit.__version__}")
print(f"Experiments Version: {qiskit_experiments.__version__}")


backend = FakeValenciaV2()
print(f"Backend loaded: {backend.name} with {backend.num_qubits} qubits")