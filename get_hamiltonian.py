
from qiskit_nature.second_q.mappers import JordanWignerMapper
from qiskit_nature.second_q.hamiltonians import ElectronicEnergy

def get_hamiltonian(frag, nelecas_sub, ncas_sub, h1, h2):
    if frag is None:
        num_alpha = nelecas_sub[0]
        num_beta = nelecas_sub[1]
        n_so = ncas_sub*2
    else:
        # Get alpha and beta electrons from LAS
        num_alpha = nelecas_sub[frag][0]
        num_beta = nelecas_sub[frag][1]
        n_so = ncas_sub[frag]*2
        h1 = h1[frag]
        h2 = h2[frag]
    electronic_energy = ElectronicEnergy.from_raw_integrals(h1, h2)
    second_q_op = electronic_energy.second_q_op()
    mapper = JordanWignerMapper()
    hamiltonian = mapper.map(second_q_op)
    return hamiltonian

