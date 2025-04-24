import numpy as np
from qiskit import QuantumCircuit, QuantumRegister

def get_so_ci_vec(ci_vec, nsporbs,nelec):
    lookup = {}
    cnt = 0
    norbs = nsporbs//2
    for ii in range (2**norbs):
        if f"{ii:0{norbs}b}".count('1') == np.sum(nelec)//2:
            lookup[f"{ii:0{norbs}b}"] = cnt
            cnt +=1
    so_ci_vec = np.zeros(2**nsporbs)
    for kk in range (2**nsporbs):
        if f"{kk:0{nsporbs}b}"[norbs:].count('1')==nelec[0] and f"{kk:0{nsporbs}b}"[:norbs].count('1')==nelec[1]:
            so_ci_vec[kk] = ci_vec[lookup[f"{kk:0{nsporbs}b}"[norbs:]],lookup[f"{kk:0{nsporbs}b}"[:norbs]]]
    return so_ci_vec

def get_initial_state(ncas_sub, las):
    qr1 = QuantumRegister(np.sum(ncas_sub)*2, 'q1')
    new_circuit = QuantumCircuit(qr1)
    new_circuit.initialize(get_so_ci_vec(las.ci[0][0],2*ncas_sub[0],las.nelecas_sub[0]) , [0,1,4,5])
    new_circuit.initialize(get_so_ci_vec(las.ci[1][0],2*ncas_sub[1],las.nelecas_sub[1]) , [2,3,6,7])
    return new_circuit

