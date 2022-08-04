import string
from qiskit import *
from qiskit import QuantumCircuit, Aer
from qiskit.quantum_info import Operator
from qiskit.circuit.library import XGate, ZGate, PhaseGate
import numpy as np
from scipy import linalg
from functools import reduce
import itertools
import qc_utils.idx
from qc_utils.gates import *
import time
import copy

class Decomposer():
    def __init__(self, circuit, csdstop=2, optimize=True, verbose=True):
        # circuit can either be a unitary matrix, a filename to load QASM frome, or a qiskit QuantumCircuit
        if type(circuit) == str:
            qc = QuantumCircuit.from_qasm_file(circuit)
            qc.remove_final_measurements()
            backend = Aer.get_backend('unitary_simulator')
            job = execute(qc, backend)
            result = job.result()
            self.unitary = np.array(result.get_unitary(qc))
        elif type(circuit) == np.ndarray:
            self.unitary = circuit
        elif type(circuit) == circuit.quantumcircuit.QuantumCircuit:
            print('qc')
            circuit.remove_final_measurements()
            backend = Aer.get_backend('unitary_simulator')
            job = execute(circuit, backend)
            result = job.result()
            self.unitary = np.array(result.get_unitary(circuit))
        else:
            self.unitary = circuit
        self.nbits = int(np.log2(self.unitary.shape[0]))
        self.csdstop = csdstop

        self.qreg = QuantumRegister(self.nbits)
        self.areg = AncillaRegister(1)
        self.circuit = QuantumCircuit(self.qreg, self.areg)

        self.multiplexes = []
        self.cu_gates = []

        self.total_multiplex = 0
        self.identity_cu_removed = 0
        self.total_cx = 0
        self.total_cnx = 0

        # whether to use all optimizations or just output raw
        self.optimize = optimize
        self.verbose = verbose

    def start_timer(self):
        self.start_t = time.perf_counter()
    
    def stop_timer(self):
        self.vprint(f'{time.perf_counter() - self.start_t : 0.2f}s')

    def vprint(self, *args, **kwargs):
        if self.verbose:
            print(*args, **kwargs)

    def isidentity(self, mat):
        if type(mat) is not np.ndarray:
            mat = mat.to_matrix()
        return np.all(np.isclose(mat, np.identity(mat.shape[0])))

    # Decompose a unitary into uniformly controlled single-qubit rotations using the Cosine-Sine Decomposition.
    # Returns a final sequence L D R L D R L D R ... L D R, where L and R are uniformly controlled gates on the last qubit and
    # D are uniformly controlled Ry gates on the second-to-last qubit.
    def CSDfactor(self):
        self.vprint('Running CSDfactor...', end='')
        self.start_timer()

        # make a block-diagonal matrix with t and b
        def diag(t,b):
            zero = np.zeros((t.shape[0], b.shape[1]))
            return np.block([
                [t, zero],
                [zero.transpose(), b]
            ])

        # recursive CSD decomposition, returning a list of matrices representing uniformly-controlled U gates
        def CSD(u):
            dim = u.shape[0]
            dim2 = int(dim//2)
            if dim <= self.csdstop:
                return [u]
            else:
                L,D,R = linalg.cossin(u, dim2, dim2)

                L1 = L[:dim2,:dim2]
                L2 = L[dim2:,dim2:]
                R1 = R[:dim2,:dim2]
                R2 = R[dim2:,dim2:]

                l1_f = CSD(L1)
                l2_f = CSD(L2)
                r1_f = CSD(R1)
                r2_f = CSD(R2)

                return [diag(l1, l2) for l1,l2 in zip(l1_f, l2_f)] + [D] + [diag(r1, r2) for r1,r2 in zip(r1_f, r2_f)]

        mats = CSD(self.unitary)

        self.stop_timer()
        self.vprint('Total multiplexed single-qubit gates:', len(mats))
        self.multiplexes = mats

    # given a matrix representing a uniformly-controlled (multiplexed) single-qubit gate,
    # extract the individual controlled gates. Return target qubit, list of full matrices,
    # list of control bit patterns, and list of individual single-qubit gates
    # TODO: is tgt deterministic based on position in multiplex list?
    def extract_unitaries(self):
        if len(self.multiplexes) == 0:
            self.CSDfactor()

        self.vprint('Running extract_unitaries...', end='')
        self.start_timer()
        self.cu_gates = []

        for mat in self.multiplexes:
            found = False
            
            # try all possible targets, pick the first one that works
            for tgt in range(self.nbits):
                if not found:
                    store_tgt = self.nbits-tgt-1
                    store_cbits = []
                    store_single_mats = []
                    store_full_mats = []
                    identity_cu_removed = 0
                    ones = np.ones(mat.shape[0], complex)
                    invalid = False
                    for cbits in itertools.product([0,1], repeat=self.nbits-1):
                        cbits = list(cbits)
                        cbits_before = cbits[:tgt]
                        cbit_after = cbits[tgt:]
                        idx0 = qc_utils.idx.idx_from_bits(cbits_before + [0] + cbit_after)
                        idx1 = qc_utils.idx.idx_from_bits(cbits_before + [1] + cbit_after)

                        single_mat = np.zeros((2,2), complex)
                        single_mat[0,0] = mat[idx0,idx0]
                        single_mat[0,1] = mat[idx0,idx1]
                        single_mat[1,0] = mat[idx1,idx0]
                        single_mat[1,1] = mat[idx1,idx1]

                        umat = np.diag(ones)
                        umat[idx0,idx0] = mat[idx0,idx0]
                        umat[idx0,idx1] = mat[idx0,idx1]
                        umat[idx1,idx0] = mat[idx1,idx0]
                        umat[idx1,idx1] = mat[idx1,idx1]

                        if not np.all(np.isclose(np.dot(np.transpose(np.conjugate(umat)), umat), np.identity(umat.shape[0]))):
                            # if not unitary, move on to next option
                            found = False
                            break
                        elif not np.all(np.isclose(single_mat, np.identity(2))):
                            store_cbits.append(list(reversed(cbits_before + [1] + cbit_after)))
                            store_single_mats.append(single_mat)
                            store_full_mats.append(umat)
                            found = True
                        else:
                            identity_cu_removed += 1
                            found = True
                    else:
                        break
            if found:
                self.identity_cu_removed += identity_cu_removed
                for (bits, smat, fmat) in zip(store_cbits, store_single_mats, store_full_mats):
                    self.cu_gates.append((store_tgt, bits, smat, fmat))
            else:
                print('WARNING: NO MATCHING MATRIX FOUND')
                print(mat)
        self.stop_timer()
            #print('No successful target bit. Is the matrix in the correct form? Returning failed matrix:')
            #return mat

    # convert a controlled unitary to a sequence of CnZ gates and single-qubit rotations
    # input: target bit index, control bit string, and single-qubit 2x2 matrix
    def convert_to_cx_optim(self, tgt, cbits, cnx, cnx1, ctrls, A, B, C, a):
        combined_x = np.logical_not(np.equal(cbits, prev_cbits))#np.logical_xor(np.equal(cbits, np.zeros(self.nbits)), np.equal(prev_cbits, np.zeros(self.nbits)))
        if j > 0:
            ctrls_prev = list(range(self.nbits))
            ctrls_prev = ctrls_prev[:prev_tgt]+ctrls_prev[prev_tgt+1:]
            if np.sum(combined_x) == 1 and self.nbits > 2:
                ctrls_prev.remove(list(combined_x).index(1))
                self.circuit.append(cnx1, ctrls_prev + [self.areg[0]])
                self.total_cnx += 1
            else:
                self.circuit.append(cnx, ctrls_prev + [self.areg[0]])
                self.total_cnx += 1

        for i,c in enumerate(cbits):
            if c != prev_cbits[i]:
                self.circuit.x(i)
        
        if j > 0 and np.sum(combined_x) == 1 and self.nbits > 2:
            pass
        else:
            self.circuit.append(cnx, ctrls + [self.areg[0]])
            self.total_cnx += 1

        if not self.isidentity(C):
            self.circuit.unitary(C, tgt, label='C')

        if self.isidentity(B):
            pass
        elif np.all(np.isclose(B, X)):
            self.circuit.x(tgt)
        else:
            self.circuit.cx(self.areg[0], tgt)
            self.circuit.unitary(B, tgt, label='B')
            self.circuit.cx(self.areg[0], tgt)
            self.total_cx += 2

        if not self.isidentity(A):
            self.circuit.unitary(A, tgt, label='A')

        if not np.isclose(a, 0) or np.isclose(a, 2*np.pi):
            self.circuit.p(a, self.areg[0])

        prev_tgt = tgt
        prev_cbits = cbits

    def convert_to_cx_inplace(self, tgt, cbits, cnx, ctrls, A, B, C, a):
        for i,c in enumerate(cbits):
            if i != tgt and c == 0:
                self.circuit.x(i)
        self.circuit.unitary(C, tgt, label='C')
        self.circuit.append(cnx, ctrls + [tgt])
        self.circuit.unitary(B, tgt, label='B')
        self.circuit.append(cnx, ctrls + [tgt])
        self.circuit.unitary(A, tgt, label='A')
        self.total_cx += 2
        if not np.isclose(a, 0):
            self.circuit.append(cnx, ctrls + [tgt])
            self.circuit.unitary(u3(0,-a,0).conjugate().transpose(), tgt, label='Ua^t')
            self.circuit.append(cnx, ctrls + [tgt])
            self.circuit.unitary(u3(0,-a,0), tgt, label='Ua')
            self.circuit.append(PhaseGate(2*a).control(self.nbits-1), ctrls + [tgt])
            self.total_cx += 2
            #self.total_cp += 1
        for i,c in enumerate(cbits):
            if i != tgt and c == 0:
                self.circuit.x(i)

    def convert_to_cx_inplace_optim(self, tgt, cbits, cnx, ctrls, A, B, C, a):
        for i,c in enumerate(cbits):
            if i != tgt and c == 0:
                self.circuit.x(i)
        self.circuit.unitary(C, tgt, label='C')
        self.circuit.append(cnx, ctrls + [tgt])
        self.circuit.unitary(B, tgt, label='B')
        self.circuit.append(cnx, ctrls + [tgt])
        self.circuit.unitary(A, tgt, label='A')
        self.total_cx += 2
        if not np.isclose(a, 0):
            self.circuit.append(cnx, ctrls + [tgt])
            self.circuit.unitary(u3(0,-a,0).conjugate().transpose(), tgt, label='Ua^t')
            self.circuit.append(cnx, ctrls + [tgt])
            self.circuit.unitary(u3(0,-a,0), tgt, label='Ua')
            self.circuit.append(PhaseGate(2*a).control(self.nbits-1), ctrls + [tgt])
            self.total_cx += 2
            #self.total_cp += 1
        for i,c in enumerate(cbits):
            if i != tgt and c == 0:
                self.circuit.x(i)

    # create a circuit of controlled U operations
    def reconstruct_with_CU(self):
        if len(self.cu_gates) == 0:
            self.extract_unitaries()
        self.vprint('running reconstruct_with_CU...', end='')
        self.start_timer()
        self.circuit = QuantumCircuit(self.nbits)
        for tgt,cbits,mat,_ in reversed(self.cu_gates):
            cbit_idxs = list(range(self.nbits))
            cbit_idxs.remove(tgt)
            for i,b in enumerate(cbits):
                if i != tgt and b == 0:
                    self.circuit.x(i)
            gate = Operator(mat).to_instruction().control(self.nbits-1)
            self.circuit.append(gate, cbit_idxs + [tgt])
            for i,b in enumerate(cbits):
                if i != tgt and b == 0:
                    self.circuit.x(i)
        self.stop_timer()

    # create a circuit of controlled X operations
    def reconstruct_with_CX(self):
        if len(self.cu_gates) == 0:
            self.extract_unitaries()
        self.vprint('running reconstruct_with_CX...', end='')
        self.start_timer()
        self.circuit = QuantumCircuit(self.qreg, self.areg)
        cnx = XGate().control(self.nbits-1)
        cnx1 = None
        if self.nbits > 2:
            cnx1 = XGate().control(self.nbits-2) # currently does not work for nbits=2

        prev_tgt = 0
        prev_cbits = np.ones(self.nbits)
        for j,(tgt,cbits,mat,_) in enumerate(reversed(self.cu_gates)):
            a,A,B,C = qc_utils.gates.ABC(mat)
            if self.isidentity(B):
                print('NO')
            ctrls = list(range(self.nbits))
            ctrls = ctrls[:tgt]+ctrls[tgt+1:]
            if self.optimize:
                self.convert_to_cx_optim(tgt, cbits, cnx, cnx1, ctrls, A, B, C, a)
            else:
                self.convert_to_cx_inplace(tgt, cbits, cnx, ctrls, A, B, C, a)
                continue
                for i,c in enumerate(cbits):
                    if i != tgt and c == 0:
                        self.circuit.x(i)
                self.circuit.append(cnx, ctrls + [self.areg[0]])
                self.circuit.unitary(C, tgt, label='C')
                self.circuit.cx(self.areg[0], tgt)
                self.circuit.unitary(B, tgt, label='B')
                self.circuit.cx(self.areg[0], tgt)
                self.circuit.unitary(A, tgt, label='A')
                self.circuit.p(a, self.areg[0])
                self.circuit.append(cnx, ctrls + [self.areg[0]])
                self.total_cx += 2
                self.total_cnx += 2
                for i,c in enumerate(cbits):
                    if i != tgt and c == 0:
                        self.circuit.x(i)
        if self.optimize:
            ctrls_prev = list(range(self.nbits))
            ctrls_prev = ctrls_prev[:prev_tgt]+ctrls_prev[prev_tgt+1:]
            self.circuit.append(cnx, ctrls_prev + [self.areg[0]])
            self.total_cnx += 1
            for i,c in enumerate(prev_cbits):
                if i != tgt and c == 0:
                    self.circuit.x(i)

        self.stop_timer()
        self.vprint('cu identity gates removed by extract_unitaries:', self.identity_cu_removed)
        self.vprint('total cu gates after extract_unitaries:', len(self.cu_gates))
        self.vprint('total cx gates after convert_to_cx:', self.total_cx)
        self.vprint('total cnx gates:', self.total_cnx)