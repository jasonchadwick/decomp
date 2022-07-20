# decomp
Decomposing arbitrary unitary matrices into multi-controlled CX/CZ gates for neutral atom computers

Mostly based on recursive cosine-sine decomposition, which turns an arbitrary unitary into "uniformly-controlled" one-qubit unitaries. 

Currently, have not figured out a way to get a better asymptotic scaling than the CNOT method from https://arxiv.org/abs/quant-ph/0504100v1

Work so far:

- Cosine-sine decomposition turns an arbitrary unitary into a product of multiplexed multi-controlled single-qubit gates (`CSDfactor`)
  - These multiplexed single-qubit gates can each be separated into $2^{n-1}$ multi-controlled single-qubit unitaries (`extract_unitaries`)
    - Along the way, we remove identity gates
  - Each of these multi-controlled single-qubit unitaries can then be decomposed into 4 $C^{n-1}X$ gates and 1 $C^{n-1}P$ gate, plus s.q. (`reconstruct_with_CX`)
    - Z-Y decomposition on the single-qubit unitary (`utils.gates.ABC`), then reconstruct
    - If Z-Y decomp. gives $\alpha=0$, only need 2 $C^{n-1}X$ and 0 $C^{n-1}P$
    - TODO: turn phase gate into CXs
  - Alternatively, each multi-controlled unitary can be decomposed into 2 $C^{n-1}X$ gates and 2 $CX$ gates using an ancilla bit
