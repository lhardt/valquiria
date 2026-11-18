"""Starter implementation for the SCC performance challenge. Copy this file and make it fast.

    python -m bench run lih-12q --impl examples/bench/my_evaluator.py:MyEvaluator
    python -m bench vqe h2o-16q --impl examples/bench/my_evaluator.py:MyEvaluator

It computes exactly what the reference does (full 2^n statevector, adjoint gradient), but with
the three hot kernels pulled out into methods you can replace one at a time: ``rotate``
(an excitation applied to a state), ``apply_h`` (H psi) and ``overlap`` (the inner products).
Every change must keep passing the correctness gate; see the challenge page for the rules.
"""
from __future__ import annotations

import numpy as np

from numba import njit

from qcore.circuit import Circuit
from qcore.excitations import excitation_generator
from qcore.problem import MolecularProblem
from scipy.sparse.linalg import LinearOperator
import importlib.util
import sys
from pathlib import Path

from qcore.pauli import PauliSum

from bench.interface import ReferenceEvaluator


@njit
def one_popcnt( x : int ) -> int:
    ans = 0
    while x > 0:
        if x & 1:
            ans = ans + 1
        x = x // 2
    return ans
# 
# @jit
# def arr_popcnt( arr : numpy.ndarray ) -> numpy.ndarray:
#     return np.array([one_popcnt(x) for x in arr])


_BYTE_POPCOUNT = np.array([bin(i).count("1") for i in range(256)], dtype=np.int64)

@njit
def popcount(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.uint64)
    flat = np.ascontiguousarray(a.reshape(-1))
    return _BYTE_POPCOUNT[flat.view(np.uint8)].reshape(-1, 8).sum(axis=1).reshape(a.shape)

@njit
def parity_sign(a: np.ndarray) -> np.ndarray:
    """(-1)^{popcount(a)} as float array."""
    return 1.0 - 2.0 * (popcount(a) & 1)


@njit
def generator_offload(psi : numpy.ndarray, 
                      self__x : numpy.uint64, 
                      self_x : int, 
                      self_zs : list[int], 
                      self_gs : list[float], 
                      idx  : numpy.ndarray):

    src = idx ^ self__x
    d = np.zeros(idx.size, dtype=np.complex128)
    for z, g in zip(self_zs, self_gs):
        # convert explicitly to array because numba is not happy about implict conversion
        ph = np.array([1j * g * (1j) ** (one_popcnt(self_x & z) % 4)] * (idx.size))
        d = d + (ph * parity_sign(src & np.uint64(z)) if z else ph)
    if np.all(np.abs(d.imag) < 1e-14):
        d = np.ascontiguousarray(d.real).astype(np.complex128)

    return d * (psi[idx ^ self__x] if self_x else psi)

@njit
def rotate_offload(self__x : numpy.uint64, 
                   self_x : int, 
                   self_zs, 
                   self_gs, 
                   idx : numpy.ndarray, 
                   a : numpy.float64,
                   psi : numpy.ndarray,
                   g_psi: numpy.ndarray):

    src = idx ^ self__x
    d = np.zeros(idx.size, dtype=np.complex128)
    for z, g in zip(self_zs, self_gs):
        ph = 1j * g * (1j) ** (one_popcnt(self_x & z) % 4)
        d += ph * parity_sign(src & np.uint64(z)) if z else ph
    if np.all(np.abs(d.imag) < 1e-14):
        d = np.ascontiguousarray(d.real)
    w = np.abs(d)
    binary = bool(np.all((np.abs(w) < 1e-12) | (np.abs(w - 1) < 1e-12)))
    w2 = (w > 0.5).astype(float) if binary else w

    #d, w, binary = self._diag(idx)
    if self_x == 0:  # diagonal generator: exp(a d) elementwise
        return np.exp(a * d) * psi
    if binary:  # w in {0, 1}
        if w.all():
            return np.cos(a) * psi + np.sin(a) * g_psi
        return psi + ((np.cos(a) - 1.0) * w) * psi + np.sin(a) * g_psi
    aw = a * w
    safe = np.where(w > 0, w, 1.0)
    return np.cos(aw) * psi + np.where(w > 0, np.sin(aw) / safe, a) * g_psi

def angle_offload(coeff, theta, param) -> float:
    return coeff * theta[param] if param is not None else coeff


class MyEvaluator:
    def __init__(self, case, comm=None):
        # Setup is timed separately from the evaluations: build tables, allocate buffers or copy
        # data to a GPU here. The case is defined in bench/cases.py (integrals h1, h2, e_core,
        # n_alpha, n_beta, hf_occupation, excitations).
        self.case, self.comm = case, comm


        p = MolecularProblem.from_integrals(case.h1, case.h2, case.e_core, case.n_alpha, case.n_beta, "jw", e_fci=np.nan)
        circ = Circuit(p.n_qubits, case.hf_index)
        for k, exc in enumerate(case.excitations):
            circ.add_generator(excitation_generator(p.mapping, exc), circ.new_parameter(f"t{k}"))
        self.circuit = circ
        if p.n_qubits <= 20:
            self.hamiltonian = p.hamiltonian_sparse
        else:
            ham = p.hamiltonian
            dim = 1 << p.n_qubits
            self.hamiltonian = LinearOperator((dim, dim), matvec=ham.apply, dtype=np.complex128)

        self.idx = self.circuit.idx
        # each excitation k is one block rotation exp(theta_k G_k) acting on pairs (c, c ^ x)
        self.ops = self.circuit.ops

    # ------------------------------------------------------------ the three hot kernels
    def generator(self, op, psi):
        """G psi: gather psi[c ^ x] and multiply by a parity sign (TODO: fuse, compile, offload)."""
        return generator_offload(psi, op._x, op.x, op.zs, op.gs, self.idx)


    def rotate(self, op, psi, angle, g_psi=None):
        """exp(angle * G) psi, an exact 2x2 rotation on each amplitude pair."""
        g_psi = self.generator(op, psi) if g_psi is None else g_psi

        #op.rotate(psi, angle, g_psi, self.idx)
        return  rotate_offload(op._x, op.x, op.zs, op.gs, self.idx, angle, psi, g_psi) 

    def apply_h(self, psi):
        """H psi (TODO: sector-restricted, matrix-free or GPU sparse)."""
        return self.hamiltonian @ psi

    def overlap(self, a, b):
        """<a|b> (TODO: fuse into the kernels that produce a and b; MPI allreduce if distributed)."""
        return np.vdot(a, b)

    # ------------------------------------------------------------ energy + adjoint gradient
    def energy_and_gradient(self, theta):
        theta = np.asarray(theta, float)
        phi = self.circuit.initial_vector()  # the Hartree-Fock determinant
        for op in self.ops:
            a = angle_offload(op.coeff, theta, op.param) # op.angle(theta)
            if a != 0.0:
                phi = self.rotate(op, phi, a)
        lam = self.apply_h(phi)
        energy = float(np.real(self.overlap(phi, lam)))
        grad = np.zeros(theta.size)
        for op in reversed(self.ops):  # walk the circuit backwards, undoing one rotation at a time
            a = op.angle(theta)
            g_phi = self.generator(op, phi)
            grad[op.param] += 2.0 * op.coeff * np.real(self.overlap(lam, g_phi))
            if a != 0.0:
                phi = self.rotate(op, phi, -a, g_phi)
                lam = self.rotate(op, lam, -a)
        return energy, grad
