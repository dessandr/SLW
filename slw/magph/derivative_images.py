"""Two-endpoint image interpolation of periodic exchange derivatives.

Input coefficients use C-ordered Rp residues. Images are chosen relative to
both magnetic endpoints, sharing ties equally. Unlike a single-box DFT this
preserves the actual (i,j,R;Rp) <-> (j,i,-R;Rp-R) image relation off grid.
"""
from __future__ import annotations

import numpy as np

from slw.wtorque.io.epr_short_range import _candidate_images, _select_tied_images


class ExchangeDerivativeImages:
    def __init__(self, *, qmesh, lattice_columns, atom_positions,
                 bond_i_atom, bond_j_atom, cell_shift, target_atoms):
        mesh = np.asarray(qmesh)
        if mesh.shape != (3,) or not np.issubdtype(mesh.dtype, np.integer) or np.any(mesh < 1):
            raise ValueError("qmesh must contain three positive integers")
        self.mesh = tuple(int(x) for x in mesh)
        lattice = np.asarray(lattice_columns, float)
        tau = np.asarray(atom_positions, float)
        ii, jj, rr = np.asarray(bond_i_atom), np.asarray(bond_j_atom), np.asarray(cell_shift)
        targets = np.asarray(target_atoms)
        if lattice.shape != (3, 3) or tau.ndim != 2 or tau.shape[1] != 3:
            raise ValueError("invalid geometry shapes")
        if rr.shape != (len(ii), 3) or jj.shape != ii.shape or targets.ndim != 1:
            raise ValueError("invalid bond/target shapes")
        if not np.issubdtype(rr.dtype, np.integer) or any(not np.issubdtype(a.dtype, np.integer) for a in [ii,jj,targets]):
            raise ValueError("bond/target indices and cell shifts must be integral")
        if any(np.any(a < 0) or np.any(a >= len(tau)) for a in [ii,jj,targets]):
            raise ValueError("atom index out of range")
        candidates = _candidate_images(self.mesh, lattice, None)
        self.plans = {}
        for t, atom in enumerate(targets):
            for b, (i, j, r) in enumerate(zip(ii, jj, rr)):
                selected, counts = _select_tied_images(
                    r[None], lattice=lattice, atom_position=tau[atom],
                    bra_center=tau[i], ket_center=tau[j], candidates=candidates, eps=1.e-6)
                vectors = candidates.vec[selected]
                residues = np.ravel_multi_index(tuple((vectors % mesh).T), self.mesh)
                ndeg = np.bincount(residues, minlength=int(np.prod(mesh)))[residues]
                self.plans[t, b] = (vectors, residues, 1./ndeg)
        mate_lookup = {(int(i), int(j), tuple(r)): b for b, (i,j,r) in enumerate(zip(ii,jj,rr))}
        for (t,b), (vectors, _, weights) in self.plans.items():
            mate = mate_lookup.get((int(jj[b]), int(ii[b]), tuple(-rr[b])))
            if mate is None:
                raise ValueError("derivative image interpolation requires complete bond mates")
            mv, _, mw = self.plans[t,mate]
            partner = {tuple(v): w for v,w in zip(mv,mw)}
            if any(tuple(v-rr[b]) not in partner or abs(partner[tuple(v-rr[b])]-w)>1.e-12 for v,w in zip(vectors,weights)):
                raise ValueError("two-endpoint image set is not closed under bond reversal")
        self.shape = (len(targets), len(ii), int(np.prod(mesh)), 3)

    def evaluate(self, coefficients, qpoints):
        """Return dJ(q) with shape (nq,ntarget,nbond,3), input units retained."""
        c, q = np.asarray(coefficients), np.asarray(qpoints, float)
        if c.shape != self.shape or q.ndim != 2 or q.shape[1] != 3:
            raise ValueError("coefficient/q shape mismatch")
        if not np.isfinite(c).all() or not np.isfinite(q).all():
            raise ValueError("nonfinite derivative or qpoints")
        out = np.empty((len(q), self.shape[0], self.shape[1], 3), complex)
        for (t,b), (vectors, residues, weights) in self.plans.items():
            out[:,t,b] = (np.exp(2j*np.pi*q @ vectors.T)*weights) @ c[t,b,residues]
        return out
