#!/usr/bin/env python3
"""Validate the phonon eigenvector rotation method for correctness"""
import numpy as np
from brilleu.brilleu import getBrillEuObj

def test_NaCl():
    # fetch and load the NaCl.castep_bin file from the brille repository
    # Do not sort the modes on neighbouring trellis vertices to make comparison with Euphonic easier
    breu = getBrillEuObj('NaCl', sort=False, max_volume=0.1, parallel=False)
    # pick a trellis Q vertex inside the irreducible polyhedron, away from
    # the boundaries to avoid degeneracies:
    pt_idx = 4
    q_ir = breu.grid.rlu[pt_idx:pt_idx+1] # shape = (1,3)
    # pull together the pointgroup operations
    ptgr = breu.grid.BrillouinZone.lattice.pointgroup
    # and apply each to q_ir to find q_nu = R^T_nu q_ir
    q_nu = np.einsum('xji,aj->xi', ptgr.W, q_ir)

    # use brille to find eigenenergies and eigenvectors at each q_nu
    # this *should* be only an application of the rotation, so
    #   all omega_nu are expected to be identical
    #   and all epsilon_nu will be permuted as Gamma(q|nu) dictates
    br_nu = breu.QpointPhononModes(q_nu, interpolate=True)
    # use Euphonic to diagonalise the dynamical matrix at each q_nu
    eu_nu = breu.QpointPhononModes(q_nu, interpolate=False)

    # verify that we did not pick a point with degeneracies:
    assert not np.any(np.isclose(np.diff(br_nu.ω, axis=1), 0.)), "The selected grid point should have no degenerate modes"
    # verify that all brille eigenvalues are identical for each q_nu
    assert np.allclose(np.diff(br_nu.ω, axis=0), 0.), "All eigenvalues should be the same"
    # and that these results match the Euphonic results
    assert np.allclose(br_nu.ω, eu_nu.ω), "All eigenvalues should match with and without symmetry"
    # Now that we're sure the permutation is the same, we can verify the eigenvectors
    # brille stores and returns eigenvectors expressed in units of the conventional direct lattice
    # while euphonic calculates, returns, and stores them in a cartesian coordinate system defined
    # relative to the lattice by its lat_vec matrix.
    br_nu_ε_xyz = breu.crystal.basis_to_orthogonal_eigenvectors(br_nu.ε)

    # The eigenvectors returned by brille and those from Euphonic should be the same up to an overall complex phase factor
    # the eigenvectors are normalised, so the phase is calculated directly
    antiphase = np.exp(-1J*np.angle(np.einsum('qmij,qmij->qm', np.conj(eu_nu.ε), br_nu_ε_xyz)))
    # and removed from the cartesian coordinate eigenvectors
    br_nu_ε_xyz_phased = np.einsum('ab,abij->abij', antiphase, br_nu_ε_xyz)
    # now all eigenvectors returned from brille should be the same as
    # calculated by Euphonic directly
    assert np.allclose(br_nu_ε_xyz_phased, eu_nu.ε), "All eigenvectors should match up to an arbitrary phase"
