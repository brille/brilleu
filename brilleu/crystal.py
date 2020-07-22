# brilleu -- an interface between brille and Euphonic
# Copyright 2020 Greg Tucker
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import warnings
import numpy as np

from euphonic import Crystal as EuCrystal

import brille

class BrCrystal:
    def __init__(self, eucr, hall=None, symmetry=None):
        """
        Construct a :py:obj:`brilleu.BrCrystal` object to hold spacegroup and
        symmetry functionality (intended for internal use only).

        Parameters
        ----------
        eucr : :py:obj:`euphonic.Crystal`
            Or any object with fields `cell_vectors`, `atom_r`, and `atom_type`
        hall : int or str [optional]
            A valid Hall symbol or its number as defined in, e.g.,
            :py:module:`spglib`. Not all valid Hall symbols are assigned
            numbers, so the use of Hall numbers is discouraged.
        symmetry : :py:obj:`brille.Symmetry` [optional]
            The symmetry operations (or their generators) for the spacegroup
            of the input crystal lattice

        Note
        ----
        The input symmetry information *must* correspond to the input lattice
        and no error checking is performed to confirm that this is the case.
        """
        if not hall and not isinstance(symmetry, brille.Symmetry):
            raise Exception("Either the Hall group or a brille.Symmetry object must be provided")
        if not isinstance(eucr, EuCrystal):
            print("Unexpected data type {}, expect failures.".format(type(eucr)))
        if symmetry and not isinstance(symmetry, brille.Symmetry):
            print('Unexpected data type {}, expect failures.'.format(type(symmetry)))
        self.basis = eucr.cell_vectors.to('angstrom').magnitude
        self.atom_positions = eucr.atom_r
        _, self.atom_index = np.unique(eucr.atom_type, return_inverse=True)
        self.hall= hall
        self.symmetry = symmetry

    def get_basis(self):
        return self.basis

    def get_inverse_basis(self):
        return np.linalg.inv(self.get_basis())

    def get_atom_positions(self):
        return self.atom_positions

    def get_atom_index(self):
        return self.atom_index

    def get_cell(self):
        return (self.basis, self.atom_positions, self.atom_index)

    def get_Direct(self):
        if self.hall:
            d = brille.Direct(*self.get_cell(), self.hall)
        else:
            d = brille.Direct(*self.get_cell())
        if self.symmetry:
            d.spacegroup = self.symmetry
        return d

    def get_BrillouinZone(self):
        return brille.BrillouinZone(self.get_Direct().star)

    def orthogonal_to_basis_eigenvectors(self, vecs):
        # some quantities are expressed in an orthogonal coordinate system, e.g.
        # eigenvectors, but brille needs them expressed in the units of the
        # basis vectors of the lattice which it uses.
        #
        # we need to convert the eigenvector components from units of
        # (x,y,z) to (a,b,c) via the inverse of the basis vectors:
        # For column vectors ⃗x and ⃗a and A ≡ self._basis()
        #        ⃗xᵀ = ⃗aᵀ A
        # which can be inverted to find ⃗a from ⃗x
        #       ⃗a = (A⁻¹)ᵀ ⃗x
        # A⁻¹ is (3,3) and the eigenvectors are (n_pt, n_br, n_io, 3)
        # we want to perform the matrix product of A⁻¹ with vecs[i,j,k,:]
        # which can be most easily accomplished using numpy's einsum
        # which even lets us skip the explicit transpose of A⁻¹
        return np.einsum('ba,ijkb->ijka', self.get_inverse_basis(), vecs)

    def basis_to_orthogonal_eigenvectors(self, vecs):
        return np.einsum('ba,ijkb->ijka', self.get_basis(), vecs)
