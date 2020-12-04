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
"""
The `BrCrystal` class
---------------------

.. currentmodule:: brilleu.crystal

.. autosummary::
    :toctree: _generate

"""

import warnings
import numpy as np

from euphonic import Crystal as EuCrystal

import brille

class BrCrystal:
    """
    The :py:class:`BrCrystal` object holds spacegroup and symmetry functionality
    (intended for internal use only).

    Parameters
    ----------
    eucr : :py:obj:`euphonic.crystal.Crystal`
        Or any object with fields `cell_vectors`, `atom_r`, and `atom_type`
    irreducible : logical
        If True the BrillouinZone of this Crystal will be an irreducible
        Brillouin zone, otherwise it will be the first Brillouin zone.
    hall : int or str, optional
        A valid Hall symbol or its number as defined in, e.g.,
        :std:doc:`spglib:dataset`. Not all valid Hall symbols are assigned
        numbers, so the use of Hall numbers is discouraged.
    symmetry : :py:obj:`brille.Symmetry`, optional
        The symmetry operations (or their generators) for the spacegroup
        of the input crystal lattice

    Note
    ----
    The input symmetry information *must* correspond to the input lattice
    and no error checking is performed to confirm that this is the case.
    """
    def __init__(self, eucr, irreducible=True, hall=None, symmetry=None):
        if not hall and not isinstance(symmetry, brille.Symmetry):
            raise Exception("Either the Hall group or a brille.Symmetry object must be provided")
        if not isinstance(eucr, EuCrystal):
            print("Unexpected data type {}, expect failures.".format(type(eucr)))
        if symmetry and not isinstance(symmetry, brille.Symmetry):
            print('Unexpected data type {}, expect failures.'.format(type(symmetry)))
        self.irreducible = irreducible
        self.basis = eucr.cell_vectors.to('angstrom').magnitude
        self.atom_positions = eucr.atom_r
        _, self.atom_index = np.unique(eucr.atom_type, return_inverse=True)
        self.hall= hall
        self.symmetry = symmetry

    def get_basis(self):
        """Return the basis as row-vectors of a matrix

        Returns
        -------
        :py:class:`numpy.ndarray`
            The basis vectors as the rows of a :math:`3 \\times 3` matrix
        """
        return self.basis

    def get_inverse_basis(self):
        """Return the inverse of the basis matrix

        Returns
        -------
        :py:class:`numpy.ndarray`
            A :math:`3 \\times 3` matrix
        """
        return np.linalg.inv(self.get_basis())

    def get_atom_positions(self):
        """Return the atom positions in units of the basis vectors

        Returns
        -------
        :py:class:`numpy.ndarray`
            A :math:`N_{\\text{atom}} \\times 3` array of positions
        """
        return self.atom_positions

    def get_atom_index(self):
        """Return the unique atom index for each atom in the basis

        Returns
        -------
        :py:class:`numpy.ndarray`
            A :math:`N_{\\text{atom}} \\times 3` array of unsigned integers
            with each :math:`i < N_{\\text{atom}}`
        """
        return self.atom_index

    def get_cell(self):
        return (self.basis, self.atom_positions, self.atom_index)

    def get_Direct(self):
        """Return a Direct lattice object including the symmetry information

        Returns
        -------
        :py:class:`brille.Direct`
        """
        if self.hall:
            d = brille.Direct(*self.get_cell(), self.hall)
        else:
            d = brille.Direct(*self.get_cell(), 'P1')
        if self.symmetry:
            d.spacegroup = self.symmetry
        return d

    def use_irreducible(self):
        """Return a logical boolean value based on the stored irreducible property"""
        return True if self.irreducible else False

    def get_BrillouinZone(self):
        """Determine and return the irreducible Brillouin zone

        Returns
        -------
        :py:class:`brille.BrillouinZone`
        """
        return brille.BrillouinZone(self.get_Direct().star, wedge_search=self.use_irreducible())

    def orthogonal_to_basis_eigenvectors(self, vecs):
        """
        Convert a set of eigenvector from orthogonal to basis coordinates

        Some quantities are expressed in an orthogonal coordinate system, e.g.
        eigenvectors, but brille needs them expressed in the units of the
        basis vectors of the lattice which it uses.

        We need to convert the eigenvector components from units of
        :math:`(x,y,z)` to :math:`(a,b,c)` via the inverse of the basis vectors.
        For column vectors :math:`\\mathbf{x}` and :math:`\\mathbf{a}` and
        basis matrix :math:`A` `≡ self.get_basis()`

        .. math::
            \\mathbf{x}^T = \\mathbf{a}^T A

        which can be inverted to find :math:`\\mathbf{a}` from :math:`\\mathbf{x}`

        .. math::
            \\mathbf{a} = \\left(A^{-1}\\right)^T \\mathbf{x}

        Parameters
        ----------
        vecs : :py:class:`numpy.ndarray`
            :math:`N_{\\text{pt}} \\times N_{\\text{br}} \\times N_{\\text{atom}} \\times 3`
            eigenvectors in the orthogonal coordinate system

        Returns
        -------
        :py:class:`numpy.ndarray`
            eigenvectors expressed in the basis coordinate system
        """
        return np.einsum('ba,ijkb->ijka', self.get_inverse_basis(), vecs)

    def basis_to_orthogonal_eigenvectors(self, vecs):
        return np.einsum('ba,ijkb->ijka', self.get_basis(), vecs)
