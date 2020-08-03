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
Additional useful utilities
---------------------------

.. currentmodule:: brilleu.utilities

.. autosummary::
    :toctree: _generate
"""
import os
import sys
import tempfile
import requests
import numpy as np
from scipy import special
from scipy.stats import norm, cauchy
from pathlib import Path

def fetchObjType(objtype, material, **kwds):
    """Fetch remote `CASTEP` binary file from the :py:mod:`brilleu` repository

    Parameters
    ----------
    objtype : class
        The output class, must have a static `class.from_castep()` method
    material : str
        The basename of the `CASTEP` file, e.g, `'NaCl'` for `NaCl.castep_bin`
    **kwds :
        All keyword arguments are passed to the `class.from_castep()` constructor

    Returns
    -------
    class
        The requested `class` object constructed from the fetched file.
    """
    base_url = "https://raw.githubusercontent.com/brille/brilleu/master/brilleu"
    file_to_fetch = material + ".castep_bin"
    with tempfile.TemporaryDirectory() as tmp_dir:
        r = requests.get(base_url + "/" + file_to_fetch)
        if not r.ok:
            raise Exception("Fetching {} failed with reason '{}'".format(file_to_fetch, r.reason))
        out_path = str(Path(tmp_dir, file_to_fetch))
        open(out_path, 'wb').write(r.content)
        return objtype.from_castep(out_path, **kwds)

def getObjType(objtype, material, **kwds):
    """Load a `CASTEP` binary file

    If the file is not located in the `brilleu` module directory this function
    will attempt to obtain it from the `brilleu` repository over any available
    network connection.

    Parameters
    ----------
    objtype : class
        The output class, must have a static `class.from_castep()` method
    material : str
        The basename of the `CASTEP` file, e.g, `'NaCl'` for `NaCl.castep_bin`
    **kwds :
        All keyword arguments are passed to the `class.from_castep()` constructor

    Returns
    -------
    class
        The requested `class` object constructed from the fetched file.
    """
    docs_dir = os.path.dirname(os.path.abspath(__file__))
    try:
        return objtype.from_castep(str(Path(docs_dir, material+".castep_bin")), **kwds)
    except FileNotFoundError:
        print('{} not found in {}. Fetching remote content.'.format(material, docs_dir))
        return fetchObjType(objtype, material, **kwds)

def broaden_modes(energy, omega, s_i, res_par_tem):
    """Compute S(Q,E) for a number of dispersion relations and intensities.

    Given any number of dispersion relations, ω(Q), and the intensities of the
    modes which they represent, S(Q), plus energy-broadening information in
    the form of a function name plus parameters (if required), calculate S(Q,E)
    at the provided energy positions.


    Parameters
    ----------
    energy : :py:class:`numpy.ndarray`, :math:`N_{\\text{point}}`
        The observation energies at which to calculate the broadened intensities
    omega : :py:class:`numpy.ndarray`, :math:`N_{\\text{mode}}`
        The characteristic energies of the modes to be broadened
    s_i : :py:class:`numpy.ndarray`, :math:`N_{\\text{mode}}`
        The integrated intensities of the modes to be broadened
    res_par_tem : array-like
        The broadening function choice (str) and its paramters as detailed below

    Note
    ----
    Selecting the energy linewidth function:

    ========================== ================ ================ ================
    Linewidth function         `res_par_tem[0]` `res_par_tem[1]` `res_par_tem[2]`
    ========================== ================ ================ ================
    Simple Harmonic Oscillator `'s'`                 `fwhm`        `temperature`
    Gaussian                   `'g'`                 `fwhm`
    Lorentzian                 `'l'`                 `fwhm`
    Voigt                      `'v'`                `g_fwhm`         `l_fwhm`
    ========================== ================ ================ ================


    Returns
    -------
    :math:`N` :py:class:`numpy.ndarray`, :math:`N_{\\text{point}} \\times N_{\\text{mode}}`
        :math:`S(\\mathbf{Q},E)` for each mode at the observation energies
    """
    if res_par_tem[0] in ('s', 'sho', 'simpleharmonicoscillator'):
        s_q_e = sho(energy, omega, s_i, res_par_tem[1], res_par_tem[2])
    elif res_par_tem[0] in ('g', 'gauss', 'gaussian'):
        s_q_e = gaussian(energy, omega, s_i, res_par_tem[1])
    elif res_par_tem[0] in ('l', 'lor', 'lorentz', 'lorentzian'):
        s_q_e = lorentzian(energy, omega, s_i, res_par_tem[1])
    elif res_par_tem[0] in ('v', 'voi', 'voigt'):
        s_q_e = voigt(energy, omega, s_i, res_par_tem[1])
    elif res_par_tem[0] in ('d', 'del', 'delta'):
        s_q_e = delta(energy, omega, s_i)
    else:
        print("Unknown function {}".format(res_par_tem[0]))
        s_q_e = s_i
    return s_q_e

def delta(x_0, x_i, y_i):
    """
    Compute the δ-function.

    y₀ = yᵢ×δ(x₀-xᵢ)

    Parameters
    ----------
    x_0 : :py:class:`numpy.ndarray`, :math:`N_{\\text{point}}`
        The observations at which to calculate intensities
    x_i : :py:class:`numpy.ndarray`, :math:`N_{\\text{point}} \\times N_{\\text{mode}}`
        The independent values of the modes to be broadened
    y_i : :py:class:`numpy.ndarray`, :math:`N_{\\text{point}} \\times N_{\\text{mode}}`
        The dependent values of the modes to be broadened

    Returns
    -------
    :py:class:`numpy.ndarray`
        :math:`N_{\\text{point}} \\times N_{\\text{mode}}`
    """
    y_0 = np.zeros(y_i.shape, dtype=y_i.dtype)
    # y_0 = np.zeros_like(y_i)
    y_0[x_0 == x_i] = y_i[x_0 == x_i]
    return y_0

def gaussian(x_0, x_i, y_i, fwhm):
    """Compute the normal distribution

    Parameters
    ----------
    x_0 : :py:class:`numpy.ndarray`, :math:`N_{\\text{point}}`
        The observations at which to calculate intensities
    x_i : :py:class:`numpy.ndarray`, :math:`N_{\\text{point}} \\times N_{\\text{mode}}`
        The independent values of the modes to be broadened
    y_i : :py:class:`numpy.ndarray`, :math:`N_{\\text{point}} \\times N_{\\text{mode}}`
        The dependent values of the modes to be broadened
    fwhm : float
        The full-width-at-half-maximum of the broadened modes

    Returns
    -------
    :py:class:`numpy.ndarray`
        :math:`N_{\\text{point}} \\times N_{\\text{mode}}`
    """
    if not np.isscalar(fwhm):
        fwhm = fwhm[0]
    sigma = fwhm/np.sqrt(np.log(256))
    z_0 = (x_0-x_i)/sigma
    y_0 = norm.pdf(z_0) * y_i
    return y_0

def lorentzian(x_0, x_i, y_i, fwhm):
    """Compute the Cauchy distribution

    Parameters
    ----------
    x_0 : :py:class:`numpy.ndarray`, :math:`N_{\\text{point}}`
        The observations at which to calculate intensities
    x_i : :py:class:`numpy.ndarray`, :math:`N_{\\text{point}} \\times N_{\\text{mode}}`
        The independent values of the modes to be broadened
    y_i : :py:class:`numpy.ndarray`, :math:`N_{\\text{point}} \\times N_{\\text{mode}}`
        The dependent values of the modes to be broadened
    fwhm : float
        The full-width-at-half-maximum of the broadened modes

    Returns
    -------
    :py:class:`numpy.ndarray`
        :math:`N_{\\text{point}} \\times N_{\\text{mode}}`
    """
    if not np.isscalar(fwhm):
        fwhm = fwhm[0]
    gamma = fwhm/2
    z_0 = (x_0-x_i)/gamma
    y_0 = cauchy.pdf(z_0) * y_i
    return y_0

def voigt(x_0, x_i, y_i, params):
    """Compute the convolution of a normal and Cauchy distribution.

    The Voigt function is the exact convolution of a normal distribution (a
    Gaussian) with full-width-at-half-max gᶠʷʰᵐ and a Cauchy distribution
    (a Lorentzian) with full-with-at-half-max lᶠʷʰᵐ. Computing the Voigt
    function exactly is computationally expensive, but it can be approximated
    to (almost always nearly) machine precision quickly using the
    `Faddeeva distribution <http://ab-initio.mit.edu/wiki/index.php/Faddeeva_Package>`_.

    The `Voigt distribution <https://en.wikipedia.org/wiki/Voigt_profile>`_
    is the real part of the Faddeeva distribution,
    given an appropriate rescaling of the parameters.

    Parameters
    ----------
    x_0 : :py:class:`numpy.ndarray`, :math:`N_{\\text{point}}`
        The observations at which to calculate intensities
    x_i : :py:class:`numpy.ndarray`, :math:`N_{\\text{point}} \\times N_{\\text{mode}}`
        The independent values of the modes to be broadened
    y_i : :py:class:`numpy.ndarray`, :math:`N_{\\text{point}} \\times N_{\\text{mode}}`
        The dependent values of the modes to be broadened
    params : float, array-like
        The full-width-at-half-maximum of the broadened modes. If scalar it is
        the Gaussian width and the Lorentzian width is zero; otherwise the
        Guassian then Lorentzian widths.

    Returns
    -------
    :py:class:`numpy.ndarray`
        :math:`N_{\\text{point}} \\times N_{\\text{mode}}`
    """
    if np.isscalar(params):
        g_fwhm = params
        l_fwhm = 0
    else:
        g_fwhm = params[0]
        l_fwhm = params[1]
    if l_fwhm == 0:
        return gaussian(x_0, x_i, y_i, g_fwhm)
    if g_fwhm == 0:
        return lorentzian(x_0, x_i, y_i, l_fwhm)

    area = np.sqrt(np.log(2)/np.pi)
    gamma = g_fwhm/2
    real_z = np.sqrt(np.log(2))*(x_0-x_i)/gamma
    imag_z = np.sqrt(np.log(2))*np.abs(l_fwhm/g_fwhm)
    # pylint: disable=no-member
    y_0 = area*np.real(special.wofz(real_z + 1j*imag_z))/gamma
    return y_0

def sho(x_0, x_i, y_i, fwhm, t_k):
    """Compute the Simple-Harmonic-Oscillator distribution.

    Parameters
    ----------
    x_0 : :py:class:`numpy.ndarray`, :math:`N_{\\text{point}}`
        The observations at which to calculate intensities
    x_i : :py:class:`numpy.ndarray`, :math:`N_{\\text{point}} \\times N_{\\text{mode}}`
        The independent values of the modes to be broadened
    y_i : :py:class:`numpy.ndarray`, :math:`N_{\\text{point}} \\times N_{\\text{mode}}`
        The dependent values of the modes to be broadened
    fwhm : float
        The full-width-at-half-maximum of the broadened modes
    t_k : float
        The temperature at which to calculate the broadening

    Returns
    -------
    :py:class:`numpy.ndarray`
        :math:`N_{\\text{point}} \\times N_{\\text{mode}}`
    """
    # (partly) ensure that all inputs have the same shape:
    if np.isscalar(fwhm):
        fwhm = fwhm * np.ones(y_i.shape)
    if np.isscalar(t_k):
        t_k = t_k * np.ones(y_i.shape)
    if x_0.ndim < x_i.ndim or (x_0.shape[1] == 1 and x_i.shape[1] > 1):
        x_0 = np.repeat(x_0, x_i.shape[1], 1)
    # include the Bose factor if the temperature is non-zero
    bose = x_0 / (1-np.exp(-11.602*x_0/t_k))
    bose[t_k == 0] = 1.0
    # We need x₀² the same shape as xᵢ
    x_02 = x_0**2
    # and to ensure that only valid (finite) modes are included
    flag = (x_i != 0) * np.isfinite(x_i)
    # create an output array
    y_0 = np.zeros(y_i.shape)
    # flatten everything so that we can use logical indexing
    # keeping the original output shape
    outshape = y_0.shape
    bose = bose.flatten()
    fwhm = fwhm.flatten()
    y_0 = y_0.flatten()
    x_i = x_i.flatten()
    y_i = y_i.flatten()
    x_02 = x_02.flatten()
    flag = flag.flatten()
    # and actually calculate the distribution
    part1 = bose[flag]*(4/np.pi)*fwhm[flag]*x_i[flag]*y_i[flag]
    part2 = ((x_02[flag]-x_i[flag]**2)**2 + 4*fwhm[flag]**2*x_02[flag])
    # if the brille object is holding complex values (it is) then its returned
    # interpolated values are all complex too, even, e.g., energies which are
    # purely real with identically zero imaginary components.
    # The division of two purely-real complex numbers in Python will annoyingly
    # raise a warning about discarding the imaginary part. So preempt it here.
    if not np.isclose(np.sum(np.abs(np.imag(part1))+np.abs(np.imag(part2))), 0.):
        raise RuntimeError('Unexpected imaginary components.')
    y_0[flag] = np.real(part1)/np.real(part2)
    return y_0.reshape(outshape)

def half_cpu_count():
    import os
    count = os.cpu_count()
    if 'sched_get_affinity' in dir(os):
        count = len(os.sched_getaffinity(0))
    return count//2


def __r_z(theta):
    c_t = np.cos(theta)
    s_t = np.sin(theta)
    return np.array([[c_t, -s_t, 0], [s_t, c_t, 0], [0, 0, 1]])


def __r_x(theta):
    c_t = np.cos(theta)
    s_t = np.sin(theta)
    return np.array([[1, 0, 0], [0, c_t, -s_t], [0, s_t, c_t]])


def __r_y(theta):
    c_t = np.cos(theta)
    s_t = np.sin(theta)
    return np.array([[c_t, 0, s_t], [0, 1, 0], [-s_t, 0, c_t]])


def __spherical_r_theta_phi(vec):
    rho = np.sqrt(np.dot(vec, vec))
    if rho > 0:
        theta = np.arccos(vec[2]/rho)
        phi = np.arctan2(vec[1], vec[0])
    else:
        # If the length of the vector is zero, there's nothing we can do
        theta = 0
        phi = 0
    return (rho, theta, phi)


def local_xyz(vec):
    """Return a local coordinate system based on the vector provided."""
    e_z = vec / np.sqrt(np.dot(vec, vec))
    e_x, e_y = __local_xy(e_z)
    return (e_x, e_y, e_z)


def __local_xy(vec):
    # Using a spherical coordinate system has the undesirable
    # quality that the local orthonormal coordinate system is discontinuous
    # at the poles, which is along the z-axis normally.
    # Since any of the axes is likely a high-symmetry direction with a high
    # chance of hosting degenerate modes, rotate the z-axis away to
    # an arbitrary (but constant) direction before determining theta and phi
    r_zx = np.matmul(__r_z(np.pi/3), __r_x(np.pi/5))
    inv_r_zx = np.matmul(__r_x(-np.pi/5), __r_z(-np.pi/3))
    _, theta, phi = __spherical_r_theta_phi(np.matmul(r_zx, vec))
    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)
    sin_phi = np.sin(phi)
    cos_phi = np.cos(phi)
    e_x = np.array((-sin_phi, cos_phi, 0))
    e_y = np.array((-cos_theta*cos_phi, -cos_theta*sin_phi, sin_theta))
    return (np.matmul(inv_r_zx, e_x), np.matmul(inv_r_zx, e_y))


def __arbitrary_xy(q_pt, e_vecs, primary_atom):
    e_x, _ = __local_xy(q_pt)
    _, n_atoms, n_dims = e_vecs.shape
    if n_dims != 3:
        raise Exceptatom("Only three dimesnatomal vectors supported.")
    assert primary_atom < n_atoms
    e_0 = e_vecs[-1, primary_atom, :]
    v0_x = np.dot(e_x, e_0)
    if np.real(np.conj(v0_x)*v0_x) == 0:
        raise Exceptatom("Primary atom ϵ⋅x̂ is zero!")
    phase_angle = np.arctan2(np.imag(v0_x), np.real(v0_x))
    if np.real(v0_x)*np.cos(phase_angle)+np.imag(v0_x)*np.sin(phase_angle) < 0:
        phase_angle = np.pi - phase_angle
    phase = np.cos(phase_angle) - 1j*np.sin(phase_angle)
    e_vecs *= phase
    # The only thing we could check at this point is that ϵ₀⋅x̂ is purely real.
    # Which unfortunately does not imply anything about ϵ₀⋅ŷ, ϵ₁⋅x̂, or ϵ₁⋅ŷ.
    return e_vecs


def __find_degenerate(e_vals):
    degenerate = []
    for val in e_vals:
        equiv = np.isclose(e_vals, val)
        if equiv.sum() > 1:
            degenerate.append(np.flatnonzero(equiv))
    return degenerate


def __check_and_fix(q_pt, e_vals, e_vecs, primary):
    degenerate = __find_degenerate(e_vals)
    while degenerate:
        d_set = degenerate.pop()
        if d_set.size == 2:
            e_vecs[d_set, :] = __arbitrary_xy(q_pt, e_vecs[d_set, :], primary)
        # if there are more than two degenerate modes we are likely at the
        # zone centre or boundary, in either case we need more than one point
        # to try and figure things out, so skip over this for now
    return e_vecs


def __find_primary_atom(q_pts, e_vecs):
    n_pts, n_modes, n_atoms, n_dims = e_vecs.shape
    q_dot_v = q_pts.reshape(n_pts, 1, 1, n_dims) * e_vecs
    q_v_norm = np.sqrt(np.sum(q_dot_v*np.conj(q_dot_v), axis=3))
    atom_q_v = np.real(np.mean(q_v_norm.reshape(n_pts*n_modes, n_atoms), axis=0))
    # The atom with smallest <|q⋅ϵ|> is most likely to have q⟂ϵ for any
    # given grid point (maybe?)
    return np.argmin(atom_q_v)


def degenerate_check(q_pts, e_vals, e_vecs, primary=None):
    """Check for degenerate eigenvalues and standardise their eigenvectors."""
    n_pts, n_modes, n_atoms, n_dims = e_vecs.shape
    assert (n_pts, n_dims) == q_pts.shape
    assert (n_pts, n_modes) == e_vals.shape
    if n_atoms > 1 and primary is None:
        primary = __find_primary_atom(q_pts, e_vecs)
    else:
        primary = 0
    for i in range(n_pts):
        e_vecs[i, :] = __check_and_fix(q_pts[i, :],
                                       e_vals[i, :],
                                       e_vecs[i, :],
                                       primary)
    return e_vecs


def align_eigenvectors(q_pts, e_vals, e_vecs, primary=None):
    """Align all eigenvectors such that :math:`\\Im(\\epsilon_p \\cdot \\hat{x}) = 0` at each point.

    Pick a smothly-varying local coordinate system based on :math:`\\mathbf{q}`
    for each point provided and apply an arbitrary phase such that the primary
    atom eigenvector is purely real and positive along the local
    :math:`\\mathbf{x}` direction.
    If a primary atom is not specified, pick one automatically which has the
    smallest mean displacement along :math:`\\hat{p}`, and therefore is most
    likely to have a significant
    :math:`\\left|\\epsilon_p \\cdot \\hat{x}\\right|` for all
    :math:`\\mathbf{q}`.
    """
    n_pts, n_modes, n_atoms, n_dims = e_vecs.shape
    assert (n_pts, n_dims) == q_pts.shape
    assert (n_pts, n_modes) == e_vals.shape
    if n_atoms > 1 and primary is None:
        primary = __find_primary_atom(q_pts, e_vecs)
    else:
        primary = 0
    for i in range(n_pts):
        e_vecs[i, :] = __arbitrary_xy(q_pts[i, :], e_vecs[i, :], primary)
    return e_vecs
