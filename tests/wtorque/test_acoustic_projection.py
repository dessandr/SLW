"""Rigid-translation physics, optical invariance, and unstable-mode gates."""
import numpy as np
import pytest

from slw.wtorque.projection.acoustic import (
    translation_basis, gamma_optical_modes, project_translation_kernel,
)


def test_mass_weighted_translations_are_not_uniform_eigenvectors():
    mass = np.array([2., 7.])
    t = translation_basis(mass)
    np.testing.assert_allclose(t.conj().T@t, np.eye(3), atol=1e-15)
    displacement = t / np.repeat(np.sqrt(mass), 3)[:, None]
    np.testing.assert_allclose(displacement[:3], displacement[3:])
    optical = np.linalg.qr(t, mode='complete')[0][:, 3:]
    d = (optical*np.array([1., 2., 3.])) @ optical.conj().T
    values, vectors, actual, diag = gamma_optical_modes(d, mass)
    np.testing.assert_allclose(values, [0, 0, 0, 1, 2, 3], atol=1e-14)
    np.testing.assert_allclose(vectors[:, :3], t)
    np.testing.assert_allclose(actual, d, atol=1e-14)
    assert diag['raw_translation_relative'] < 1e-14
    phase = np.exp(1j*np.array([.17, 1.21]))
    z = np.repeat(phase, 3)
    gauged = z[:, None]*d*z[None, :].conj()
    ev, _, _, _ = gamma_optical_modes(gauged, mass, phase=phase)
    np.testing.assert_allclose(ev, values, atol=1e-14)


def test_optical_instability_and_large_translation_error_are_not_hidden():
    mass = [2., 7.];t=translation_basis(mass)
    u=np.linalg.qr(t,mode='complete')[0][:,3:]
    stable=(u*np.array([1.,2.,3.]))@u.conj().T
    with pytest.raises(ValueError,match='translation residual'):
        gamma_optical_modes(stable+.01*t@t.conj().T,mass)
    with pytest.raises(ValueError,match='nonpositive Gamma optical'):
        gamma_optical_modes((u*np.array([-1.,2.,3.]))@u.conj().T,mass)


def test_kernel_projection_preserves_optical_couplings_and_component_sum():
    rng=np.random.default_rng(78);mass=np.array([2.,7.])
    t=translation_basis(mass);u=np.linalg.qr(t,mode='complete')[0][:,3:]
    bubble=rng.normal(size=(2,2,2,3))+1j*rng.normal(size=(2,2,2,3))
    direct=-.9*bubble
    projected,diag=project_translation_kernel(bubble,mass)
    projected_d,_=project_translation_kernel(direct,mass)
    projected_t,_=project_translation_kernel(bubble+direct,mass)
    np.testing.assert_allclose(projected+projected_d,projected_t,atol=1e-14)
    np.testing.assert_allclose(projected.sum(axis=-2),0,atol=1e-14)
    old=(bubble.reshape(2,2,6)/np.repeat(np.sqrt(mass),3))@u
    new=(projected.reshape(2,2,6)/np.repeat(np.sqrt(mass),3))@u
    np.testing.assert_allclose(new,old,atol=1e-14)
    assert diag['raw_translation_relative']>.1
    assert diag['projected_translation_relative']<1e-14
