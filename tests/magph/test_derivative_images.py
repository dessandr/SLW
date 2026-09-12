import numpy as np

from slw.magph.derivative_images import ExchangeDerivativeImages
from slw.exchange.kernels.spinor_epr_derivative import uniform_points


def test_coarse_preservation_and_dense_bond_reversal():
    mesh=(3,3,3)
    rp=np.indices(mesh).reshape(3,-1).T
    r=np.array([1,0,-1])
    rng=np.random.default_rng(903)
    c=rng.normal(size=(2,2,27,3))
    shifted=np.ravel_multi_index(tuple(((rp-r)%mesh).T),mesh)
    c[:,1,shifted]=c[:,0]
    plan=ExchangeDerivativeImages(qmesh=mesh,lattice_columns=np.eye(3),
        atom_positions=[[0,0,0],[.5,.5,0]],bond_i_atom=np.array([0,1]),
        bond_j_atom=np.array([1,0]),cell_shift=np.array([r,-r]),target_atoms=np.array([0,1]))
    q=uniform_points(mesh)
    expected=np.einsum('qr,tbra->qtba',np.exp(2j*np.pi*q@rp.T),c)
    np.testing.assert_allclose(plan.evaluate(c,q),expected,atol=3.e-14)
    dense=rng.uniform(-.5,.5,size=(9,3))
    value=plan.evaluate(c,dense)
    np.testing.assert_allclose(value[:,:,0],np.exp(2j*np.pi*dense@r)[:,None,None]*value[:,:,1],atol=3.e-14)
    np.testing.assert_allclose(plan.evaluate(c,-dense),value.conj(),atol=3.e-14)
