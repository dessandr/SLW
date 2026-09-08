"""Independent scalar/SOC/exchange oracle for finite-q model g_XC."""
import numpy as np
from slw.wtorque.model.native_spinor_frame import build_native_spinor_frame
from slw.wtorque.gauge.kq_map import build_kq_map
from slw.wtorque.torque.vertices import PAULI


def test_model_gxc_excludes_scalar_and_soc_in_arbitrary_wannier_gauge():
    k = np.array([[0,0,0],[1/3,0,0],[2/3,0,0]],float)
    minus = np.array([0,2,1])
    q = np.array([1/3,0,0])
    iq, imq = build_kq_map(k,q).indices, build_kq_map(k,-q).indices
    rng = np.random.default_rng(441)
    u = np.array([np.linalg.qr(rng.normal(size=(4,4))+1j*rng.normal(size=(4,4)))[0] for _ in k])
    dagger = lambda x:x.conj().swapaxes(-1,-2)
    overlap = dagger(u)
    h = np.kron(np.diag([.8,-.8]),PAULI[2]).astype(complex)
    frame = build_native_spinor_frame(dagger(u)@h@u,overlap,minus,
        kpoints=k,orbital_centers=np.zeros((2,3)),orbital_masks=np.eye(2,dtype=bool),
        magnetic_site_positions=np.zeros((2,3)),local_frames=np.array([np.eye(3),np.diag([1,-1,-1])]),atomic_spin_order='interleaved')
    exchange = np.kron(np.diag([.4,-.3]),.7*PAULI[0]+.2*PAULI[2])
    scalar = np.kron(np.array([[1,.2],[.2,.7]]),np.eye(2))
    # Imaginary orbital L_y times spin sigma_z is TR-even SOC.
    soc = .17*np.kron(PAULI[1],PAULI[2])
    physical = exchange+scalar+soc
    g = (dagger(u[iq])@physical@u)[:,None]
    gm = (dagger(u[imq])@physical@u)[:,None]
    actual = frame.model_exchange_derivative_wannier(g,gm,q_red=q)
    expected = (dagger(u[iq])@exchange@u)[:,None]
    np.testing.assert_allclose(actual,expected,atol=2.e-14)
    gm_xc = frame.model_exchange_derivative_wannier(gm,g,q_red=-q)
    np.testing.assert_allclose(actual,dagger(gm_xc[iq]),atol=2.e-14)
    np.testing.assert_allclose(frame.model_exchange_derivative_wannier(actual,gm_xc,q_red=q),actual,atol=2.e-14)


def test_model_gxc_is_finite_difference_of_supercell_tr_odd_hamiltonian():
    # An independent real-space 3-cell Hamiltonian and its spin-TR operation
    # supply the derivative oracle before Fourier transforming to k+q,k.
    nc, nw = 3, 2
    k = np.array([[i/nc,0,0] for i in range(nc)])
    minus = np.array([0,2,1]);q = np.array([1/3,0,0])
    frame = build_native_spinor_frame(np.tile(PAULI[2],(nc,1,1)),np.tile(np.eye(nw,dtype=complex),(nc,1,1)),minus,
        kpoints=k,orbital_centers=np.zeros((1,3)),orbital_masks=np.ones((1,1),bool),
        magnetic_site_positions=np.zeros((1,3)),local_frames=np.eye(3)[None],atomic_spin_order='interleaved')
    f = np.exp(2j*np.pi*np.arange(nc)[:,None]*np.arange(nc)[None,:]/nc)/np.sqrt(nc)
    transform = np.kron(f,np.eye(nw))
    base = np.kron(np.eye(nc),PAULI[2])
    perturb = np.kron(np.diag(2*np.cos(2*np.pi*np.arange(nc)/nc)),.3*PAULI[0]+.7*np.eye(nw))
    theta = np.kron(np.eye(nc),np.array([[0,1],[-1,0]],complex))
    def xc(u):
        h=base+u*perturb
        return .5*(h-theta@h.conj()@theta.conj().T)
    delta=1.e-5
    numerical = transform.conj().T@((xc(delta)-xc(-delta))/(2*delta))@transform
    fullg = transform.conj().T@perturb@transform
    iq, imq = build_kq_map(k,q).indices,build_kq_map(k,-q).indices
    block = lambda a,i,j:a[2*i:2*i+2,2*j:2*j+2]
    g=np.array([block(fullg,j,i) for i,j in enumerate(iq)])[:,None]
    gm=np.array([block(fullg,j,i) for i,j in enumerate(imq)])[:,None]
    expected=np.array([block(numerical,j,i) for i,j in enumerate(iq)])[:,None]
    np.testing.assert_allclose(frame.model_exchange_derivative_wannier(g,gm,q_red=q),expected,atol=1.e-11)
