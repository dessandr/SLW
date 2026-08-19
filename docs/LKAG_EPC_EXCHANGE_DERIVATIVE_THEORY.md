---
title: LKAG에서 exchange interaction과 electron-phonon derivative를 구하는 방법
aliases:
  - LKAG exchange derivative
  - EPC-LKAG formalism
tags:
  - LKAG
  - electron-phonon-coupling
  - exchange-interaction
  - spin-phonon-coupling
  - DMI
  - SLW
date: 2026-08-13
---

# LKAG에서 exchange interaction과 electron-phonon derivative를 구하는 방법

> [!summary] 핵심 요약
> LKAG는 작은 spin rotation에 대한 전자 band energy의 이차 변화를 Green's-function loop로 표현하여 exchange interaction을 구한다. 원자 변위가 들어오면 electron-phonon coupling(EPC)
> $$
> g^{\sigma,\kappa\mu}=\frac{\partial H^\sigma}{\partial u_{\kappa\mu}}
> $$
> 이 Green's function을
> $$
> \partial_u G=GgG
> $$
> 로 변화시키고, 동시에 local exchange splitting도
> $$
> \partial_u\Delta_i=\partial_uH_i^\uparrow-\partial_uH_i^\downarrow
> $$
> 로 변화시킨다. 따라서 $dJ/du$는 두 개의 local exchange vertex와 두 개의 propagator 중 하나를 차례로 미분한 **네 항의 합**이다. SOC를 포함한 spinor 문제에서는 같은 논리를 Pauli-resolved $A^{uv}$ kernel에 적용한 뒤 isotropic exchange, symmetric anisotropy, DMI로 분해한다.

## 1. 전체 논리

```text
spin-resolved or spinor H(k)
        │
        ├── local spin rotation ──> exchange vertices Δ_i or P_i
        │
        └── electronic propagation ──> Green function G(k,z)
                                      │
                                      └── LKAG contour integral ──> J_ij(R)

atomic displacement u_{κμ}(Rp)
        │
        └── EPC g = ∂H/∂u
                ├── ∂G = G g G
                └── ∂Δ or ∂P
                         │
                         └── differentiated LKAG loop ──> ∂J_ij(R)/∂u_{κμ}(Rp)
```

여기서 EPC는 electron linewidth를 계산하기 위해 등장하는 것이 아니다. 동일한 일차 전자 응답 $\partial H/\partial u$가 lattice displacement에 대한 magnetic force의 응답, 즉 exchange derivative를 만드는 perturbation vertex이기 때문에 필요하다.

## 2. Hamiltonian 및 bond convention

먼저 unit-vector Heisenberg model을

$$
\mathcal H_{\mathrm{spin}}
=-\sum_{ij\mathbf R}
J_{ij}(\mathbf R)\,
\mathbf e_{i\mathbf R}\cdot\mathbf e_{j\mathbf 0}
$$

로 둔다. $\mathbf e_i$는 local moment 방향의 단위벡터이다. 이 convention에서는 일반적으로

- $J>0$: ferromagnetic coupling,
- $J<0$: antiferromagnetic coupling

이다.

SLW의 real-space 출력은 $(i,j,\mathbf R)$와 $(j,i,-\mathbf R)$를 모두 가질 수 있는 **directed-bond list**를 사용한다. 따라서 외부 spin model에 옮길 때 $1/2$를 추가하거나 제거하기 전에, 그 코드가 directed pair를 모두 합하는지 canonical pair만 합하는지 확인해야 한다.

전자 LKAG가 직접 주는 것은 unit-vector model의 curvature이다. 길이 $S$인 spin operator model

$$
\mathcal H=-\sum J_{ij}^{(S)}\mathbf S_i\cdot\mathbf S_j
$$

에 넣으려면 사용하는 spin normalization에 따라 보통

$$
J_{ij}^{(S)}=\frac{J_{ij}^{(\mathbf e)}}{S^2}
$$

의 변환이 필요하다. SLW의 static tensor 경로에서는 `--spin_magnitude`가 이 scaling을 담당한다.

## 3. Collinear LKAG로 scalar Heisenberg $J$ 구하기

### 3.1 Spin-resolved Green's function

Collinear reference state에서

$$
G^\sigma(\mathbf k,z)
=\left[z+E_F-H^\sigma(\mathbf k)\right]^{-1},
\qquad \sigma\in\{\uparrow,\downarrow\}.
$$

코드가 $H-E_F$를 먼저 diagonalize한다면 같은 식을

$$
G^\sigma(\mathbf k,z)
=\left[z-(H^\sigma(\mathbf k)-E_F)\right]^{-1}
$$

로 쓸 수 있다. 두 표현의 $z$ 기준만 다를 뿐 물리는 같다.

Bond를 $(j,\mathbf 0)\leftrightarrow(i,\mathbf R)$로 정의하면 active EPR 경로의 real-space convention은

$$
G_{ij}^\sigma(\mathbf R,z)
=\frac{1}{N_k}\sum_{\mathbf k}
e^{-i2\pi\mathbf k\cdot\mathbf R}
\left[G^\sigma(\mathbf k,z)\right]_{ij}.
$$

아래첨자 $ij$는 단일 matrix element가 아니라 site $i$와 $j$에 배정된 local orbital subspace 사이의 block이다.

### 3.2 Local exchange splitting

Magnetic site $i$의 onsite spin splitting은

$$
\Delta_i
=H_{ii}^\uparrow(\mathbf R=0)-H_{ii}^\downarrow(\mathbf R=0)
$$

이다. Multi-orbital 문제에서는 $\Delta_i$를 scalar로 축약하지 않고 local orbital matrix 전체로 유지해야 한다.

### 3.3 Magnetic force theorem과 LKAG 식

Frozen-potential magnetic force theorem은 self-consistent total energy를 매번 다시 계산하는 대신, 고정된 reference potential에서 occupied one-electron spectrum의 변화를 사용한다. Lloyd-type expression을 schematic하게 쓰면

$$
\delta\Omega
=-\frac{1}{\pi}\operatorname{Im}
\int^{E_F}dE\,\operatorname{Tr}\ln(1-G\delta V).
$$

$\ln(1-X)=-X-X^2/2-\cdots$를 전개하면 서로 다른 두 site의 mixed second-order term은

$$
\delta^2\Omega_{ij}
\propto
\operatorname{Im}\int^{E_F}dE\,
\operatorname{Tr}
\left[\delta V_iG_{ij}\delta V_jG_{ji}\right]
$$

가 된다. Collinear local potential을

$$
H_i=H_i^0\sigma_0+\frac{\Delta_i}{2}\sigma_z
$$

로 쓰면 작은 transverse rotation $\delta\mathbf e_i$에 대한 일차 vertex는

$$
\delta V_i
=\frac{\Delta_i}{2}
(\delta e_i^x\sigma_x+\delta e_i^y\sigma_y).
$$

이를 mixed second-order term에 넣고 spin trace를 수행한 뒤, Heisenberg energy의 작은-angle 전개와 계수를 맞추면

$$
J_{ij}(\mathbf R)
=\frac{1}{4\pi}\operatorname{Im}
\int_{\mathcal C}dz\,
\operatorname{Tr}_{L}
\left[
\Delta_i
G_{ij}^\uparrow(\mathbf R,z)
\Delta_j
G_{ji}^\downarrow(-\mathbf R,z)
\right].
$$

$\operatorname{Tr}_L$은 local orbital index에 대한 trace이고, $\mathcal C$는 occupied electronic states를 포함하는 complex-energy contour이다. 이 식은 다음 closed scattering loop를 나타낸다.

```text
j@0 --G↑--> i@R --Δ_i--> i@R --G↓--> j@0 --Δ_j--> j@0
```

Trace의 cyclicity 때문에 실제 코드에서 matrix product의 시작점은 물리적 loop를 읽는 시작점과 달라도 된다.

> [!note] AFM reference의 SLW scalar sign normalization
> Global spin quantization axis로 AFM의 up/down Hamiltonian을 표현하면 site별 $\operatorname{Tr}\Delta_i$ 부호가 달라진다. SLW scalar 경로는
> $$
> s_i=\operatorname{sign}\operatorname{ReTr}\Delta_i,
> \qquad s_{ij}=s_is_j
> $$
> 를 정의하고 최종 결과를 $s_{ij}$로 나누어 local-moment frame의 부호로 정렬한다. 이는 project-specific output convention이며, 다른 LKAG 코드와 비교할 때 반드시 확인해야 한다.

### 3.4 수치 적분

실제로는 contour quadrature $(z_l,w_l)$을 사용하여

$$
J_{ij}(\mathbf R)
\simeq
\frac{1}{4\pi}\operatorname{Im}
\sum_l w_l\,
\operatorname{Tr}_{L}
\left[
\Delta_iG_{ij}^\uparrow
\Delta_jG_{ji}^\downarrow
\right]
$$

를 계산한다. SLW에는 semicircle contour와 pole-based integrator가 있으며, 최종 eV 값을 $1000$배 하여 meV로 저장한다.

## 4. EPC를 LKAG perturbation으로 넣기

### 4.1 Real-space EPC의 두 lattice vector

Atom $\kappa$의 Cartesian 방향 $\mu$ 변위에 대한 Hamiltonian derivative를

$$
g_{mn}^{\sigma,\kappa\mu}(\mathbf R_e,\mathbf R_p)
=
\frac{\partial H_{mn}^\sigma(\mathbf R_e)}
{\partial u_{\kappa\mu}(\mathbf R_p)}
$$

로 정의한다.

- $\mathbf R_e$: electronic hopping 또는 bra-ket separation을 나타내는 lattice vector
- $\mathbf R_p$: displaced atom image를 나타내는 lattice vector

두 vector는 서로 다른 자유도이다. 이를 하나의 $\mathbf R$로 합치면 $k\to k+q$ momentum transfer와 displacement-cell dependence를 동시에 잃는다.

Finite displacement로는

$$
g^{\sigma,\kappa\mu}
\simeq
\frac{H^\sigma(+\delta u_{\kappa\mu})
-H^\sigma(-\delta u_{\kappa\mu})}{2\delta u}
$$

로 얻는다. 따라서 단위는 orthogonal Wannier basis에서 eV/Å이다.

### 4.2 Double Fourier transform

SLW EPR phase convention에서는

$$
g^{\sigma,\kappa\mu}(\mathbf k,\mathbf q)
=\sum_{\mathbf R_e,\mathbf R_p}
e^{+i2\pi\mathbf k\cdot\mathbf R_e}
e^{+i2\pi\mathbf q\cdot\mathbf R_p}
g^{\sigma,\kappa\mu}(\mathbf R_e,\mathbf R_p).
$$

이는 ket $\mathbf k$를 bra $\mathbf k+\mathbf q$로 연결하는 perturbation matrix이다.

$$
g(\mathbf k,\mathbf q)
\equiv
\left\langle\mathbf k+\mathbf q\left|
\frac{\partial H}{\partial u_{\kappa\mu}(\mathbf q)}
\right|\mathbf k\right\rangle.
$$

이 정의에 대해 consistency relation은

$$
g(\mathbf k,\mathbf q)
=g^\dagger(\mathbf k+\mathbf q,-\mathbf q)
$$

이다.

### 4.3 Dyson identity: EPC가 $G$의 derivative가 되는 과정

Orthogonal basis에서

$$
G^{-1}=z+E_F-H
$$

이므로 inverse-matrix derivative

$$
dA^{-1}=-A^{-1}(dA)A^{-1}
$$

를 사용하면

$$
\frac{\partial G}{\partial u_{\kappa\mu}}
=G\frac{\partial H}{\partial u_{\kappa\mu}}G
=Gg^{\kappa\mu}G
$$

를 얻는다. Phonon momentum을 유지하면 정확한 off-diagonal momentum 구조는

$$
\boxed{
\delta G^\sigma(\mathbf k+\mathbf q,\mathbf k;z)
=G^\sigma(\mathbf k+\mathbf q,z)
g^{\sigma,\kappa\mu}(\mathbf k,\mathbf q)
G^\sigma(\mathbf k,z)
}
$$

이다. 즉 EPC가 Green's-function loop 안에 한 번 삽입된다.

### 4.4 Local exchange splitting의 derivative

두 spin channel의 local onsite EPC 차이는

$$
\delta\Delta_i^{\kappa\mu}
=\frac{\partial\Delta_i}{\partial u_{\kappa\mu}}
=g_{ii}^{\uparrow,\kappa\mu}(\mathbf R_e=0)
-g_{ii}^{\downarrow,\kappa\mu}(\mathbf R_e=0)
$$

이다. 이것은 hopping을 통한 propagation 변화와 별개로 local magnetic exchange field 자체가 displacement에 의해 변하는 효과이다.

## 5. Scalar LKAG의 analytic $dJ/du$

변위 좌표를 간단히 $\lambda\equiv u_{\kappa\mu}(\mathbf R_p)$로 쓰면 LKAG integrand의 네 factor $\Delta_i$, $G^\uparrow$, $\Delta_j$, $G^\downarrow$를 product rule로 미분하여

$$
\boxed{
\begin{aligned}
\frac{\partial J_{ij}(\mathbf R)}{\partial\lambda}
=\frac{1}{4\pi}\operatorname{Im}
\int_{\mathcal C}dz\,\operatorname{Tr}_L\Big[&
(\partial_\lambda\Delta_i)
G_{ij}^\uparrow
\Delta_jG_{ji}^\downarrow\\
&+\Delta_i
(\partial_\lambda G_{ij}^\uparrow)
\Delta_jG_{ji}^\downarrow\\
&+\Delta_iG_{ij}^\uparrow
(\partial_\lambda\Delta_j)
G_{ji}^\downarrow\\
&+\Delta_iG_{ij}^\uparrow\Delta_j
(\partial_\lambda G_{ji}^\downarrow)
\Big].
\end{aligned}
}
$$

각 항의 의미는 다음과 같다.

| 항 | 미분되는 object | 물리적 의미 |
|---|---|---|
| 1 | $\partial\Delta_i$ | endpoint $i$의 local exchange field 변화 |
| 2 | $\partial G_{ij}^\uparrow$ | forward electronic propagation의 EPC 변화 |
| 3 | $\partial\Delta_j$ | endpoint $j$의 local exchange field 변화 |
| 4 | $\partial G_{ji}^\downarrow$ | backward electronic propagation의 EPC 변화 |

따라서 $dJ$는 단순한 hopping derivative만도 아니고, 단순한 onsite exchange-splitting derivative만도 아니다. 두 종류의 response를 모두 포함한 LKAG loop 전체의 derivative이다.

SLW scalar AFM output convention에서는 static $J$와 마찬가지로 위 식 전체를 $s_{ij}=s_is_j$로 나눈다. 반면 tensor TB2J 경로는 local exchange-field 방향으로 $P_i$를 구성하므로 scalar `rel_sign` 보정을 별도로 적용하지 않는다.

### 5.1 Absolute cell과 relative displacement index

Pair가 $(j,\mathbf0)\leftrightarrow(i,\mathbf R)$이고 실제 displaced atom이 cell $\mathbf R_p^{\mathrm{abs}}$에 있다면 translational covariance에 의해 local response는 electronic site와 displaced atom 사이의 상대 vector에 의존한다. EPC 저장 convention을

$$
\boldsymbol\rho
=\mathbf R_{\mathrm{electronic}}-\mathbf R_{\mathrm{displaced}}
$$

로 잡으면

$$
\boldsymbol\rho_j=-\mathbf R_p^{\mathrm{abs}},
\qquad
\boldsymbol\rho_i=\mathbf R-\mathbf R_p^{\mathrm{abs}}.
$$

반대 convention을 쓰는 데이터에서는 두 부호가 모두 뒤집힌다. 그러므로 $\mathbf R_p\pm\mathbf R$ shift를 식의 외형만 보고 수동으로 넣으면 안 된다. SLW의 EPR $k,q$ 경로는 저장된 EPC phase convention을 그대로 double Fourier transform한 뒤 $q\to R_p$ 변환하므로 이 shift를 자동으로 처리한다.

### 5.2 $q$-space에서 계산한 뒤 $R_p$로 복원

각 $\mathbf q$에 대해

$$
\partial_uG(\mathbf k+\mathbf q,\mathbf k)
=G(\mathbf k+\mathbf q)g(\mathbf k,\mathbf q)G(\mathbf k)
$$

를 계산하고 $\mathbf k$를 합하여 bond $\mathbf R$의 derivative를 만든다. EPR transform이 $e^{+i\mathbf q\cdot\mathbf R_p}$를 사용하므로 inverse transform은

$$
\frac{\partial J_{ij}(\mathbf R)}
{\partial u_{\kappa\mu}(\mathbf R_p)}
=\frac{1}{N_q}\sum_{\mathbf q}
e^{-i2\pi\mathbf q\cdot\mathbf R_p}
\frac{\partial J_{ij}(\mathbf R;\mathbf q)}
{\partial u_{\kappa\mu}(\mathbf q)}.
$$

코드에서는 `np.fft.fftn(...)/Nq`가 이 negative-sign DFT를 수행한다.

### 5.3 Phonon normal-mode derivative

Dynamical-matrix eigenvector를 $\sum_{\kappa\mu}|e_{\kappa\mu}^{\nu}(\mathbf q)|^2=1$로 정규화하고

$$
\frac{\partial J_{ij}}{\partial u_{\kappa\mu}(\mathbf q)}
=\sum_{\mathbf R_p}e^{+i2\pi\mathbf q\cdot\mathbf R_p}
\frac{\partial J_{ij}}{\partial u_{\kappa\mu}(\mathbf R_p)}
$$

로 정의하면 mode coordinate에 대한 exchange derivative는 원자 Cartesian derivative의 projection이다.

$$
\frac{\partial J_{ij}}{\partial Q_{\mathbf q\nu}}
=\sum_{\kappa\mu}
\frac{e_{\kappa\mu}^{\nu}(\mathbf q)}{\sqrt{M_\kappa}}
\frac{\partial J_{ij}}{\partial u_{\kappa\mu}(\mathbf q)}.
$$

Quantized phonon vertex까지 만들 때는 추가로 zero-point amplitude가 들어간다.

$$
g_{ij,\nu}^{J}(\mathbf q)
=\sqrt{\frac{\hbar}{2\omega_{\mathbf q\nu}}}
\frac{\partial J_{ij}}{\partial Q_{\mathbf q\nu}}.
$$

Phonon eigenvector가 이미 mass-normalized인지, atomic gauge인지 cell gauge인지에 따라 $1/\sqrt{M_\kappa}$와 $e^{i\mathbf q\cdot\boldsymbol\tau_\kappa}$의 위치가 달라지므로 실제 projection에서는 phonon 파일 convention을 따라야 한다.

## 6. Non-orthogonal LCAO basis에서의 보정

Current EPR tensor path는 orthogonal Wannier representation을 사용한다. ABACUS LCAO처럼 basis가 non-orthogonal이면

$$
G(z)=\left[(z+E_F)S-H\right]^{-1}
$$

이고 derivative는

$$
\boxed{
\partial_uG
=G\left[\partial_uH-(z+E_F)\partial_uS\right]G
}
$$

이다. 따라서 LKAG에 삽입되는 energy-dependent effective perturbation은

$$
g_{\mathrm{res}}(z)
=\partial_uH-(z+E_F)\partial_uS
$$

이다. $\partial S/\partial u$를 버릴 수 있는 것은 Wannier orthogonalization 이후이거나 overlap response가 무시 가능하다고 별도로 검증한 경우뿐이다.

Band EPC 자체를 엄밀히 만들 때는 basis derivative의 left/right overlap 항도 구별해야 한다. 따라서 ABACUS finite-displacement CSR에서 바로 LKAG를 수행하는 경로와 orthogonal EPR/Wannier 경로의 EPC definition을 혼용해서는 안 된다.

## 7. Exchange tensor로 확장

### 7.1 Tensor spin Hamiltonian

SOC가 있으면 spin-rotation symmetry가 깨지므로 scalar $J$를 $3\times3$ tensor로 확장한다.

$$
\mathcal H_{\mathrm{spin}}
=-\sum_{ij\mathbf R}
e_{i\mathbf R}^{a}
J_{ij}^{ab}(\mathbf R)
e_{j\mathbf0}^{b}.
$$

Tensor는

$$
\mathbf J_{ij}
=J_{ij}^{\mathrm{iso}}\mathbf I
+\boldsymbol\Gamma_{ij}
+\mathbf J_{ij}^{\mathrm{DMI}}
$$

로 분해한다.

$$
J^{\mathrm{iso}}=\frac13\operatorname{Tr}\mathbf J,
\qquad
\boldsymbol\Gamma
=\frac12(\mathbf J+\mathbf J^T)-J^{\mathrm{iso}}\mathbf I.
$$

$\boldsymbol\Gamma$는 symmetric traceless tensor이다. slw/TB2J DMI matrix convention은

$$
\mathbf J^{\mathrm{DMI}}
=\begin{pmatrix}
0&D_z&-D_y\\
-D_z&0&D_x\\
D_y&-D_x&0
\end{pmatrix},
$$

따라서

$$
-\mathbf e_i^T\mathbf J^{\mathrm{DMI}}\mathbf e_j
=-\mathbf D_{ij}\cdot(\mathbf e_i\times\mathbf e_j).
$$

### 7.2 Spinor Hamiltonian과 local exchange projector

Local spinor block을 orbital-space Pauli component로 분해한다.

$$
H_i=H_i^0\sigma_0+M_i^x\sigma_x+M_i^y\sigma_y+M_i^z\sigma_z.
$$

Static local exchange-field 방향을 $\hat{\mathbf e}_i$라 하면 TB2J-style local projector는

$$
P_i=\mathbf M_i\cdot\hat{\mathbf e}_i.
$$

Collinear $z$ limit에서는

$$
M_i^z=\frac{H_i^\uparrow-H_i^\downarrow}{2}
=\frac{\Delta_i}{2},
\qquad P_i=\frac{\Delta_i}{2}.
$$

SOC를 포함한 spinor Green's function도 Pauli component로 쓴다.

$$
G_{ij}=G_{ij}^{0}\sigma_0+
G_{ij}^{x}\sigma_x+
G_{ij}^{y}\sigma_y+
G_{ij}^{z}\sigma_z.
$$

각 $G_{ij}^{u}$는 여전히 orbital matrix이다.

### 7.3 TB2J-aligned $A^{uv}$ kernel

SLW production tensor 경로의 기본 object는

$$
\boxed{
A_{ij}^{uv}(\mathbf R)
=\frac{1}{\pi}\int_{\mathcal C}dz\,
\operatorname{Tr}_{L}
\left[
P_iG_{ij}^{u}(\mathbf R,z)
P_jG_{ji}^{v}(-\mathbf R,z)
\right]
}
$$

이며 $u,v\in\{0,x,y,z\}$이다. Pair-complete data에서 $(i,j,\mathbf R)$와 $(j,i,-\mathbf R)$를 함께 이용하여 물리 tensor를 구성한다.

TB2J-aligned decomposition은

$$
J_{ij}^{\mathrm{iso}}
=\operatorname{Im}
\left(A^{00}-A^{xx}-A^{yy}-A^{zz}\right),
$$

$$
J_{ij,\mathrm{ani}}^{\alpha\beta}
=\operatorname{Im}
\left[
A_{ij}^{\alpha\beta}(\mathbf R)
+A_{ji}^{\alpha\beta}(-\mathbf R)
\right],
$$

$$
D_{ij}^{\alpha}
=\operatorname{Re}
\left(A_{ij}^{0\alpha}-A_{ij}^{\alpha0}\right).
$$

Raw $\mathbf J_{\mathrm{ani}}$는 일반적으로 traceless가 아니다. 따라서

$$
\mathbf J_s
=\frac12(\mathbf J_{\mathrm{ani}}+\mathbf J_{\mathrm{ani}}^T),
\qquad
\boldsymbol\Gamma
=\mathbf J_s-\frac{\operatorname{Tr}\mathbf J_s}{3}\mathbf I
$$

로 physical symmetric-traceless anisotropy를 만든다. 최종 production tensor는

$$
\mathbf J_{\mathrm{full}}
=J^{\mathrm{iso}}\mathbf I
+\boldsymbol\Gamma
+\mathbf J^{\mathrm{DMI}}
$$

이다. Raw antisymmetric $A^{\alpha\beta}$ 조합은 비교·debug용이며, DMI vector로 재구성한 antisymmetric tensor와 구분해야 한다.

### 7.4 왜 scalar 식과 tensor 식의 prefactor가 달라 보이는가

Scalar LKAG는 vertex로 $\Delta_i$를 쓰고 prefactor가 $1/(4\pi)$이다. TB2J $A^{uv}$는

$$
P_i=\frac{\Delta_i}{2},\qquad P_j=\frac{\Delta_j}{2}
$$

를 쓰므로 두 projector에서 $1/4$가 생긴다. 따라서

$$
\frac1\pi P_iP_j
=\frac1{4\pi}\Delta_i\Delta_j.
$$

즉 두 convention은 collinear limit에서 같은 normalization을 준다. $P=\Delta/2$를 사용하면서 다시 $1/(4\pi)$를 곱하거나, $\Delta$를 사용하면서 $1/\pi$를 곱하면 factor-of-four 오류가 발생한다.

## 8. Exchange tensor derivative $dJ^{ab}/du$

### 8.1 Direct torque-trace 관점

일반적인 spinor magnetic-force theorem에서는 local rotation에 대한 torque vertex를 $\mathcal T_i^a=\partial H/\partial\theta_i^a$로 정의할 수 있다. 그러면 schematic tensor kernel은

$$
J_{ij}^{ab}
\propto\operatorname{Im}\int dz\,
\operatorname{Tr}
\left[
\mathcal T_i^aG_{ij}
\mathcal T_j^bG_{ji}
\right].
$$

변위 derivative는 scalar 식과 동일하게

$$
\begin{aligned}
\partial_uJ_{ij}^{ab}\propto
\operatorname{Im}\int dz\,\operatorname{Tr}\Big[&
(\partial_u\mathcal T_i^a)G_{ij}\mathcal T_j^bG_{ji}
+\mathcal T_i^a(\partial_uG_{ij})\mathcal T_j^bG_{ji}\\
&+\mathcal T_i^aG_{ij}(\partial_u\mathcal T_j^b)G_{ji}
+\mathcal T_i^aG_{ij}\mathcal T_j^b(\partial_uG_{ji})
\Big]
\end{aligned}
$$

의 네 항을 갖는다. SLW에는 raw direct trace 경로도 있지만, 현재 static/dynamic tensor production convention은 아래 TB2J $A^{uv}$ decomposition이다.

### 8.2 Production TB2J $dA^{uv}$ kernel

$A^{uv}$를 직접 미분하면

$$
\boxed{
\begin{aligned}
\partial_u A_{ij}^{uv}
=\frac1\pi\int dz\,\operatorname{Tr}_L\Big[&
(\partial_uP_i)G_{ij}^uP_jG_{ji}^v\\
&+P_i(\partial_uG_{ij}^u)P_jG_{ji}^v\\
&+P_iG_{ij}^u(\partial_uP_j)G_{ji}^v\\
&+P_iG_{ij}^uP_j(\partial_uG_{ji}^v)
\Big].
\end{aligned}
}
$$

Static local direction $\hat{\mathbf e}_i$를 고정하면

$$
\partial_uP_i
=(\partial_u\mathbf M_i)\cdot\hat{\mathbf e}_i.
$$

Collinear limit에서는

$$
\partial_uP_i=\frac12\partial_u\Delta_i.
$$

그 다음 static decomposition을 선형적으로 미분하여

$$
\partial_uJ^{\mathrm{iso}}
=\operatorname{Im}
\left(\partial_uA^{00}-\partial_uA^{xx}
-\partial_uA^{yy}-\partial_uA^{zz}\right),
$$

$$
\partial_uJ_{\mathrm{ani}}^{\alpha\beta}
=\operatorname{Im}
\left[
\partial_uA_{ij}^{\alpha\beta}(\mathbf R)
+\partial_uA_{ji}^{\alpha\beta}(-\mathbf R)
\right],
$$

$$
\partial_uD^\alpha
=\operatorname{Re}
\left(\partial_uA^{0\alpha}-\partial_uA^{\alpha0}\right)
$$

를 얻는다. 최종 tensor derivative는

$$
\boxed{
\frac{\partial\mathbf J_{ij}}{\partial u}
=\frac{\partial J_{ij}^{\mathrm{iso}}}{\partial u}\mathbf I
+\frac{\partial\boldsymbol\Gamma_{ij}}{\partial u}
+\frac{\partial\mathbf J_{ij}^{\mathrm{DMI}}}{\partial u}
}
$$

이다.

### 8.3 SLW active tensor 경로의 onsite-projector 범위

`compute_dJ_epr_tensor`에서는 모든 displaced atom이 full EPC matrix를 통해 $\partial G$ 두 항에 기여한다. 반면 $\partial P_i$와 $\partial P_j$는 기본 설정에서 displaced target이 해당 magnetic bond endpoint와 일치할 때만 포함된다. 즉 현재 구현은

- nonlocal EPC effect: $\partial G$를 통해 포함,
- local magnetic exchange-field response: endpoint target에 대한 $\partial P$로 포함,
- displacement-induced local direction 변화 $\partial_u\hat{\mathbf e}_i$: 포함하지 않음

이라는 범위를 갖는다. `--no-onsite_deriv_projector`를 사용하면 frozen-projector approximation이 되어 $\partial G$ 두 항만 남는다.

## 9. 현재 SLW의 실제 계산 순서

### 9.1 Scalar $J$

1. $H^\uparrow(\mathbf k)$, $H^\downarrow(\mathbf k)$ 구성
2. Magnetic orbital slice별 $\Delta_i$ 구성
3. 각 contour energy에서 $G^\uparrow(\mathbf k,z)$, $G^\downarrow(\mathbf k,z)$ 계산
4. $\mathbf k\to\mathbf R$ Fourier transform
5. $\operatorname{Tr}[\Delta_iG^\uparrow_{ij}\Delta_jG^\downarrow_{ji}]$ 계산
6. Energy contour sum, imaginary part, $1/(4\pi)$ 및 AFM sign normalization 적용

### 9.2 Scalar $dJ/du$

1. EPR에서 $g^\uparrow(\mathbf k,\mathbf q)$와 $g^\downarrow(\mathbf k,\mathbf q)$ 구성
2. Commensurate mesh에서 정확한 $\mathbf k+\mathbf q$ index map 구성
3. $\partial G^\sigma=G^\sigma_{\mathbf k+\mathbf q}g^\sigma G^\sigma_{\mathbf k}$ 계산
4. 필요하면 $\partial\Delta_i=g_i^\uparrow-g_i^\downarrow$ onsite block 구성
5. 네 derivative trace 항을 contour 적분
6. $q\to R_p$ FFT
7. meV/Å 단위로 저장

### 9.3 Tensor $J^{ab}$

1. Collinear $H^\uparrow,H^\downarrow$로 spinor $H$ 구성하거나 full spinor Wannier Hamiltonian 입력
2. 같은 static Hamiltonian에 SOC 포함
3. Local $P_i$와 spinor $G^u$ 구성
4. Pair-complete $A^{uv}$ contour integral 계산
5. $J^{\mathrm{iso}}$, $\Gamma$, $\mathbf D$, $\mathbf J_{\mathrm{full}}$로 분해

### 9.4 Tensor $dJ^{ab}/du$

1. Collinear EPR $g^\uparrow,g^\downarrow$를 static spin frame의 spinor EPC로 embed
2. Band basis에서
   $$
   g_B(\mathbf k,\mathbf q)
   =C^\dagger(\mathbf k+\mathbf q)g(\mathbf k,\mathbf q)C(\mathbf k)
   $$
   구성
3. $G(\mathbf k+\mathbf q)gG(\mathbf k)$로 $\partial G$ 계산
4. 선택적으로 local $\partial P$ 계산
5. $\partial A^{uv}$의 네 항을 contour 적분
6. $q\to R_p$ FFT
7. `dJ_iso_r`, `dJ_gamma_r`, `dDMI_r`, `dJ_tensor_r`로 분해·저장

## 10. Approximation과 해석 범위

> [!warning] 반드시 명시할 가정
> 아래 가정이 결과의 의미를 결정한다. 논문이나 발표 자료에서는 계산식과 함께 밝혀야 한다.

1. **Magnetic force theorem / frozen potential**
   Spin rotation에 대해 self-consistent potential을 다시 풀지 않고 reference-state one-electron Hamiltonian의 band-energy curvature를 사용한다.

2. **Adiabatic spin-lattice separation**
   Phonon displacement는 전자와 spin이 순간적으로 따라가는 static parameter로 취급한다. 주파수 의존 exchange kernel은 계산하지 않는다.

3. **Linear displacement response**
   $J(u)=J(0)+(\partial J/\partial u)u+O(u^2)$까지만 유지한다.

4. **Chosen local magnetic subspace**
   $\Delta_i$, $P_i$, orbital trace는 지정된 local orbital slice에 의존한다. Wannier gauge와 magnetic projector를 바꾸면 orbital-resolved 값이 달라질 수 있다.

5. **Static reference SOC**
   Tensor $J$와 tensor $dJ$는 동일한 SOC Hamiltonian과 spin direction을 사용해야 한다. Static $J$는 SOC를 켜고 dynamic $dJ$는 다른 SOC를 쓰는 조합은 일관되지 않다.

6. **Current EPR EPC의 spin structure**
   Collinear $g^\uparrow,g^\downarrow$를 chosen spin direction으로 spinor embedding한다. Full spin-off-diagonal EPC나 displacement derivative of SOC가 별도로 입력되지 않으면 포함되지 않는다.

7. **Fixed Fermi reference**
   현재 derivative kernel은 입력 $E_F$를 고정한다. Fixed particle number에서의 $\partial E_F/\partial u$ 보정이 중요한 metallic system에서는 별도 검토가 필요하다.

8. **Orthogonal versus non-orthogonal basis**
   EPR/Wannier 경로의 $\partial G=GgG$와 raw LCAO 경로의 $G[\partial H-z\partial S]G$를 구분해야 한다.

## 11. 필수 검증

### 11.1 Scalar 및 tensor consistency

- SOC-off collinear limit에서 $J^{xx}\simeq J^{yy}\simeq J^{zz}\simeq J^{\mathrm{iso}}$
- SOC-off에서 off-diagonal tensor, DMI, $d$DMI가 numerical noise 수준
- Scalar LKAG와 tensor TB2J의 $J^{\mathrm{iso}}$가 같은 bond/sign/normalization convention에서 일치
- $(i,j,\mathbf R)$와 $(j,i,-\mathbf R)$ mate가 모두 존재

### 11.2 EPC 및 derivative consistency

- Reciprocal-space Hermiticity:
  $$
  g(\mathbf k,\mathbf q)
  \simeq g^\dagger(\mathbf k+\mathbf q,-\mathbf q)
  $$
- Analytic derivative와 finite difference of full LKAG:
  $$
  \frac{\partial J}{\partial u}
  \simeq\frac{J(+\delta u)-J(-\delta u)}{2\delta u}
  $$
- $\partial P$ on/off 비교로 onsite exchange-field response의 크기 확인
- $k$ mesh, $q$ mesh, contour point 수, contour lower bound, displacement amplitude 수렴

### 11.3 Acoustic sum rule

모든 atom을 동일하게 rigid translation하면 exchange는 변하지 않아야 하므로 각 bond와 Cartesian 방향에 대해

$$
\sum_{\kappa,\mathbf R_p}
\frac{\partial J_{ij}^{ab}(\mathbf R)}
{\partial u_{\kappa\mu}(\mathbf R_p)}=0
$$

이어야 한다. ASR이 맞는 것은 필요조건이지만 개별 bond component가 수렴했다는 충분조건은 아니다.

### 11.4 단위와 factor audit

| quantity | SLW의 대표 단위 |
|---|---|
| $H$, $E_F$, contour $z$ | eV 내부 단위 |
| $g=\partial H/\partial u$ | eV/Å |
| $J$ | meV |
| $\partial J/\partial u$ | meV/Å |
| $\partial J/\partial Q$ | normal-coordinate convention에 의존 |

특히 다음 세 항목을 동시에 audit해야 한다.

- scalar $\Delta$ 경로의 $1/(4\pi)$와 tensor $P=\Delta/2$ 경로의 $1/\pi$
- directed-bond list의 $1/2$
- quantum-spin model로 변환할 때의 $1/S^2$

## 12. 코드와 이론식의 대응

| 이론 object | 구현 위치 | 핵심 함수 또는 dataset |
|---|---|---|
| Scalar LKAG trace | [`slw/exchange/lkag_solver.py`](../slw/exchange/lkag_solver.py) | `lkag_trace_njit`, `compute_J_bulk` |
| Scalar 네 항 $dJ$ | [`slw/exchange/lkag_solver.py`](../slw/exchange/lkag_solver.py) | `lkag_deriv_trace_njit`, `lkag_deriv_terms_njit` |
| EPR $g(\mathbf k,\mathbf q)$ | [`slw/exchange/compute_dJ_epr_kspace.py`](../slw/exchange/compute_dJ_epr_kspace.py) | `_build_gkq_one` |
| $\mathbf k+\mathbf q$ mapping 및 scalar $\partial G$ | [`slw/exchange/compute_dJ_epr_kspace.py`](../slw/exchange/compute_dJ_epr_kspace.py) | `_kq_map`, `_compute_chunk` |
| Static spinor/TB2J tensor | [`slw/exchange/compute_J_epr_tensor.py`](../slw/exchange/compute_J_epr_tensor.py) | `_build_tb2j_projectors`, `_compute_tensor_tb2j`, `_decompose_tb2j_pair` |
| Tensor $\partial A^{uv}$ | [`slw/exchange/compute_dJ_epr_tensor.py`](../slw/exchange/compute_dJ_epr_tensor.py) | `_compute_band_dg`, `_accumulate_dA_from_blocks`, `_compute_tensor_chunk_analytic` |
| $dJ^{ab}$, $d\Gamma$, $d$DMI 분해 | [`slw/exchange/compute_dJ_epr_tensor.py`](../slw/exchange/compute_dJ_epr_tensor.py) | `_decompose_tb2j_dA`; `dJ_tensor_r`, `dJ_gamma_r`, `dDMI_r` |
| MPI task distribution | [`slw/exchange/compute_dJ_epr_tensor_mpi.py`](../slw/exchange/compute_dJ_epr_tensor_mpi.py) | target atom $\times$ displacement axis 분배 |
| EPC EPR real-space wrapper | [`slw/exchange/eph_epr_wrapper.py`](../slw/exchange/eph_epr_wrapper.py) | $g(\mathbf R_e,\mathbf R_p)$ loading/transform |
| ASR 검사 | [`slw/exchange/check_dJ_asr.py`](../slw/exchange/check_dJ_asr.py) | tensor derivative acoustic sum rule |

## 13. 관련 프로젝트 노트

- [`dynamical_dmi_magph.md`](dynamical_dmi_magph.md): $d$DMI로부터 magnon-phonon vertex를 만드는 후속 이론
- [`SCOPE.md`](SCOPE.md): 현재 저장소의 코드 경계와 제외 대상

## 14. 한 문단으로 정리

LKAG exchange는 두 magnetic site의 local spin-splitting vertex와 그 사이를 왕복하는 spin-dependent Green's function으로 이루어진 closed electronic loop이다. Atomic displacement가 가해지면 EPC $g=\partial H/\partial u$가 Dyson identity를 통해 loop의 두 propagator를 $\partial G=GgG$로 바꾸고, spin-dependent onsite EPC 차이가 두 local exchange vertex를 $\partial\Delta$ 또는 $\partial P$로 바꾼다. 이 네 변화를 모두 합한 것이 $\partial J/\partial u$이다. SOC가 있는 경우에는 spinor Green's function을 Pauli component로 분해한 $A^{uv}$ kernel에 똑같은 product rule을 적용하고, 그 derivative를 isotropic exchange, symmetric-traceless anisotropy, DMI derivative로 재조립한다. 따라서 electron-phonon coupling은 LKAG와 별개의 보정이 아니라, exchange interaction의 lattice derivative를 생성하는 전자 perturbation vertex 그 자체이다.
