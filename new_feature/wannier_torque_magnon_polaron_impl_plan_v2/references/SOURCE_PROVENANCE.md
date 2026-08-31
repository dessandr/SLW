# Equation-Level Source Provenance

## Provenance classes

- `SOURCE-DERIVED`: the cited paper explicitly derives or states the ingredient.
- `ORTHONORMAL-LIMIT`: direct specialization of a cited nonorthogonal-basis result to `S=I`.
- `STANDARD`: standard supporting formalism from the cited literature.
- `PROJECT-EXTENSION`: proposed completion or implementation choice not explicitly derived in the primary sources.

## Primary source MC2023

**Exact citation:** G. Martínez-Carracedo *et al.*, *Physical Review B* **108**, 214418 (2023), DOI `10.1103/PhysRevB.108.214418`.

| Ingredient | Location in source | Class | Package use |
|---|---|---|---|
| Hermitian local operator with half cross blocks | Eq. (13), Sec. II B | ORTHONORMAL-LIMIT | `L_ell[X]=(P_ell X+X P_ell)/2` |
| Local-operator sum rule | Eq. (14) | SOURCE-DERIVED | closure test |
| Spin-rotation commutators | Eq. (46) | SOURCE-DERIVED | infinitesimal rotation algebra |
| TRS/TRB Hamiltonian split | Eqs. (73)-(77) | SOURCE-DERIVED | exchange-field extraction |
| Full spin rotation also rotates SOC | Eqs. (78)-(82), Fig. 2 | SOURCE-DERIVED | prohibition of `-i[S,H_full]` |
| Only exchange field should be rotated | end of Sec. IV C | SOURCE-DERIVED | vertex construction |
| Local first-order XC rotation | Eq. (100) | SOURCE-DERIVED | `Gamma=i[H_XC,sigma]/2` |
| Localized-orbital selection sensitivity | Sec. IV E and Sec. V | SOURCE-DERIVED | magnetic-subspace audit |
| Local versus onsite sum-rule benchmark | Table I, Sec. V A | SOURCE-DERIVED | default local partition |

## Primary source MLE2023

**Exact citation:** S. Mankovsky *et al.*, *Physical Review B* **107**, 144428 (2023), DOI `10.1103/PhysRevB.107.144428`.

| Ingredient | Location in source | Class | Package use |
|---|---|---|---|
| Spin and displacement as independent perturbations | Sec. II B | SOURCE-DERIVED | bubble-only MVP |
| Linear spin and displacement matrices | Eqs. (6)-(7), (26)-(27) | SOURCE-DERIVED | two-vertex structure |
| Mixed Green-function expansion | Eqs. (20)-(25) | SOURCE-DERIVED | `T G g G` mapping |
| Site-diagonal mixed SLC coefficient | Eq. (29) | SOURCE-DERIVED | target mixed kernel structure |
| Effective field and torque interpretation | Eqs. (30)-(31) | SOURCE-DERIVED | physical torque response |
| KKR torque operator | Eqs. (C1)-(C2) | SOURCE-DERIVED | source-side vertex meaning |
| Torque as energy derivative | Appendix D | SOURCE-DERIVED | coordinate/sign relation |
| Phonon-like finite-q effective field | Eq. (35), Sec. III B | SOURCE-DERIVED | finite-q motivation |
| Nonlocal anisotropic contributions to effective field | Sec. III B and summary | SOURCE-DERIVED | total-kernel interpretation |

## Standard support

| Ingredient | Reference | Class |
|---|---|---|
| LKAG magnetic-force-theorem exchange mapping | Liechtenstein1987; Szilva2023 | STANDARD |
| Relativistic exchange-tensor formalism | Udvardi2003; EbertMankovsky2009 | STANDARD |
| Nonorthogonal local operators | SorianoPalacios2014; Oroszlany2019 | STANDARD |
| Screened DFPT lattice response | Baroni2001 | STANDARD |
| Wannier electron-phonon interpolation | Giustino2007 | STANDARD |
| Wannier localization and spinor support | Marzari2012; Pizzi2020 | STANDARD |
| Atomistic spin-lattice model context | Hellsvik2019 | STANDARD |
| Bosonic paraunitary diagonalization | Colpa1978 | STANDARD |
| Example magnon-polaron BdG physics | Klogetvedt2023 | STANDARD |

## Project extensions

| Package equation/decision | Class | Assumption |
|---|---|---|
| KKR `tau,U` replaced structurally by Wannier `G,g` | PROJECT-EXTENSION | common Dyson-insertion structure, not operator identity |
| arbitrary-Wannier-gauge time-reversal sewing matrix | PROJECT-EXTENSION | sewing matrix supplied and unitary |
| finite-q local selector and local-partition vertex | PROJECT-EXTENSION | explicit orbital/site centers and declared atomic gauge |
| finite-q momentum loop `T(-q)G(k+q)g(k,q)G(k)` | PROJECT-EXTENSION | common electronic gauge and commensurate meshes |
| q-pair kernel `-[A^R(q)-A^R(-q)^*]/(2 pi i)` | PROJECT-EXTENSION | Hermitian real-space perturbations |
| direct mixed vertex from `g_XC` | PROJECT-EXTENSION | exchange-field derivative and projector policy available |
| phonon zero-point projection | PROJECT-EXTENSION using STANDARD phonon quantization | eigenvector normalization known |
| external magnon projection `T_m^dagger V T_p` | PROJECT-EXTENSION using STANDARD paraunitarity | Nambu order and local frames known |
| HDF5/restart/validation architecture | PROJECT-EXTENSION | software design |

## Novelty wording boundary

Safe wording:

> A q-resolved spinor-Wannier Green-function implementation of displacement-induced magnetic torque using screened DFPT perturbations, followed by direct projection onto independently calculated phonon and magnon modes.

Avoid claiming invention of relativistic localized-orbital LKAG, exchange-field-only rotation, the KKR site-diagonal spin-lattice torque expression, or bosonic paraunitarity.
