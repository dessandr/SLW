# Equation Index

Use these labels in docstrings, tests, and validation reports.

## Model and time reversal

**WT-E01** `G_k(z)=[zI-H(k)]^{-1}`.

**WT-E02** `H=H_TRS+H_XC`.

**WT-E03** `H^Theta(k)=B_Theta(k) H(-k)^* B_Theta(k)^dagger`.

**WT-E04** `H_TRS=(H+H^Theta)/2`, `H_XC=(H-H^Theta)/2`.

## Local magnetic operators

**WT-L01** `L_ell[X]=(P_ell X+X P_ell)/2`.

**WT-T01** `e_ell=n_ell+pi_ell,a t_ell,a+O(pi^2)`.

**WT-T02** `T_ell,a=dH/dpi_ell,a`.

**WT-T03** `Gamma_ell,c=i[H_XC,ell,sigma_c]/2`.

**WT-T04** `Gamma_ell,1=-T_ell,2`, `Gamma_ell,2=+T_ell,1`.

**WT-T05** `Gamma_ell,c(k+q,k)=[P_ell(q)Gamma_c(k)+Gamma_c(k+q)P_ell(q)]/2`.

## Mixed response

**WT-K01** `dG/du=G g G`.

**WT-K02** retarded finite-q loop `A^R=T(-q)G(k+q)g(q)G(k)`.

**WT-K03** `K(q)=-[A^R(q)-A^R(-q)^*]/(2 pi i)`.

**WT-K04** direct vertex `T^(1)=d^2H/(dpi du)`.

## Projection

**WT-P01** `xi=sqrt(hbar/(2 M omega)) e`.

**WT-P02** `V_pi,nu=sum K_pi,kappa,mu xi_kappa,mu,nu`.

**WT-M01** `Psi_m=T_m Gamma_m`, `T_m^dagger Sigma T_m=Sigma`.

**WT-M02** `pi^T(-q)=Psi_m^dagger C_m`.

**WT-M03** `V_mp=C_m V_pi,nu C_p`.

**WT-M04** `V_tilde=T_m^dagger V_mp T_p`.

## Validation

**WT-S01** `K(-q)=K(q)^*`.

**WT-S02** mixed central finite difference.

**WT-S03** q=0 rigid-translation sum rule.
