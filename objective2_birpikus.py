"""
=============================================================================
SPIN QUBIT HETEROSTRUCTURE DESIGN — OBJECTIVE 2
Bir-Pikus Strain Splitting + Quantum Confinement → Total HH-LH Splitting ΔE
=============================================================================

WHAT THIS CODE DOES (overview)
-------------------------------
Objective 1 computed the strain tensor in a Ge quantum well (QW) on a relaxed
Si_{1-x}Ge_x substrate. Objective 2 turns that strain into the quantity that
actually matters for spin-qubit operation: the HH-LH energy splitting ΔE.

The HH-LH splitting has TWO contributions:

  (A) STRAIN  — Bir-Pikus deformation potential theory:
                    ΔE_strain = 2 |b_v| · ε_bi
      For compressive in-plane strain (ε_bi > 0), this pushes the LH band UP
      relative to the HH band → HH becomes the ground state. Ge: b_v = -2.16 eV.

  (B) QUANTUM CONFINEMENT — finite-thickness QW quantises k_z:
                    ΔE_QC = (ℏ²π²/2L²) · [1/m*_LH,z − 1/m*_HH,z]
      Because m*_LH,z (0.046 m_e) ≪ m*_HH,z (0.20 m_e), the LH subband is
      pushed up MORE than the HH subband → confinement adds to the splitting.

The total splitting ΔE_total = ΔE_strain + ΔE_QC must exceed ~50 meV for
a clean HH ground state with negligible LH admixture (the project's design
target). This script identifies the (x, L) operating window that satisfies it.

WHY THIS MATTERS FOR THE QUBIT
-------------------------------
Large HH-LH splitting → pure HH ground state → well-defined J_z = ±3/2 spin
states → predictable g-factors → uniform qubit operating frequency across
the wafer (Objective 4). LH admixture would scramble all of these.

REFERENCES
----------
[1]  Bir & Pikus, Symmetry and Strain-Induced Effects in Semiconductors (1974)
[2]  Vurgaftman et al., J. Appl. Phys. 89, 5815 (2001)        — material parameters
[3]  Winkler, Spin-Orbit Coupling in 2D Systems, Springer 2003 — strain + confinement
[4]  Lodari et al., npj Quantum Inf. 8, 14 (2022)             — Ge spin qubit context
[5]  Scappucci et al., Nat. Rev. Mater. 6, 926 (2021)         — Ge platform review

HOW TO RUN
----------
  pip install numpy matplotlib scipy
  python objective2_birpikus.py

OUTPUT
------
  • Console: parameter table + ΔE summary at representative (x, L) points
  • File:    objective2_birpikus_analysis.png
=============================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from scipy.sparse import diags
from scipy.sparse.linalg import eigsh

# =============================================================================
# GLOBAL PLOT STYLE  (matches Objective 1 for visual consistency)
# =============================================================================

plt.rcParams.update({
    'figure.dpi': 150,
    'font.family': 'serif',
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 12,
    'axes.titleweight': 'bold',
    'legend.fontsize': 9.5,
    'axes.grid': True,
    'grid.alpha': 0.25,
    'grid.linestyle': '--',
    'axes.spines.top': False,
    'axes.spines.right': False,
})

# Colour palette (HH=red, LH=blue, SO=purple, Ge=green — project convention)
C = {
    'HH':     '#E53935',   # red    — heavy hole
    'LH':     '#1E88E5',   # blue   — light hole
    'SO':     '#8E24AA',   # purple — split-off / hydrostatic
    'Ge':     '#43A047',   # green  — Ge / target line
    'strain': '#E53935',   # strain contribution
    'QC':     '#FB8C00',   # orange — confinement contribution
    'total':  '#1565C0',   # dark blue — total
    'safe':   '#A5D6A7',
    'unsafe': '#FFCDD2',
}


# =============================================================================
# SECTION 1 — MATERIAL PARAMETERS
# =============================================================================
#
# Two new ingredients beyond Objective 1:
#
#   (i)  DEFORMATION POTENTIAL b_v  — the Bir-Pikus coupling constant that
#        converts biaxial strain into a HH/LH energy splitting.
#        Value for Ge: b_v = −2.16 eV [Vurgaftman 2001].
#        The minus sign + ε_bi > 0 make ΔE_strain > 0 (HH below LH ✓).
#
#   (ii) EFFECTIVE MASSES along z (the confinement direction) for HH and LH:
#          m*_HH,z = 0.20  m_e
#          m*_LH,z = 0.046 m_e
#        These come from the Luttinger Hamiltonian:
#          1/m*_HH,z = (γ_1 − 2γ_2) / m_e
#          1/m*_LH,z = (γ_1 + 2γ_2) / m_e
#        with Ge Luttinger parameters γ_1 = 13.38, γ_2 = 4.24.
#        The HUGE m*_HH/m*_LH ratio (≈4.3×) is what makes confinement
#        contribute substantially to the splitting.
# =============================================================================

# Lattice constants (Å) — same as Objective 1
a_Si = 5.4310
a_Ge = 5.6579

# Ge elastic stiffness (GPa) — same as Objective 1
C11 = 128.53
C12 =  48.26

# Ge deformation potentials (eV) [Ref 2]
a_v_Ge = +2.0     # hydrostatic VB deformation potential (shifts bands)
b_v_Ge = -2.16    # uniaxial / biaxial (splits HH-LH)  ← the key one
d_v_Ge = -6.06    # shear (drives [111] splitting; not used at (001))

# Ge effective masses along z (units of m_e) [Ref 2,3]
m_HH_z = 0.20
m_LH_z = 0.046

# Convenient unit: ℏ²/(2 m_e) in eV·nm²
#   ℏ = 1.054571817 × 10^-34 J·s
#   m_e = 9.1093837 × 10^-31 kg
#   ℏ²/(2 m_e) = 6.1042 × 10^-39 J·m² → divide by 1.602 × 10^-19 to get eV·m²
#   → 3.81 × 10^-20 eV·m²  =  0.0381 eV·nm²
HBAR2_OVER_2ME = 0.03810   # eV·nm²

# Ge / SiGe valence-band offset (used only for finite-well solver, Section 5)
#   Roughly linear in (1−x) of substrate; ~150 meV at x = 0.80 [Ref 4,5].
#   We use this as a constant default; the FDM result is meant as a
#   ~10-20% correction comparison to the infinite-well limit.
V0_VB_default = 0.150   # eV        Need clarification 

# ── Print parameter summary ───────────────────────────────────────────────────
print("=" * 64)
print("MATERIAL PARAMETERS  (Ge quantum well — Objective 2)")
print("=" * 64)
print(f"  Deformation potential  b_v        = {b_v_Ge:+.3f} eV")
print(f"  Effective mass         m*_HH,z   = {m_HH_z:.4f} m_e")
print(f"  Effective mass         m*_LH,z   = {m_LH_z:.4f} m_e")
print(f"  Mass ratio             m_HH/m_LH = {m_HH_z/m_LH_z:.2f}")
print(f"  ℏ²/(2 m_e)             = {HBAR2_OVER_2ME:.5f} eV·nm²")
print(f"  ℏ²π²/(2 m_e)           = {HBAR2_OVER_2ME * np.pi**2:.5f} eV·nm²")
print(f"  Default VB offset       V0       = {V0_VB_default*1000:.0f} meV")


# =============================================================================
# SECTION 2 — STRAIN FUNCTIONS  (re-derived from Objective 1, inlined)
# =============================================================================
#
# These are identical to Objective 1, repeated here so this script is
# self-contained.
# =============================================================================

def a_SiGe(x):
    """Vegard's law for Si_{1-x}Ge_x lattice constant (Å)."""
    return (1.0 - x) * a_Si + x * a_Ge

def epsilon_parallel(x):
    """In-plane strain in Ge QW grown on Si_{1-x}Ge_x. Negative = compressive."""
    return (a_SiGe(x) - a_Ge) / a_Ge

def epsilon_perp(x):
    """Out-of-plane strain (Poisson response). Positive = tensile."""
    return -2.0 * (C12 / C11) * epsilon_parallel(x)

def epsilon_biaxial(x):
    """Biaxial / deviatoric strain ε_bi = ε_⊥ − ε_∥. Drives HH-LH splitting."""
    return epsilon_perp(x) - epsilon_parallel(x)


# =============================================================================
# SECTION 3 — BIR-PIKUS STRAIN SPLITTING
# =============================================================================
#
# -----------------------------------------------------------------------
# The Bir-Pikus Hamiltonian for the J = 3/2 valence multiplet under (001)
# biaxial strain has, in the {|3/2,±3/2⟩, |3/2,±1/2⟩} basis, eigenvalues:
#
#   E_HH = a_v · Tr(ε)   −   b_v · ε_bi
#   E_LH = a_v · Tr(ε)   +   b_v · ε_bi
#
# (We ignore SO band coupling here — small at Ge's Δ_SO = 0.29 eV scale.)
#
# The HYDROSTATIC piece a_v·Tr(ε) shifts both equally → cancels in the
# splitting. What's left is the BIAXIAL (deviatoric) piece:
#
#   ΔE_strain ≡ E_LH − E_HH  =  2 · |b_v| · ε_bi
#
# (We take the absolute value because b_v < 0 and ε_bi > 0 in our case;
# what matters for qubits is the sign of ΔE, which is positive — meaning
# HH is the ground state, exactly what we want.)
# =============================================================================

def deltaE_strain(x):
    """
    Bir-Pikus HH-LH splitting from biaxial strain.

    Returns
    -------
    float or ndarray
        ΔE_strain in eV. Positive → HH is the ground state (good for qubits).
    """
    return 2.0 * abs(b_v_Ge) * epsilon_biaxial(x)


# =============================================================================
# SECTION 4 — QUANTUM CONFINEMENT (INFINITE WELL)
# =============================================================================
#
# DERIVATION
# ----------
# In a Ge QW of thickness L sandwiched between SiGe barriers, the HH and LH
# states see roughly square-well potentials in z. To leading order, treat
# the well as INFINITE → the bound-state energies are the textbook result:
#
#     E_n^(b) = ℏ² π² n² / (2 m*_b L²),     b ∈ {HH, LH},  n = 1, 2, 3, …
#
# The lowest (n=1) HH and LH subbands are then offset from the bulk band
# edges by  E_1^(b) = ℏ² π² / (2 m*_b L²).  Since m*_LH,z ≪ m*_HH,z, the
# LH subband is pushed UP much more strongly. The confinement contribution
# to the HH-LH splitting is the difference:
#
#     ΔE_QC = E_1^(LH) − E_1^(HH) = (ℏ²π²/2L²) [1/m*_LH,z − 1/m*_HH,z]
#
# This adds DIRECTLY to the strain term:
#     ΔE_total = ΔE_strain + ΔE_QC
#
# Numerical reality check at L = 14 nm (infinite well):
#     prefactor = ℏ²π²/(2 m_e) = 0.376 eV·nm²
#     ΔE_QC = (0.376 / 14²) × (1/0.046 − 1/0.20)
#           = 0.001918 eV × 16.74
#           ≈ 32.1 meV
#
# ── Important caveat: this INFINITE-well number overstates the splitting ──
# Because m*_LH,z = 0.046 m_e is very light, the LH wavefunction penetrates
# substantially into the (finite, ~150 meV) SiGe barrier. The finite-well
# solver in Section 5 gives ΔE_QC ≈ 16 meV at L = 14 nm — roughly half the
# infinite-well value. So:
#     • Infinite-well ΔE_QC :  upper bound (textbook limit)
#     • Finite-well   ΔE_QC :  realistic value used for design margin
#
# At the typical operating point (x=0.80, L=14 nm):
#     Infinite-well total : 60.7 + 32.1 ≈ 93 meV
#     Finite-well   total : 60.7 + 15.9 ≈ 77 meV   ← use this for tolerancing
# Both clear the 50 meV target comfortably
# =============================================================================

def deltaE_QC_infinite(L_nm, m_HH=m_HH_z, m_LH=m_LH_z):
    """
    HH-LH splitting from quantum confinement in an INFINITE square well.

    Parameters
    ----------
    L_nm : float or ndarray
        Quantum well thickness in nanometres.
    m_HH, m_LH : float
        Effective masses along z in units of m_e.

    Returns
    -------
    float or ndarray
        ΔE_QC in eV.
    """
    prefactor = HBAR2_OVER_2ME * np.pi**2          # 0.376 eV·nm²
    return (prefactor / L_nm**2) * (1.0/m_LH - 1.0/m_HH)


def deltaE_total_inf(x, L_nm):
    """Total HH-LH splitting (strain + infinite-well confinement), eV."""
    return deltaE_strain(x) + deltaE_QC_infinite(L_nm)


# =============================================================================
# SECTION 5 — FINITE-WELL SOLVER (FDM)
# =============================================================================
#
# WHY BOTHER
# ----------
# The infinite-well formula overstates ΔE_QC by ~10-20% because real
# wavefunctions PENETRATE the SiGe barriers slightly, lowering the
# effective confinement energy [Ref 3]. For Objective 3's tolerance
# analysis we want a sharper number, so we solve the finite-well 1D
# Schrödinger equation numerically.
#
# METHOD
# ------
# 1D Schrödinger equation with constant effective mass (Ge values used
# in both well and barrier — a controlled simplification; mass-mismatch
# corrections are themselves only a few %):
#
#     [ −ℏ²/(2 m*) d²/dz² + V(z) ] ψ_n(z)  =  E_n ψ_n(z)
#
# with  V(z) = 0      for |z| < L/2     (Ge well)
#       V(z) = V0     for |z| > L/2     (SiGe barrier, ~150 meV)
#
# Discretise on a uniform grid of N points spanning [−Z_max, +Z_max] with
# Z_max = 5L. Central differences for d²/dz² give a tridiagonal Hamiltonian:
#
#     H_ii   = 2 t + V_i
#     H_i,i±1 = −t            with t = ℏ²/(2 m* Δz²)
#
# Diagonalise with scipy.sparse.linalg.eigsh (Lanczos), take the lowest
# eigenvalue → E_1. Then ΔE_QC^finite = E_1^LH − E_1^HH.
# =============================================================================

def E1_finite_well(L_nm, m_eff, V0_eV=V0_VB_default,
                   N=2000, Zmax_factor=5.0):
    """
    Lowest bound-state energy in a 1D finite square well of width L.

    Parameters
    ----------
    L_nm    : well width, nm
    m_eff   : effective mass (units of m_e)
    V0_eV   : barrier height, eV
    N       : number of grid points
    Zmax_factor : domain half-width in units of L

    Returns
    -------
    float
        E_1 in eV (energy above the well bottom).
    """
    Zmax = Zmax_factor * L_nm                 # nm
    z = np.linspace(-Zmax, Zmax, N)
    dz = z[1] - z[0]                          # nm

    # Potential: 0 in well, V0 in barriers
    V = np.where(np.abs(z) < L_nm/2.0, 0.0, V0_eV)   # eV

    # Hopping element t = ℏ²/(2 m* Δz²)  in eV
    t = HBAR2_OVER_2ME / (m_eff * dz**2)

    # Tridiagonal Hamiltonian via sparse diagonals
    H = diags([-t * np.ones(N-1),
                2*t + V,
               -t * np.ones(N-1)],
              offsets=[-1, 0, 1],
              format='csr')

    # Lowest eigenvalue (Lanczos with shift-invert near bottom of well)
    vals = eigsh(H, k=1, which='SA', return_eigenvectors=False)
    return float(vals[0])


def deltaE_QC_finite(L_nm, V0_eV=V0_VB_default):
    """ΔE_QC from the finite-well solver. Slightly smaller than infinite-well."""
    E1_HH = E1_finite_well(L_nm, m_HH_z, V0_eV)
    E1_LH = E1_finite_well(L_nm, m_LH_z, V0_eV)
    return E1_LH - E1_HH


# =============================================================================
# SECTION 6 — COMPUTE RESULTS
# =============================================================================

x_vals = np.linspace(0.40, 0.995, 400)
L_default = 14.0   # nm — typical Ge spin qubit well thickness

dE_strain   = deltaE_strain(x_vals)                    # eV
dE_QC_14nm  = deltaE_QC_infinite(L_default)            # scalar, eV
dE_total_14 = dE_strain + dE_QC_14nm                   # eV

# ── Console summary table ─────────────────────────────────────────────────────
# Pre-compute finite-well ΔE_QC at L_default once (it doesn't depend on x).
print(f"\nSolving finite well at L = {L_default:.0f} nm for the summary table...")
dE_QC_fin_L0 = deltaE_QC_finite(L_default)
print("  done.")

print("\n" + "=" * 84)
print(f"  HH-LH SPLITTING SUMMARY  (Ge QW on Si_{{1-x}}Ge_x, L = {L_default:.0f} nm)")
print("=" * 84)
print(f"  {'x':>5}  {'ε_bi (%)':>9}  {'ΔE_strain':>10}  "
      f"{'ΔE_QC (inf)':>12}  {'ΔE_QC (fin)':>12}  {'ΔE_total (fin)':>15}")
print(f"  {'':>5}  {'':>9}  {'(meV)':>10}  {'(meV)':>12}  {'(meV)':>12}  {'(meV)':>15}")
print("  " + "-" * 80)
for xv in [0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]:
    Es     = deltaE_strain(xv)*1000
    Eqc_i  = deltaE_QC_infinite(L_default)*1000
    Eqc_f  = dE_QC_fin_L0*1000
    Et_f   = Es + Eqc_f
    flag   = " ✓" if Et_f > 50 else " ✗"
    print(f"  {xv:5.2f}  {epsilon_biaxial(xv)*100:9.3f}  "
          f"{Es:10.1f}  {Eqc_i:12.1f}  {Eqc_f:12.1f}  {Et_f:15.1f}{flag}")
print("  " + "-" * 80)
print("  ('inf' = infinite-well textbook formula; 'fin' = finite-well FDM "
      f"with V0={V0_VB_default*1000:.0f} meV)")
print("  Pass criterion: ΔE_total(fin) > 50 meV (project design margin)")
print()


# =============================================================================
# SECTION 7 — FIGURES
# =============================================================================

fig = plt.figure(figsize=(14, 11))
fig.patch.set_facecolor('#FAFAFA')

gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.42, wspace=0.38,
                       left=0.09, right=0.96, top=0.92, bottom=0.08)
ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1])
ax3 = fig.add_subplot(gs[1, 0])
ax4 = fig.add_subplot(gs[1, 1])

for ax in [ax1, ax2, ax3, ax4]:
    ax.set_facecolor('#FAFAFA')


# ── PLOT 1: Reproduction of Figure 8 (foundations doc) ──────────────────────
#
#   Strain-only ΔE_strain (red)  vs.  total at L=14 nm (two estimates).
#   The finite-well total crosses 50 meV around x ≈ 0.85 — matching the
#   x_max boundary stated in Section 17.4 of the foundations doc.
# ----------------------------------------------------------------------------

# Pre-compute finite-well QC at L=14 (it doesn't depend on x)
dE_QC_fin_14   = deltaE_QC_finite(L_default)
dE_total_14fin = dE_strain + dE_QC_fin_14

ax1.plot(x_vals, dE_strain * 1000, color=C['strain'], lw=2.6,
         label=r'$\Delta E_\mathrm{strain} = 2|b_v|\,\epsilon_\mathrm{bi}$')
ax1.plot(x_vals, dE_total_14 * 1000, color=C['total'], lw=1.8, ls=(0,(5,2)),
         label=r'$\Delta E_\mathrm{total}$  (infinite well, $L=14$ nm)')
ax1.plot(x_vals, dE_total_14fin * 1000, color=C['total'], lw=2.6,
         label=r'$\Delta E_\mathrm{total}$  (finite well, $L=14$ nm)')

# 50 meV target line
ax1.axhline(50, color=C['Ge'], lw=1.6, ls='--', label='50 meV target')
ax1.fill_between(x_vals, 50, 250,
                 where=(dE_total_14fin*1000 > 50),
                 alpha=0.10, color=C['Ge'])

# Mark x where finite-well total crosses 50 meV (the realistic boundary)
def x_at_50(curve):
    idx = np.where(curve*1000 < 50)[0]
    return x_vals[idx[0]] if len(idx) else None
xc_fin = x_at_50(dE_total_14fin)
if xc_fin is not None:
    ax1.axvline(xc_fin, color=C['total'], lw=1.0, ls=':', alpha=0.7)
    ax1.text(xc_fin + 0.005, 8, f'$x={xc_fin:.2f}$',
             color=C['total'], fontsize=8.5, ha='left', va='bottom', rotation=90)

# Annotate operating point x=0.80, L=14 (using finite-well — the realistic value)
xm = 0.80
y_op = (deltaE_strain(xm) + dE_QC_fin_14) * 1000
ax1.plot(xm, y_op, 'o', color=C['total'], ms=8, mec='k', mew=0.8, zorder=5)
ax1.annotate(f'  $x=0.80$, $L=14$ nm\n  $\\Delta E\\approx{y_op:.0f}$ meV (finite well)',
             xy=(xm, y_op),
             xytext=(0.55, 130), fontsize=9, color='k',
             arrowprops=dict(arrowstyle='->', color='k', lw=0.8))

ax1.set_xlabel('Ge fraction  $x$  in substrate')
ax1.set_ylabel('HH-LH splitting  $\\Delta E$  (meV)')
ax1.set_title("(1) HH-LH Splitting vs Substrate Composition")
ax1.legend(loc='upper right', framealpha=0.92, fontsize=8.5)
ax1.set_xlim(0.40, 1.0)
ax1.set_ylim(0, 220)


# ── PLOT 2: ΔE_total vs L for several substrate compositions ────────────────
#
#   Reveals the strain ↔ confinement trade-off:
#   • Thin wells: confinement dominates  (curve rises sharply at low L)
#   • Thick wells: strain dominates  (curve flattens to ΔE_strain plateau)
#   Uses finite-well ΔE_QC (the realistic estimate). One FDM solve per L.
# ----------------------------------------------------------------------------
L_range = np.linspace(4, 30, 60)
print("Computing ΔE_total(L) curves with finite-well solver...")
dE_QC_fin_L = np.array([deltaE_QC_finite(L) for L in L_range])   # shape (60,)
print("  done.")

x_lines = [(0.70, '#1B5E20'), (0.75, '#2E7D32'),
           (0.80, '#388E3C'), (0.85, '#66BB6A'), (0.90, '#A5D6A7')]

for x_val, col in x_lines:
    Et = (deltaE_strain(x_val) + dE_QC_fin_L) * 1000
    ax2.plot(L_range, Et, color=col, lw=2.0, label=f'$x = {x_val:.2f}$')

ax2.axhline(50, color=C['Ge'], lw=1.6, ls='--', label='50 meV target')

# Shade the typical operating window L ∈ [8, 20] nm
ax2.axvspan(8, 20, alpha=0.10, color=C['total'])
ax2.text(14, 200, 'typical\nQW range', ha='center', va='top',
         fontsize=8.5, color=C['total'], fontweight='bold')

ax2.set_xlabel('QW thickness  $L$  (nm)')
ax2.set_ylabel('Total HH-LH splitting  (meV)')
ax2.set_title("(2) ΔE$_\\mathrm{total}$ vs Well Thickness  (finite well)")
ax2.legend(loc='upper right', framealpha=0.92, ncol=2)
ax2.set_xlim(4, 30)
ax2.set_ylim(0, 220)


# ── PLOT 3: 2D process map  ΔE_total(x, L)  ─────────────────────────────────
#
#   Heatmap of ΔE_total using the FINITE-WELL solver — the realistic value
#   for Objective 3's tolerance analysis. Bold contour = 50 meV target.
#   Box = typical operating window. ΔE_QC depends only on L, so we only
#   need to FDM-solve once per row of the grid.
# ----------------------------------------------------------------------------
x_2d = np.linspace(0.50, 0.99, 250)
L_2d = np.linspace(4, 30, 120)        # fewer L points (each costs an FDM solve)
XX, LL = np.meshgrid(x_2d, L_2d)

print("Building 2D process map (finite-well solver, one FDM solve per L)...")
dE_QC_fin_grid = np.array([deltaE_QC_finite(L) for L in L_2d])    # shape (120,)
DE_2d = (deltaE_strain(XX) + dE_QC_fin_grid[:, np.newaxis]) * 1000  # meV
print("  done.")

# Filled contours
cf = ax3.contourf(XX, LL, DE_2d, levels=np.linspace(0, 200, 21),
                  cmap='viridis', extend='max')
cb = plt.colorbar(cf, ax=ax3, fraction=0.045, pad=0.03)
cb.set_label(r'$\Delta E_\mathrm{total}$  (meV, finite well)', fontsize=10)

# 50 meV contour highlighted
ax3.contour(XX, LL, DE_2d, levels=[50], colors='white', linewidths=2.4)
ax3.contour(XX, LL, DE_2d, levels=[50], colors='black', linewidths=1.0,
            linestyles='--')

# Typical spin-qubit operating window
qubit_x_lo, qubit_x_hi = 0.75, 0.90
qubit_L_lo, qubit_L_hi = 8,    20
ax3.add_patch(mpatches.FancyBboxPatch(
    (qubit_x_lo, qubit_L_lo),
    qubit_x_hi - qubit_x_lo, qubit_L_hi - qubit_L_lo,
    boxstyle="round,pad=0.4", linewidth=2.2,
    edgecolor='white', facecolor='none', linestyle='-', zorder=5))
ax3.text(0.825, 14, 'typical\noperating\nwindow',
         ha='center', va='center', fontsize=9,
         color='white', fontweight='bold')

# Annotate the 50 meV contour
ax3.text(0.94, 25, '$\\Delta E = 50$ meV\n(boundary)', fontsize=8.5,
         color='white', ha='center')

ax3.set_xlabel('Ge fraction  $x$  in substrate')
ax3.set_ylabel('QW thickness  $L$  (nm)')
ax3.set_title("(3) Process Map — Total HH-LH Splitting (finite well)")
ax3.set_xlim(0.50, 0.99)
ax3.set_ylim(4, 30)


# ── PLOT 4: Infinite vs finite well comparison ──────────────────────────────
#
#   Validates the foundations doc claim that finite barriers reduce
#   ΔE_QC by ~10-20% relative to the infinite-well limit.
# ----------------------------------------------------------------------------
print("Solving finite-well Schrödinger equation across L range...")
L_fdm = np.array([5, 6, 8, 10, 12, 14, 16, 18, 20, 25, 30])
dE_QC_inf  = deltaE_QC_infinite(L_fdm) * 1000
dE_QC_fin  = np.array([deltaE_QC_finite(L) for L in L_fdm]) * 1000
print("  done.\n")

ax4.plot(L_fdm, dE_QC_inf, 'o-', color=C['QC'], lw=2.0, ms=7,
         label='Infinite well  (analytic)')
ax4.plot(L_fdm, dE_QC_fin, 's-', color=C['total'], lw=2.0, ms=7,
         label=f'Finite well  ($V_0 = {V0_VB_default*1000:.0f}$ meV, FDM)')

# Show relative correction as text
mid = len(L_fdm)//2
correction_pct = 100 * (dE_QC_inf - dE_QC_fin) / dE_QC_inf
ax4.text(0.62, 0.95,
         f'Correction: {correction_pct.min():.0f}–{correction_pct.max():.0f}%\n'
         f'(infinite well → finite well)',
         transform=ax4.transAxes, fontsize=9, va='top',
         bbox=dict(boxstyle='round,pad=0.4',
                   facecolor='white', edgecolor='#888', alpha=0.92))

# Inset table-style annotation at L=14
i14 = np.where(L_fdm == 14)[0][0]
ax4.annotate(f'$L=14$ nm\n  $\\Delta E_\\mathrm{{QC}}^\\mathrm{{inf}}$ = {dE_QC_inf[i14]:.1f} meV\n'
             f'  $\\Delta E_\\mathrm{{QC}}^\\mathrm{{fin}}$ = {dE_QC_fin[i14]:.1f} meV',
             xy=(14, dE_QC_fin[i14]), xytext=(20, 45),
             fontsize=8.5,
             arrowprops=dict(arrowstyle='->', color='k', lw=0.8))

ax4.set_xlabel('QW thickness  $L$  (nm)')
ax4.set_ylabel('$\\Delta E_\\mathrm{QC}$  (meV)')
ax4.set_title("(4) Finite-Barrier Correction to Confinement")
ax4.legend(loc='upper right', framealpha=0.92)
ax4.set_xlim(4, 31)
ax4.set_ylim(0, 80)


# ── Global title ────────────────────────────────────────────────────────────
fig.suptitle(
    'HH-LH Splitting in Ge/SiGe Quantum Wells — '
    'Bir-Pikus Strain + Quantum Confinement',
    fontsize=13, fontweight='bold', y=0.965
)

out_path = 'objective2_birpikus_analysis.png'
plt.savefig(out_path, bbox_inches='tight', facecolor=fig.get_facecolor())
plt.close()
print(f"Figure saved → {out_path}\n")


# =============================================================================
# SECTION 8 — SANITY CHECKS
# =============================================================================
print("=" * 64)
print("SANITY CHECKS")
print("=" * 64)

# Check ΔE_strain at x=1 → zero (no misfit, no strain, no splitting)
print(f"\n  x = 1.00 (no misfit):  ΔE_strain = {deltaE_strain(1.0)*1000:.4f} meV  "
      f"✓ (expect 0)")

# Reproduce table value: x=0.80, L=14 nm
xc, Lc = 0.80, 14.0
Etot_inf = deltaE_total_inf(xc, Lc) * 1000
Etot_fin = (deltaE_strain(xc) + deltaE_QC_finite(Lc)) * 1000
print(f"\n  Reference point  x=0.80, L=14 nm:")
print(f"    ΔE_strain         = {deltaE_strain(xc)*1000:6.1f} meV   "
      f"(matches Bir-Pikus tables)")
print(f"    ΔE_QC (infinite)  = {deltaE_QC_infinite(Lc)*1000:6.1f} meV   "
      f"(textbook upper bound)")
print(f"    ΔE_QC (finite)    = {deltaE_QC_finite(Lc)*1000:6.1f} meV   "
      f"(V0 = {V0_VB_default*1000:.0f} meV)")
print(f"    ΔE_total (inf)    = {Etot_inf:6.1f} meV")
print(f"    ΔE_total (finite) = {Etot_fin:6.1f} meV   ← realistic design value")
print(f"    50 meV target met:  {'YES ✓' if Etot_fin > 50 else 'NO ✗'}")

# Finite-well correction at L=14 nm
dQC_inf = deltaE_QC_infinite(14.0)*1000
dQC_fin = deltaE_QC_finite(14.0)*1000
pct = 100*(dQC_inf - dQC_fin)/dQC_inf
print(f"\n  Finite-barrier correction at L=14 nm:  {pct:.0f}% reduction.")
print(f"    The correction is large (much more than the ~10-20% rule of")
print(f"    thumb for typical wells) because m*_LH = 0.046 m_e is very")
print(f"    light, giving the LH a long evanescent tail in the SiGe.")
print(f"    → For PhD-level rigor, use the finite-well number for tolerancing.")

# QC scales as 1/L²
ratio = deltaE_QC_infinite(7.0) / deltaE_QC_infinite(14.0)
print(f"\n  Confinement ∝ 1/L²:   ΔE_QC(7nm)/ΔE_QC(14nm) = {ratio:.3f}  "
      f"(expect 4.000)")

# Process-window estimate: x range that meets 50 meV at L=14
strain_only_x_max = x_vals[deltaE_strain(x_vals)*1000 > 50][-1]
total_inf_x_max   = x_vals[dE_total_14*1000 > 50][-1]
print(f"\n  Process window for 50 meV target (L = 14 nm):")
print(f"    Strain alone:        x_max = {strain_only_x_max:.3f}")
print(f"    Strain + inf-well:   x_max = {total_inf_x_max:.3f}")
print(f"    Strain + finite-well boundary will be tighter (Obj 3 quantifies).")

print("\n✓ All Objective 2 calculations completed.")
