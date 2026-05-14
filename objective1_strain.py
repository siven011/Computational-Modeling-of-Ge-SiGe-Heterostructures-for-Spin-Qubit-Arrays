"""
=============================================================================
SPIN QUBIT HETEROSTRUCTURE DESIGN — OBJECTIVE 1
Strain Tensor Calculation for a Ge Quantum Well on Si_{1-x}Ge_x
=============================================================================

WHAT THIS CODE DOES (overview)
-------------------------------
A Germanium quantum well (QW) is sandwiched between layers of Si_{1-x}Ge_x.
Because Ge has a larger lattice constant than SiGe (for x < 1), the Ge layer
is squeezed in the plane of the interface → biaxial COMPRESSIVE strain.

This strain is not an annoyance — it is ESSENTIAL for spin qubits.
It splits the Heavy Hole (HH) and Light Hole (LH) valence bands, which:
  • Confines hole spins to a pure HH ground state
  • Enables all-electrical spin control via spin-orbit coupling
  • Reduces decoherence from LH admixture

This script computes:
  1. The in-plane and out-of-plane strain tensors (ε_∥, ε_⊥) vs substrate Ge fraction x
  2. Supporting quantities: hydrostatic and deviatoric (biaxial) strain
  3. The Matthews-Blakeslee critical thickness h_c — the maximum QW thickness
     before strain-relieving dislocations (device killers) nucleate

REFERENCES
----------
[1] Vurgaftman et al., J. Appl. Phys. 89, 5815 (2001)        — material parameters
[2] Matthews & Blakeslee, J. Cryst. Growth 27, 118 (1974)    — critical thickness
[3] Winkler, Spin-Orbit Coupling in 2D Systems, Springer 2003 — strain + spin physics
[4] Lodari et al., npj Quantum Inf. 8, 14 (2022)             — Ge spin qubit context

HOW TO RUN
----------
  pip install numpy matplotlib scipy
  python objective1_strain.py

OUTPUT
------
  • Console: parameter table + strain summary
  • File:    objective1_strain_analysis.png
=============================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import scipy.optimize

# =============================================================================
# GLOBAL PLOT STYLE
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

# Colour palette
C = {
    'par':   '#E53935',   # red    — in-plane strain
    'perp':  '#1E88E5',   # blue   — out-of-plane strain
    'hyd':   '#8E24AA',   # purple — hydrostatic
    'bi':    '#43A047',   # green  — biaxial/deviatoric
    'hc':    '#F4511E',   # orange — critical thickness
    'safe':  '#A5D6A7',   # light green
    'unsafe':'#FFCDD2',   # light red
}


# =============================================================================
# SECTION 1 — MATERIAL PARAMETERS
# =============================================================================
#
# All parameters for pure Si and pure Ge at 300 K.
# For the QW (Ge), we need:
#   • Lattice constant (to compute the misfit with the substrate)
#   • Elastic stiffness constants C11, C12 (to relate stress ↔ strain)
#
# The elastic stiffness tensor C_ijkl for a cubic crystal like Ge or Si has
# only 3 independent components: C11, C12, C44.
# Under (001) biaxial loading, only C11 and C12 appear.
#
# Hooke's Law (Voigt notation, biaxial loading with σ_zz = 0):
#
#   σ_xx = σ_yy = C11 ε_∥ + C12 ε_∥ + C12 ε_⊥
#   σ_zz = 0    = C12 ε_∥ + C12 ε_∥ + C11 ε_⊥
#
# Solving σ_zz = 0 for ε_⊥ gives the out-of-plane strain formula (Section 3).
# =============================================================================

# ── Lattice constants (Ångström) [Ref 1] ─────────────────────────────────────
a_Si = 5.4310
a_Ge = 5.6579

# ── Ge elastic stiffness constants (GPa) [Ref 1] ─────────────────────────────
C11 = 128.53
C12 =  48.26
# C44 =  66.80   # shear modulus — not needed for Obj 1

# ── Derived elastic quantities ────────────────────────────────────────────────

# Biaxial Poisson ratio: controls ε_⊥ / ε_∥ ratio (see Section 3)
#   Derived by solving σ_zz = 0 (free surface):
#     2 C12 ε_∥ + C11 ε_⊥ = 0  →  ε_⊥ = -2(C12/C11) ε_∥
poisson_biaxial = C12 / C11           # ≈ 0.375

# Standard (isotropic) Poisson's ratio — used in Matthews-Blakeslee formula
#   ν = C12 / (C11 + C12)   (isotropic approximation)
nu = C12 / (C11 + C12)                # ≈ 0.273   doubt

# Burgers vector for 60° mixed dislocations in diamond-cubic Ge
#   b = (a/2)<110>  →  |b| = a_Ge / √2   (in metres)
b_Ge_m = (a_Ge * 1e-10) / np.sqrt(2)  # metres

# ── Print parameter summary ───────────────────────────────────────────────────
print("=" * 60)
print("MATERIAL PARAMETERS  (Ge quantum well)")
print("=" * 60)
print(f"  Lattice constant a_Ge          = {a_Ge:.4f} Å")
print(f"  Lattice constant a_Si          = {a_Si:.4f} Å")
print(f"  C11                            = {C11:.2f} GPa")
print(f"  C12                            = {C12:.2f} GPa")
print(f"  Biaxial Poisson ratio C12/C11  = {poisson_biaxial:.4f}")
print(f"  Isotropic Poisson ratio ν      = {nu:.4f}")
print(f"  Burgers vector |b|             = {b_Ge_m*1e10:.4f} Å")


# =============================================================================
# SECTION 2 — VEGARD'S LAW: SUBSTRATE LATTICE CONSTANT
# =============================================================================
#
# The virtual substrate is a RELAXED Si_{1-x}Ge_x alloy grown on Si.
# "Relaxed" means it has fully accommodated its own misfit via dislocations
# during growth — it is a stress-free reference crystal whose lattice constant
# we control by choosing x.
#
# Vegard's Law: for a random alloy, the lattice constant interpolates linearly
# between the two pure-element values as a function of composition.
#
#   a_SiGe(x) = (1-x) · a_Si  +  x · a_Ge
#
# The bowing correction δ(x) = b_bow · x(1-x) for SiGe is ~0.003 Å at x=0.5.
# This is <0.05% and negligible compared to device tolerances, so we omit it.
# =============================================================================

def a_SiGe(x):
    """
    Lattice constant of relaxed Si_{1-x}Ge_x (Vegard's Law).

    Parameters
    ----------
    x : float or ndarray
        Ge mole fraction in the substrate alloy (0 = pure Si, 1 = pure Ge).

    Returns
    -------
    float or ndarray
        Lattice constant in Ångström.
    """
    return (1.0 - x) * a_Si + x * a_Ge


# =============================================================================
# SECTION 3 — STRAIN TENSOR COMPONENTS
# =============================================================================
#
# The Ge quantum well is deposited on the SiGe substrate.
# In the pseudomorphic growth regime (below critical thickness):
#   • The QW adopts the substrate's in-plane lattice constant → biaxial strain
#   • The QW is free to relax in z → out-of-plane strain from Poisson effect
#
# ┌─────────────────────────────────────────────────────────────────────┐
# │  IN-PLANE STRAIN (ε_∥)                                              │
# │                                                                     │
# │       a_substrate − a_Ge                                            │
# │  ε_∥ = ─────────────────────                                        │
# │              a_Ge                                                   │
# │                                                                     │
# │  Since a_SiGe(x) < a_Ge for x < 1:  ε_∥ < 0  (COMPRESSIVE)          │
# └─────────────────────────────────────────────────────────────────────┘
#
# ┌─────────────────────────────────────────────────────────────────────┐
# │  OUT-OF-PLANE STRAIN (ε_⊥)                                          │
# │                                                                     │
# │  Force balance σ_zz = 0 (free surface) gives:                       │
# │                                                                     │
# │           C12                                                       │
# │  ε_⊥ = −2 ─── · ε_∥                                                 │
# │           C11                                                       │
# │                                                                     │
# │  Since ε_∥ < 0:  ε_⊥ > 0  (TENSILE — layer expands vertically)      │
# └─────────────────────────────────────────────────────────────────────┘
#
# DECOMPOSITION INTO HYDROSTATIC + DEVIATORIC PARTS:
#
#   The full strain tensor ε = ε_hydro · I  +  ε_deviatoric
#
#   Hydrostatic (isotropic, volume change):
#     ε_hyd  = (2ε_∥ + ε_⊥) / 3      ← trace / 3
#
#   Biaxial / Deviatoric (symmetry-breaking):
#     ε_bi   = ε_⊥ − ε_∥
#
#   WHY THIS MATTERS:
#     • ε_hyd  shifts ALL bands together (no splitting)
#     • ε_bi   breaks cubic → tetragonal symmetry → lifts HH/LH degeneracy
#     → ε_bi is the KEY parameter for Objective 2 (Bir-Pikus)
# =============================================================================

def epsilon_parallel(x):
    """
    In-plane biaxial strain of Ge QW on relaxed Si_{1-x}Ge_x.

    Returns negative values (compressive) for x < 1.
    """
    return (a_SiGe(x) - a_Ge) / a_Ge


def epsilon_perp(x):
    """
    Out-of-plane strain of Ge QW (Poisson response to in-plane compression).

    Derived from σ_zz = 0:  ε_⊥ = -2(C12/C11) · ε_∥
    Returns positive values (tensile) for x < 1.
    """
    return -2.0 * (C12 / C11) * epsilon_parallel(x)


def epsilon_hydrostatic(x):
    """
    Hydrostatic (isotropic) strain component.

    ε_hyd = Tr(ε) / 3 = (2·ε_∥ + ε_⊥) / 3

    Shifts all band edges uniformly. Does NOT break HH/LH degeneracy.
    Needed for the full Bir-Pikus Hamiltonian in Objective 2.
    """
    return (2.0 * epsilon_parallel(x) + epsilon_perp(x)) / 3.0


def epsilon_biaxial(x):
    """
    Biaxial (deviatoric) strain component.

    ε_bi = ε_⊥ − ε_∥

    This is the symmetry-breaking quantity that drives HH-LH splitting.
    For compressive in-plane strain: ε_bi > 0 → HH lies BELOW LH
    (HH becomes the ground state, which is what we want for spin qubits).
    """
    return epsilon_perp(x) - epsilon_parallel(x)


# =============================================================================
# SECTION 4 — MATTHEWS-BLAKESLEE CRITICAL THICKNESS
# =============================================================================
#
# PHYSICS BACKGROUND
# ------------------
# The elastic strain energy stored in a pseudomorphic film scales as:
#
#   E_strain ~ C11 · f² · h       (h = film thickness, f = misfit strain)
#
# Meanwhile, a misfit dislocation relieves strain but costs line energy:
#
#   E_dislo  ~ μb² · ln(h/b)
#
# Below h_c: elastic energy < dislocation energy → NO dislocations (safe)
# Above h_c: elastic energy > dislocation energy → dislocations nucleate
#
# THE IMPLICIT EQUATION (Matthews & Blakeslee, 1974)
# ---------------------------------------------------
# For 60° mixed dislocations on {111} slip planes in (001)-grown cubic films:
#
#         b · (1 − ν cos²θ)
#   h_c = ─────────────────────  ×  [ln(h_c / b) + 1]
#         8π · f · (1+ν) · cosλ
#
# where:
#   b = a_Ge/√2         Burgers vector length (|b| for a/2⟨110⟩ dislocation)
#   f = |ε_∥|           misfit strain magnitude
#   ν = C12/(C11+C12)   isotropic Poisson's ratio
#   θ = 60°             angle between Burgers vector and dislocation line
#   λ = 60°             angle between slip direction and interface trace
#
# This is IMPLICIT because h_c appears on both sides → solve numerically.
#
# PRACTICAL SIGNIFICANCE FOR SPIN QUBITS
# ---------------------------------------
# Typical Ge QW thickness: 8–20 nm
# Typical substrate:       x ~ 0.70–0.90
# This function tells us which (x, thickness) combinations are safe.
# =============================================================================

def critical_thickness_MB(x):
    """
    Matthews-Blakeslee critical thickness h_c for Ge on Si_{1-x}Ge_x.

    Solves the implicit MB equation self-consistently using Brent's method.

    Parameters
    ----------
    x : float
        Ge fraction in virtual substrate.

    Returns
    -------
    float
        Critical thickness h_c in nanometres.
        Returns np.inf if x ≈ 1 (zero misfit).
    """
    f = abs(epsilon_parallel(x))
    if f < 1e-9:
        return np.inf

    # Dislocation geometry for 60° mixed dislocation
    theta = np.radians(60.0)   # angle between Burgers vector & dislocation line
    lam   = np.radians(60.0)   # angle between slip direction & interface trace

    # Geometric prefactor (dimensionless)
    geo_factor = (1.0 - nu * np.cos(theta)**2) / ((1.0 + nu) * np.cos(lam))

    # Overall prefactor A (in metres):  h_c = A · [ln(h_c / b) + 1]
    A = (b_Ge_m * geo_factor) / (8.0 * np.pi * f)

    # Rearranged implicit equation: g(h) = h - A·[ln(h/b) + 1] = 0
    def g(h):
        return h - A * (np.log(h / b_Ge_m) + 1.0)

    # Brent's method: bracket [b, 10 µm] covers all physically relevant h_c
    try:
        h_c_metres = scipy.optimize.brentq(g, b_Ge_m * 1.001, 1.0e-5, xtol=1e-18)
        return h_c_metres * 1.0e9   # metres → nanometres
    except ValueError:
        return np.nan


# Vectorise for array inputs
critical_thickness_MB_vec = np.vectorize(critical_thickness_MB)


# =============================================================================
# SECTION 5 — COMPUTE RESULTS
# =============================================================================

x_vals = np.linspace(0.01, 0.995, 600)   # Ge fraction sweep

eps_par = epsilon_parallel(x_vals)
eps_per = epsilon_perp(x_vals)
eps_hyd = epsilon_hydrostatic(x_vals)
eps_bi  = epsilon_biaxial(x_vals)
a_sub   = a_SiGe(x_vals)
h_c     = critical_thickness_MB_vec(x_vals)

# ── Console summary table ─────────────────────────────────────────────────────
print("\n" + "=" * 72)
print("  STRAIN SUMMARY  —  Ge QW on relaxed Si_{1-x}Ge_x")
print("=" * 72)
header = f"  {'x':>5}  {'a_sub (Å)':>10}  {'ε_∥ (%)':>8}  {'ε_⊥ (%)':>8}  {'ε_bi (%)':>9}  {'h_c (nm)':>9}"
print(header)
print("  " + "-" * 68)
for xv in [0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]:
    print(f"  {xv:5.2f}  {a_SiGe(xv):10.4f}  "
          f"{epsilon_parallel(xv)*100:8.3f}  "
          f"{epsilon_perp(xv)*100:8.3f}  "
          f"{epsilon_biaxial(xv)*100:9.3f}  "
          f"{critical_thickness_MB(xv):9.2f}")
print()


# =============================================================================
# SECTION 6 — FIGURES
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

# ── PLOT 1: Vegard's Law ──────────────────────────────────────────────────────
ax1.plot(x_vals, a_sub, color='#1565C0', lw=2.5, label="$a_{\\mathrm{Si}_{1-x}\\mathrm{Ge}_x}$ (Vegard)")
ax1.axhline(a_Ge, ls='--', color=C['par'], lw=1.3, label=f"$a_{{\\mathrm{{Ge}}}}$ = {a_Ge} Å")
ax1.axhline(a_Si, ls='--', color='#555',  lw=1.3, label=f"$a_{{\\mathrm{{Si}}}}$ = {a_Si} Å")

# Shade the mismatch region at x=0.8 as a visual
x_ref = 0.80
ax1.annotate('', xy=(x_ref, a_SiGe(x_ref)), xytext=(x_ref, a_Ge),
             arrowprops=dict(arrowstyle='<->', color='#E53935', lw=1.5))
ax1.text(x_ref + 0.02, (a_SiGe(x_ref) + a_Ge)/2,
         'misfit\n$\\Delta a$', color='#E53935', fontsize=8.5, va='center')

ax1.set_xlabel('Ge fraction  $x$  in substrate')
ax1.set_ylabel('Lattice constant  (Å)')
ax1.set_title("(1) Vegard's Law — Substrate $a(x)$")
ax1.legend(loc='upper left')
ax1.set_xlim(0, 1)
ax1.set_ylim(5.39, 5.70)

# ── PLOT 2: Strain tensor components ─────────────────────────────────────────
ax2.plot(x_vals, eps_par * 100, color=C['par'],  lw=2.5, label=r'$\epsilon_\parallel$  in-plane (compressive)')
ax2.plot(x_vals, eps_per * 100, color=C['perp'], lw=2.5, label=r'$\epsilon_\perp$  out-of-plane (tensile)')
ax2.plot(x_vals, eps_hyd * 100, color=C['hyd'],  lw=1.5, ls=(0,(4,2)), label=r'$\epsilon_\mathrm{hyd}$  hydrostatic (shifts bands)')
ax2.plot(x_vals, eps_bi  * 100, color=C['bi'],   lw=1.5, ls=(0,(4,2)), label=r'$\epsilon_\mathrm{bi}$  biaxial → drives HH-LH split')

ax2.axhline(0, color='k', lw=0.8, zorder=0)
ax2.fill_between(x_vals, eps_par * 100, 0, alpha=0.08, color=C['par'])
ax2.fill_between(x_vals, eps_per * 100, 0, alpha=0.08, color=C['perp'])

# Annotate at x = 0.8
xm = 0.80
ax2.annotate(f"$x=0.80$\n$\\epsilon_\\parallel={epsilon_parallel(xm)*100:.2f}\\%$",
             xy=(xm, epsilon_parallel(xm)*100),
             xytext=(0.55, -1.3),
             fontsize=8.5, color=C['par'],
             arrowprops=dict(arrowstyle='->', color=C['par'], lw=1))

ax2.set_xlabel('Ge fraction  $x$  in substrate')
ax2.set_ylabel('Strain  (%)')
ax2.set_title("(2) Strain Tensor Components in Ge QW")
ax2.legend(loc='lower right', framealpha=0.9)
ax2.set_xlim(0, 1)

# ── PLOT 3: Critical thickness (log scale) ────────────────────────────────────
h_c_plot = np.where(h_c > 1000, 1000, h_c)

ax3.semilogy(x_vals, h_c, color=C['hc'], lw=2.5, label='MB critical thickness $h_c$')

# Mark typical QW thicknesses used in spin qubit experiments
for thickness, ls, label in [(8, ':', '8 nm'), (14, '--', '14 nm'), (20, '-', '20 nm')]:
    ax3.axhline(thickness, color='#37474F', lw=1.2, ls=ls, label=f'{label} QW')

# Shade safe vs unsafe for a 14 nm QW
x_safe_14 = x_vals[h_c > 14]
if len(x_safe_14) > 0:
    x_boundary_14 = x_safe_14[0]
    ax3.axvline(x_boundary_14, color='#37474F', lw=1.0, ls=':', alpha=0.5)
    ax3.text(x_boundary_14 + 0.01, 2.5, f'$x_{{c}}$≈{x_boundary_14:.2f}\nfor 14 nm',
             fontsize=8, color='#37474F')

ax3.set_xlabel('Ge fraction  $x$  in substrate')
ax3.set_ylabel('Critical thickness  (nm, log)')
ax3.set_title("(3) Matthews-Blakeslee Critical Thickness")
ax3.legend(loc='upper left')
ax3.set_xlim(0, 1)
ax3.set_ylim(0.8, 3000)

# ── PLOT 4: 2D Process Window (x vs QW thickness) ────────────────────────────
#
# For each substrate composition x and QW thickness t,
# the combination is SAFE if  t < h_c(x)  (pseudomorphic growth).
# This 2D map is the key design tool for Objective 1.
#
x_2d = np.linspace(0.40, 0.995, 400)
t_2d = np.linspace(1, 60, 400)          # QW thickness in nm
XX, TT = np.meshgrid(x_2d, t_2d)

h_c_2d = critical_thickness_MB_vec(x_2d)      # shape (400,)
safe_mask = TT < h_c_2d[np.newaxis, :]        # broadcast: shape (400, 400)

ax4.contourf(XX, TT, safe_mask.astype(float),
             levels=[-0.5, 0.5, 1.5],
             colors=[C['unsafe'], C['safe']], alpha=0.75)
ax4.contour(XX, TT, safe_mask.astype(float),
            levels=[0.5], colors=['#212121'], linewidths=[2.0])

# Overlay the MB boundary as a line
ax4.plot(x_2d, h_c_2d, color='#212121', lw=2.0, label='MB boundary $h_c(x)$')

# Mark the typical spin qubit operating window (literature range)
qubit_x_lo, qubit_x_hi   = 0.75, 0.90
qubit_t_lo, qubit_t_hi   = 8,    20
ax4.add_patch(mpatches.FancyBboxPatch(
    (qubit_x_lo, qubit_t_lo),
    qubit_x_hi - qubit_x_lo,
    qubit_t_hi - qubit_t_lo,
    boxstyle="round,pad=0.5",
    linewidth=2, edgecolor='#0D47A1', facecolor='none',
    linestyle='--', zorder=5))
ax4.text(0.825, 14.5, 'Typical\nspin qubit\nwindow',
         ha='center', va='center', fontsize=8.5, color='#0D47A1', fontweight='bold')

ax4.text(0.58, 45, 'DISLOCATIONS\n(unsafe)',
         ha='center', va='center', color='#B71C1C', fontsize=9, fontweight='bold')
ax4.text(0.88, 5, 'PSEUDOMORPHIC\n(safe)',
         ha='center', va='center', color='#1B5E20', fontsize=9, fontweight='bold')

ax4.set_xlabel('Ge fraction  $x$  in substrate')
ax4.set_ylabel('Ge QW thickness  (nm)')
ax4.set_title("(4) Epitaxial Safety Map  (Process Window)")
ax4.legend(loc='upper right', framealpha=0.9)
ax4.set_xlim(0.40, 1.0)
ax4.set_ylim(1, 60)

# ── Global title ──────────────────────────────────────────────────────────────
fig.suptitle(
    'Strain Tensor Analysis — Ge QW on Si$_{1-x}$Ge$_x$ Virtual Substrate',
    fontsize=13, fontweight='bold', y=0.965
)

out_path = 'objective1_strain_analysis.png'
plt.savefig(out_path, bbox_inches='tight', facecolor=fig.get_facecolor())
plt.close()
print(f"Figure saved → {out_path}")
print()

# =============================================================================
# SECTION 7 — ANALYTICAL SANITY CHECKS
# =============================================================================
print("=" * 60)
print("SANITY CHECKS")
print("=" * 60)

# At x=1: pure Ge on pure Ge → zero strain (should be 0)
print(f"\n  x=1.00 (no misfit):  ε_∥ = {epsilon_parallel(1.0)*100:.6f} %  ✓ (expect 0)")

# At x=0: pure Ge on pure Si → maximum strain
max_strain = epsilon_parallel(0.0)
print(f"  x=0.00 (max misfit): ε_∥ = {max_strain*100:.3f} %")

# Poisson ratio check: ε_⊥ / ε_∥ = -2(C12/C11)
x_check = 0.80
ratio = epsilon_perp(x_check) / epsilon_parallel(x_check)
print(f"\n  ε_⊥/ε_∥ at x=0.80: {ratio:.4f}  (expect {-2*C12/C11:.4f})")

# Volume conservation (approximately): Δv/v = ε_hyd * 3
hyd_at_80 = epsilon_hydrostatic(x_check)
print(f"  ε_hyd at x=0.80:    {hyd_at_80*100:.4f} %")
print(f"  → Volume change ΔV/V ≈ {3*hyd_at_80*100:.4f} %")

# Critical thickness limits
print(f"\n  h_c at x=0.75: {critical_thickness_MB(0.75):.1f} nm")
print(f"  h_c at x=0.80: {critical_thickness_MB(0.80):.1f} nm")
print(f"  h_c at x=0.85: {critical_thickness_MB(0.85):.1f} nm")
print(f"  h_c at x=0.90: {critical_thickness_MB(0.90):.1f} nm")

print("\n All Objective 1 calculations complete.")


# Need to understand bowing parameter, MB equation
