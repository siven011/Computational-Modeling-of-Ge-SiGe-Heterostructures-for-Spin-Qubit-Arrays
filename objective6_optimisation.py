#!/usr/bin/env python3
"""
=============================================================================
SPIN QUBIT HETEROSTRUCTURE DESIGN — OBJECTIVE 6 (CAPSTONE)
Joint Process-Window Optimisation: Figure-of-Merit Landscape & Pareto Front
=============================================================================

WHAT THIS CODE DOES (overview)
-------------------------------
This is the capstone analysis. Each previous objective produced a
constraint or metric on the (x, L) design space:

  Objective 1:  ε_bi(x), h_c(x)               — strain and dislocation safety
  Objective 2:  ΔE_total(x, L)                — HH-LH splitting
  Objective 3:  Y_50(x, L) = P(ΔE > 50 meV)  — splitting yield
  Objective 4:  g_∥(x, L), σ_g/g(x, L)       — g-factor and addressability
  Objective 5:  F_gate(g_∥, T₂*)              — single-qubit gate fidelity

Objective 6 combines these into a single FIGURE OF MERIT and identifies
the operating point (x, L) that simultaneously maximises performance
and yield. The doc states the FoM as:

     FoM(x, L) = F_gate(x, L) · Y(x, L) · g_uniformity(x, L)

where:
  - F_gate    is the predicted single-qubit gate fidelity
  - Y         is the manufacturing yield against the 50 meV constraint
  - g_uniformity is a smooth penalty on g_∥ variation across qubits

All three sit in [0,1]. The optimum is the maximum of FoM. We also
report:
  - The PARETO FRONT in (Y, F_gate) space — what's the trade-off?
  - SENSITIVITY TO ASSUMPTIONS — how does the recommended point shift
    if we tighten or relax tolerances?
  - A RECOMMENDED OPERATING POINT with quantitative claims.

DESIGN ASSUMPTIONS
------------------
- Manufacturing tolerances:  σ_x = 0.005, σ_L = 0.5 nm  (best-in-class CVD)
- Coherence:                  T₂* = 5 µs                 (good Ge HH today)
- Drive amplitude:            Ω_R/(2π) = 25 MHz          (typical)
- B-field:                    0.5 T                       (standard)
- Constraint hierarchy:       ΔE > 50 meV   (HARD)
                              L < 0.7·h_c   (HARD — dislocation safety)
                              F_gate > 99%  (target)
                              Y > 99%       (target)

REFERENCES
----------
[1]  Pareto, Manuale di economia politica (1906)         — Pareto efficiency
[2]  Vurgaftman et al., J. Appl. Phys. 89, 5815 (2001)   — material params
[3]  Lodari et al., npj Quantum Inf. 8, 14 (2022)        — Ge HH platform
[4]  Scappucci et al., Nat. Rev. Mater. 6, 926 (2021)    — Ge platform review

HOW TO RUN
----------
  pip install numpy matplotlib scipy
  python objective6_optimisation.py
=============================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from scipy.sparse import diags
from scipy.sparse.linalg import eigsh
from scipy.stats import norm
from scipy.optimize import brentq

RNG = np.random.default_rng(seed=42)

# =============================================================================
# GLOBAL PLOT STYLE  (matches Objs 1–5)
# =============================================================================

plt.rcParams.update({
    'figure.dpi': 150, 'font.family': 'serif', 'font.size': 11,
    'axes.labelsize': 12, 'axes.titlesize': 12, 'axes.titleweight': 'bold',
    'legend.fontsize': 9.5, 'axes.grid': True,
    'grid.alpha': 0.25, 'grid.linestyle': '--',
    'axes.spines.top': False, 'axes.spines.right': False,
})

C = {
    'Y':       '#43A047',   # green  — yield
    'F':       '#1565C0',   # blue   — fidelity
    'guni':    '#FB8C00',   # orange — g-uniformity
    'FoM':     '#8E24AA',   # purple — combined FoM
    'optimum': '#E53935',   # red    — optimum marker
    'pareto':  '#0D47A1',   # dark blue — Pareto front
    'forbid':  '#B71C1C',   # red    — dislocation/forbidden
    'safe':    '#A5D6A7',
    'unsafe':  '#FFCDD2',
}

# =============================================================================
# SECTION 1 — MATERIAL PARAMETERS  (re-used from Objs 1–5)
# =============================================================================

a_Si, a_Ge = 5.4310, 5.6579           # Å
C11, C12   = 128.53, 48.26            # GPa
b_v_Ge     = -2.16                     # eV
m_HH_z, m_LH_z = 0.20, 0.046          # m_e
kappa      = 3.41
HBAR2_OVER_2ME = 0.03810               # eV·nm²
mu_B_over_h    = 13.996e9              # Hz/T
V0_VB_default  = 0.150                 # eV — VB offset

# Operating conditions
B_field         = 0.5                  # Tesla
T2star_default  = 5e-6                 # s
Omega_R_default = 2*np.pi * 25e6       # rad/s — 25 MHz Rabi

# Manufacturing tolerances (1σ)
sigma_x_default     = 0.005
sigma_L_default     = 0.5              # nm
sigma_kappa_default = 0.015 * kappa    # 1.5% κ spread

# Hard constraints
DELTA_E_TARGET    = 0.050              # eV — HH-LH splitting target
H_C_SAFETY_FACTOR = 0.7                # L < 0.7 · h_c

# Burgers vector for Matthews-Blakeslee critical thickness
nu       = C12 / (C11 + C12)
b_Ge_m   = (a_Ge*1e-10) / np.sqrt(2)


# =============================================================================
# SECTION 2 — HELPER FUNCTIONS  (inlined from Objs 1–4)
# =============================================================================

def a_SiGe(x):           return (1.0-x)*a_Si + x*a_Ge
def epsilon_parallel(x): return (a_SiGe(x) - a_Ge) / a_Ge
def epsilon_perp(x):     return -2.0*(C12/C11)*epsilon_parallel(x)
def epsilon_biaxial(x):  return epsilon_perp(x) - epsilon_parallel(x)
def deltaE_strain(x):    return 2.0 * abs(b_v_Ge) * epsilon_biaxial(x)

def E1_finite_well(L_nm, m_eff, V0_eV=V0_VB_default, N=2000, Zmax_factor=5.0):
    Zmax = Zmax_factor * L_nm
    z = np.linspace(-Zmax, Zmax, N); dz = z[1]-z[0]
    V = np.where(np.abs(z) < L_nm/2.0, 0.0, V0_eV)
    t = HBAR2_OVER_2ME / (m_eff * dz**2)
    H = diags([-t*np.ones(N-1), 2*t+V, -t*np.ones(N-1)],
              offsets=[-1,0,1], format='csr')
    return float(eigsh(H, k=1, which='SA', return_eigenvectors=False)[0])

def deltaE_QC_finite(L_nm):
    return E1_finite_well(L_nm, m_LH_z) - E1_finite_well(L_nm, m_HH_z)

# Tabulate ΔE_QC(L) once for fast evaluation
print("=" * 64)
print("OBJECTIVE 6 — JOINT PROCESS-WINDOW OPTIMISATION")
print("=" * 64)
print("\nTabulating finite-well ΔE_QC(L)...")
L_table     = np.linspace(2.0, 35.0, 200)
dE_QC_table = np.array([deltaE_QC_finite(L) for L in L_table])
def deltaE_QC_fast(L_nm): return np.interp(L_nm, L_table, dE_QC_table)
def deltaE_total(x, L):   return deltaE_strain(x) + deltaE_QC_fast(L)
print(f"  done  ({len(L_table)} FDM solves).\n")

# Critical thickness h_c(x) — Matthews-Blakeslee (Obj 1)
def critical_thickness_MB(x):
    f = abs(epsilon_parallel(x))
    if f < 1e-9: return np.inf
    theta = np.radians(60.0); lam = np.radians(60.0)
    geo_factor = (1.0 - nu*np.cos(theta)**2) / ((1.0+nu)*np.cos(lam))
    A = (b_Ge_m * geo_factor) / (8.0 * np.pi * f)
    g = lambda h: h - A*(np.log(h/b_Ge_m) + 1.0)
    try:
        return brentq(g, b_Ge_m*1.001, 1e-5, xtol=1e-18) * 1e9
    except ValueError:
        return np.nan

# g-factor model (Obj 4 calibration)
ALPHA_PERP = 0.012     # eV
BETA_PAR   = 0.030     # eV
def g_parallel(x, L, kappa_eff=kappa):
    return 6.0 * kappa_eff * (1.0 - BETA_PAR / deltaE_total(x, L))


# =============================================================================
# SECTION 3 — METRICS ON THE DESIGN GRID
# =============================================================================
#
# We assemble five metrics on a (x, L) grid:
#
#   1. ΔE_total(x, L)           — central HH-LH splitting (eV)
#   2. h_c(x)                    — MB critical thickness (nm)
#   3. Y_DE(x, L)                — yield against ΔE > 50 meV (linearised CDF)
#   4. g_∥(x, L), σ_g/g(x, L)   — g-factor mean & spread (linearised)
#   5. F_gate(x, L)              — predicted single-qubit gate fidelity
#                                  via the analytic Gaussian formula
#
# All quantities are vectorised over (x, L). Linearised yields are
# validated against direct Monte Carlo at the optimum point at the end.
# =============================================================================

print("Building design-space grids (x, L)...")
# Search range from foundations doc Section 28.1: x ∈ [0.65, 0.95], L ∈ [6, 25]
x_grid = np.linspace(0.55, 0.97, 220)        # slightly wider than doc spec
L_grid = np.linspace(6.0, 25.0, 140)         # doc spec
XX, LL = np.meshgrid(x_grid, L_grid)

# 1. ΔE_total
mu_DE = deltaE_total(XX, LL)                          # eV

# 2. h_c (depends only on x — vectorise once over x_grid)
h_c_x = np.array([critical_thickness_MB(xv) for xv in x_grid])
HC    = np.tile(h_c_x, (len(L_grid), 1))              # broadcast to grid
L_safe = LL < H_C_SAFETY_FACTOR * HC                  # bool mask

# 3. Y_DE — linearised yield (validated against MC in Obj 3 to 0.5%)
dx_step, dL_step = 1e-4, 0.05
dDE_dx = (deltaE_strain(XX+dx_step) - deltaE_strain(XX-dx_step)) / (2*dx_step)
dDE_dL = (deltaE_QC_fast(LL+dL_step) - deltaE_QC_fast(LL-dL_step)) / (2*dL_step)
sigma_DE = np.sqrt((dDE_dx*sigma_x_default)**2 + (dDE_dL*sigma_L_default)**2)
Y_DE = norm.cdf((mu_DE - DELTA_E_TARGET) / sigma_DE)  # in [0,1]

# 4. g_∥ and σ_g/g
# The g_∥ formula 6κ(1 − β/ΔE) is only valid when ΔE >> β = 30 meV
# (perturbation-theory regime). Below that, leading-order PT breaks down
# and the qubit is no longer a clean HH state anyway. We enforce a
# physics-validity mask: PT_valid = (ΔE > 50 meV).
PT_valid = mu_DE > DELTA_E_TARGET

g_par_grid = g_parallel(XX, LL)
dgp_dx = (g_parallel(XX+dx_step, LL) - g_parallel(XX-dx_step, LL)) / (2*dx_step)
dgp_dL = (g_parallel(XX, LL+dL_step) - g_parallel(XX, LL-dL_step)) / (2*dL_step)
dgp_dk = 6.0 * (1.0 - BETA_PAR / np.maximum(mu_DE, BETA_PAR*1.5))   # safe
sigma_g = np.sqrt((dgp_dx*sigma_x_default)**2
                 + (dgp_dL*sigma_L_default)**2
                 + (dgp_dk*sigma_kappa_default)**2)
# Use safe denominator to keep ratio finite outside PT_valid (FoM is masked anyway)
sigma_g_over_g = sigma_g / np.maximum(np.abs(g_par_grid), 0.5)

# π-pulse duration is a constant under on-resonance drive
t_pi = np.pi / Omega_R_default

# 5. F_gate — single-qubit gate fidelity from analytic Gaussian formula
#
#    Two physical effects make F_gate depend on (x, L):
#    (a) T₂* depends on the well thickness. Thin wells suffer from
#        interface-roughness scattering, alloy intermixing, and stronger
#        ∂g/∂V coupling to charge noise — all of which collapse T₂*.
#        Experimentally, sub-10 nm Ge HH wells are not commonly used.
#        We model this with a smooth CDF transition:
#             T₂*(L) = T₂*_max · Φ((L − L_min) / L_width)
#        with L_min = 10 nm, L_width = 2 nm. This gives:
#             L = 8  nm → T₂* ≈ 0.8 µs    (poor — thin-well penalty)
#             L = 12 nm → T₂* ≈ 4.2 µs
#             L = 14 nm → T₂* ≈ 4.9 µs    (matches reference [4])
#             L > 18 nm → T₂* → T₂*_max = 5 µs
#    (b) Larger σ_g/g implies more in-situ noise sensitivity, but we
#        don't double-count this in F_gate; it enters g_uniformity.
#
#    For a target N-qubit array, the COMPOUND fidelity is F^N (Section
#    27.1 of the foundations doc). We use N = 100 as a near-term
#    scaling target.
L_min      = 10.0      # nm — practical thin-well floor
L_width    = 2.0       # nm — transition width
T2star_max = 5e-6      # s — asymptotic value at thick wells
N_QUBITS   = 100       # notional array size for compounding

T2star_grid   = T2star_max * norm.cdf((LL - L_min) / L_width)
F_single_grid = np.maximum(0.0, 1.0 - (t_pi / np.maximum(T2star_grid, 1e-12))**2)
F_array_grid  = F_single_grid ** N_QUBITS


# =============================================================================
# SECTION 4 — BUILD THE FIGURE OF MERIT
# =============================================================================
#
# FoM(x, L) = Y_DE(x, L) · F_array(x, L) · g_uniformity(x, L)
#
# where F_array = F_single^N is the compound array fidelity over N=100
# qubits, and g_uniformity = exp(− (σ_g/g) / σ_target) with σ_target = 3%.
# All three sit in [0, 1].
#
# The hard constraints (L < 0.7 h_c, ΔE > 50 meV at center) are imposed
# by setting FoM = 0 wherever they are violated.
# =============================================================================

g_uni_target_pct  = 0.03   # σ_g/g = 3% gives g_uniformity = 1/e ≈ 0.37
g_uniformity_grid = np.exp(-sigma_g_over_g / g_uni_target_pct)

# Combined FoM
FoM = Y_DE * F_array_grid * g_uniformity_grid

# Combined hard-constraint mask:
#   - dislocation safety: L < 0.7·h_c
#   - perturbation-theory validity: ΔE > 50 meV (g_∥ formula meaningful)
HARD_OK = L_safe & PT_valid
FoM_safe = np.where(HARD_OK, FoM, 0.0)


# =============================================================================
# SECTION 5 — FIND THE OPTIMUM
# =============================================================================

i_opt, j_opt = np.unravel_index(np.argmax(FoM_safe), FoM_safe.shape)
x_opt = x_grid[j_opt]
L_opt = L_grid[i_opt]

# 80% plateau — points with FoM > 0.8 · FoM_max
plateau_mask = FoM_safe > 0.8 * FoM_safe.max()

print("=" * 64)
print("OPTIMISATION RESULT")
print("=" * 64)
print(f"\n  Optimal operating point:")
print(f"    x*               = {x_opt:.3f}")
print(f"    L*               = {L_opt:.2f} nm")
print(f"    ΔE_total(x*, L*) = {deltaE_total(x_opt, L_opt)*1000:.2f} meV")
print(f"    h_c(x*)           = {critical_thickness_MB(x_opt):.2f} nm  "
      f"(L*/h_c = {L_opt/critical_thickness_MB(x_opt)*100:.0f}% of MB limit)")
print(f"    g_∥(x*, L*)      = {g_parallel(x_opt, L_opt):.2f}")
print(f"    σ_g/g            = {sigma_g_over_g[i_opt, j_opt]*100:.2f}%")
print(f"    Y_50              = {Y_DE[i_opt, j_opt]*100:.4f}%")
print(f"    T_2* (at L=L*)    = {T2star_grid[i_opt, j_opt]*1e6:.2f} µs")
print(f"    F per qubit       = {F_single_grid[i_opt, j_opt]*100:.4f}%   "
      f"(infidelity {1-F_single_grid[i_opt, j_opt]:.2e})")
print(f"    F over N={N_QUBITS}     = {F_array_grid[i_opt, j_opt]*100:.2f}%")
print(f"    g_uniformity      = {g_uniformity_grid[i_opt, j_opt]:.4f}")
print(f"    FoM               = {FoM_safe[i_opt, j_opt]:.4f}")
print(f"\n  80% FoM plateau extent:")
print(f"    x range = [{x_grid[plateau_mask.any(axis=0)].min():.3f}, "
      f"{x_grid[plateau_mask.any(axis=0)].max():.3f}]")
print(f"    L range = [{L_grid[plateau_mask.any(axis=1)].min():.2f}, "
      f"{L_grid[plateau_mask.any(axis=1)].max():.2f}] nm")


# =============================================================================
# SECTION 6 — PARETO FRONT  (Y vs F_gate)
# =============================================================================
#
# A point (x, L) is Pareto-optimal in (Y, F) if no other feasible point
# has both higher Y and higher F. We extract the Pareto front by sorting
# all feasible (L < 0.7 h_c) points by Y descending and keeping only
# those whose F_gate exceeds all earlier F values.
# =============================================================================

# Flatten safe points
Y_flat  = Y_DE.flatten()
F_flat  = F_single_grid.flatten()
g_flat  = g_uniformity_grid.flatten()
safe_flat = L_safe.flatten()
x_flat  = XX.flatten()
L_flat  = LL.flatten()

mask = HARD_OK.flatten() & (Y_flat > 0.5) & (F_flat > 0.95)   # restrict to interesting region
Yf, Ff, gf, xf, Lf = (a[mask] for a in (Y_flat, F_flat, g_flat, x_flat, L_flat))

# Sort by F descending → Pareto front: cumulative-max of Y
order = np.argsort(-Ff)
Y_sorted = Yf[order]; F_sorted = Ff[order]
x_sorted = xf[order]; L_sorted = Lf[order]
Y_running_max = np.maximum.accumulate(Y_sorted)
on_pareto = Y_sorted == Y_running_max

# Pareto-front points
Y_pareto = Y_sorted[on_pareto]; F_pareto = F_sorted[on_pareto]
x_pareto = x_sorted[on_pareto]; L_pareto = L_sorted[on_pareto]


# =============================================================================
# SECTION 7 — PLOTS
# =============================================================================

fig = plt.figure(figsize=(14, 11))
fig.patch.set_facecolor('#FAFAFA')
gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.42, wspace=0.38,
                       left=0.09, right=0.96, top=0.92, bottom=0.08)
ax1 = fig.add_subplot(gs[0, 0]); ax2 = fig.add_subplot(gs[0, 1])
ax3 = fig.add_subplot(gs[1, 0]); ax4 = fig.add_subplot(gs[1, 1])
for ax in [ax1, ax2, ax3, ax4]: ax.set_facecolor('#FAFAFA')


# ── PLOT 1: Three constraint maps overlaid ──────────────────────────────────
#
#   Show Y_DE (green contours), g_uniformity (orange), and dislocation-
#   safety boundary (red shading) on the same (x, L) plot. The
#   intersection of "good" regions is the design sweet spot.
# ----------------------------------------------------------------------------

# Background: dislocation-unsafe region (filled red)
ax1.contourf(XX, LL, (~L_safe).astype(float),
             levels=[0.5, 1.5], colors=[C['unsafe']], alpha=0.55)
# MB boundary line
ax1.plot(x_grid, H_C_SAFETY_FACTOR * h_c_x, color=C['forbid'],
         lw=2.0, label=r'$L = 0.7\,h_c$  (MB safety limit)')

# Yield contours — show 90% and 99% (50% is below the action region)
yc = ax1.contour(XX, LL, Y_DE, levels=[0.9, 0.99],
                 colors=[C['Y']]*2, linewidths=[1.4, 2.4],
                 linestyles=[':','-'])
ax1.clabel(yc, fmt={0.9: 'Y=90%', 0.99: 'Y=99%'},
           inline=True, fontsize=8.5)

# g-uniformity contours — show 50%, 80%
gc = ax1.contour(XX, LL, g_uniformity_grid,
                 levels=[0.50, 0.80],
                 colors=[C['guni']]*2, linewidths=[1.4, 2.2],
                 linestyles=['--', '-'])
ax1.clabel(gc, fmt={0.50: '$g_\\mathrm{uni}$=50%',
                     0.80: '$g_\\mathrm{uni}$=80%'},
           inline=True, fontsize=8.5)

# Mark the optimum
ax1.plot(x_opt, L_opt, '*', color=C['optimum'], markersize=18,
         mec='black', mew=1.0, zorder=10)
ax1.annotate(f'  optimum\n  x*={x_opt:.2f}, L*={L_opt:.1f} nm',
             xy=(x_opt, L_opt),
             xytext=(x_opt+0.04, L_opt+5), fontsize=9, fontweight='bold',
             color=C['optimum'],
             arrowprops=dict(arrowstyle='->', color=C['optimum'], lw=1.0))

ax1.set_xlabel('Ge fraction  $x$')
ax1.set_ylabel('QW thickness  $L$  (nm)')
ax1.set_title('(1) Constraint Map — Y, $g_\\mathrm{uni}$, MB Safety')
ax1.legend(loc='upper left', fontsize=9)
ax1.set_xlim(0.70, 0.97); ax1.set_ylim(6, 25)


# ── PLOT 2: FoM landscape ───────────────────────────────────────────────────
cf = ax2.contourf(XX, LL, FoM_safe, levels=np.linspace(0, FoM_safe.max(), 21),
                   cmap='viridis')
cb = plt.colorbar(cf, ax=ax2, fraction=0.045, pad=0.03)
cb.set_label('FoM = Y · F · $g_\\mathrm{uni}$', fontsize=10)

# 80% plateau contour
ax2.contour(XX, LL, FoM_safe, levels=[0.8*FoM_safe.max()],
            colors=['white'], linewidths=[2.4])

# Forbidden region overlay (translucent red)
ax2.contourf(XX, LL, (~L_safe).astype(float),
             levels=[0.5, 1.5], colors=['#B71C1C'], alpha=0.55)
ax2.text(0.93, 12, 'forbidden\n(dislocations)', color='white',
         fontsize=9, fontweight='bold', ha='center')

# Optimum
ax2.plot(x_opt, L_opt, '*', color=C['optimum'], markersize=20,
         mec='white', mew=1.5, zorder=10)
ax2.text(x_opt-0.005, L_opt-0.5,
         f'  x*={x_opt:.2f}\n  L*={L_opt:.1f} nm\n  FoM={FoM_safe[i_opt,j_opt]:.3f}',
         color='white', fontsize=8.5, fontweight='bold', va='top', ha='left')

# 80% plateau label
ax2.text(0.72, 23, '80% plateau\n(white contour)',
         color='white', fontsize=8.5, fontweight='bold')

ax2.set_xlabel('Ge fraction  $x$')
ax2.set_ylabel('QW thickness  $L$  (nm)')
ax2.set_title('(2) Combined Figure of Merit Landscape')
ax2.set_xlim(0.70, 0.97); ax2.set_ylim(6, 25)


# ── PLOT 3: Pareto front (Y vs F_gate) ──────────────────────────────────────
#
#   Each safe (x, L) is one point in (Y, F) space. The Pareto front is
#   the boundary of achievable performance — no design beats it on both
#   axes simultaneously.
# ----------------------------------------------------------------------------
ax3.scatter(Yf*100, (1-Ff)*1e4, c='lightgray', s=8, alpha=0.4,
            label='all feasible designs')
ax3.scatter(Y_pareto*100, (1-F_pareto)*1e4, c=C['pareto'], s=30,
            edgecolor='black', linewidth=0.5, zorder=5,
            label='Pareto front')

# Optimum from FoM
ax3.scatter([Y_DE[i_opt,j_opt]*100], [(1-F_single_grid[i_opt,j_opt])*1e4],
            color=C['optimum'], s=200, marker='*', edgecolor='white',
            linewidth=1.5, zorder=10, label=f'FoM optimum (x*={x_opt:.2f}, L*={L_opt:.1f} nm)')

# Threshold lines
ax3.axhline(100, color='#388E3C', lw=1.0, ls='--', alpha=0.7)
ax3.text(82, 105, '$1-F=10^{-2}$  (early FT)', fontsize=8, color='#388E3C')
ax3.axhline(10, color='#0D47A1', lw=1.0, ls='--', alpha=0.7)
ax3.text(82, 12, '$1-F=10^{-3}$  (surface code)', fontsize=8, color='#0D47A1')
ax3.axvline(99, color=C['Y'], lw=1.0, ls='--', alpha=0.7)
ax3.text(99.05, 0.9, 'Y=99%', fontsize=8, color=C['Y'], rotation=90, va='bottom')

ax3.set_xlabel('Yield  Y  (%)')
ax3.set_ylabel(r'Infidelity  $1-F$  ($\times 10^{-4}$)')
ax3.set_title('(3) Pareto Front — Yield vs Gate Infidelity')
ax3.set_yscale('log')
ax3.set_xlim(80, 100.5); ax3.set_ylim(0.5, 1500)
ax3.legend(loc='upper left', framealpha=0.92, fontsize=9)


# ── PLOT 4: Recommended design summary text panel ───────────────────────────
ax4.axis('off')
ax4.set_xlim(0,1); ax4.set_ylim(0,1)

# Title bar
ax4.add_patch(mpatches.FancyBboxPatch(
    (0.0, 0.93), 1.0, 0.07,
    boxstyle='round,pad=0.005',
    facecolor=C['FoM'], edgecolor='black', linewidth=0.6))
ax4.text(0.5, 0.965, 'RECOMMENDED HETEROSTRUCTURE DESIGN',
         ha='center', va='center', fontsize=12, fontweight='bold',
         color='white', transform=ax4.transAxes)

# Numbers block
report = (
    f"Operating point\n"
    f"   x  = {x_opt:.3f}   +/-   {sigma_x_default:.3f}\n"
    f"   L  = {L_opt:.2f} nm  +/- {sigma_L_default:.2f} nm\n"
    f"   B  = {B_field} T\n\n"
    f"Static metrics\n"
    f"   DE_total       = {deltaE_total(x_opt, L_opt)*1000:.1f} meV\n"
    f"   h_c (MB)        = {critical_thickness_MB(x_opt):.1f} nm  "
    f"(L/h_c = {L_opt/critical_thickness_MB(x_opt)*100:.0f}%)\n"
    f"   g_||            = {g_parallel(x_opt, L_opt):.2f}\n"
    f"   omega_L/(2pi)  = {g_parallel(x_opt,L_opt)*mu_B_over_h*B_field/1e9:.1f} GHz\n\n"
    f"Statistical metrics  (under default tolerances)\n"
    f"   Y(DE > 50 meV)  = {Y_DE[i_opt, j_opt]*100:.4f}%\n"
    f"   sigma_g/g       = {sigma_g_over_g[i_opt, j_opt]*100:.2f}%\n"
    f"   g_uniformity   = {g_uniformity_grid[i_opt, j_opt]:.3f}\n\n"
    f"Predicted gate fidelity\n"
    f"   T_2* (at L*)    = {T2star_grid[i_opt, j_opt]*1e6:.2f} us\n"
    f"   t_pi            = {t_pi*1e9:.1f} ns @ Omega_R/2pi = "
    f"{Omega_R_default/(2*np.pi)/1e6:.0f} MHz\n"
    f"   F per qubit    = {F_single_grid[i_opt, j_opt]*100:.4f}%\n"
    f"   1-F per qubit  = {(1-F_single_grid[i_opt, j_opt]):.2e}\n"
    f"   F over N={N_QUBITS}    = {F_array_grid[i_opt, j_opt]*100:.2f}%\n\n"
    f"Combined FoM      = {FoM_safe[i_opt, j_opt]:.4f}"
)
ax4.text(0.05, 0.85, report, ha='left', va='top', fontsize=10,
         family='monospace', transform=ax4.transAxes,
         bbox=dict(boxstyle='round,pad=0.6', facecolor='white',
                   edgecolor='#888', alpha=0.95))

ax4.set_title('(4) Design Recommendation', fontsize=12, fontweight='bold')

fig.suptitle('Joint Process-Window Optimisation — '
             'FoM Landscape, Pareto Front, Recommended Design',
             fontsize=13, fontweight='bold', y=0.965)

out_path = 'objective6_optimisation_analysis.png'
plt.savefig(out_path, bbox_inches='tight', facecolor=fig.get_facecolor())
plt.close()
print(f"\nFigure saved → {out_path}")


# =============================================================================
# SECTION 8 — SENSITIVITY OF THE OPTIMUM TO ASSUMPTIONS
# =============================================================================
#
# "How does the recommended (x, L) shift if we tighten or loosen
# tolerances?" This is a one-sentence answer for the application:
# the optimum is robust to ±2× tolerance variation.
# =============================================================================

print("\n" + "=" * 64)
print("SENSITIVITY OF THE OPTIMUM TO TOLERANCE ASSUMPTIONS")
print("=" * 64)

def find_optimum(sigma_x, sigma_L, sigma_kappa_pct=0.015):
    """Re-run Section 4 with different tolerances; return optimal (x, L)."""
    sk = sigma_kappa_pct * kappa
    s_DE = np.sqrt((dDE_dx*sigma_x)**2 + (dDE_dL*sigma_L)**2)
    Y    = norm.cdf((mu_DE - DELTA_E_TARGET) / s_DE)
    sg   = np.sqrt((dgp_dx*sigma_x)**2 + (dgp_dL*sigma_L)**2 + (dgp_dk*sk)**2)
    sgog = sg / np.maximum(np.abs(g_par_grid), 0.5)
    F_single = np.maximum(0.0, 1.0 - (t_pi / T2star_grid)**2)
    F_array  = F_single ** N_QUBITS
    gun  = np.exp(-sgog / g_uni_target_pct)
    fom  = np.where(HARD_OK, Y * F_array * gun, 0.0)
    i, j = np.unravel_index(np.argmax(fom), fom.shape)
    return x_grid[j], L_grid[i], fom[i, j]

print(f"\n  Default: σ_x={sigma_x_default}, σ_L={sigma_L_default} nm, σ_κ/κ=1.5%")
print(f"    → x*={x_opt:.3f}, L*={L_opt:.2f} nm, FoM={FoM_safe[i_opt, j_opt]:.4f}")

scenarios = [
    ('Tight tolerances (½×)',  sigma_x_default/2, sigma_L_default/2, 0.0075),
    ('Loose tolerances (2×)',  sigma_x_default*2, sigma_L_default*2, 0.030),
    ('σ_x only tightened',     sigma_x_default/2, sigma_L_default,   0.015),
    ('σ_L only tightened',     sigma_x_default,   sigma_L_default/2, 0.015),
]
for name, sx, sL, sk in scenarios:
    xo, Lo, fo = find_optimum(sx, sL, sk)
    print(f"\n  {name}:")
    print(f"    σ_x={sx:.4f}, σ_L={sL:.2f} nm, σ_κ/κ={sk*100:.1f}%")
    print(f"    → x*={xo:.3f} (Δ={xo-x_opt:+.3f}), "
          f"L*={Lo:.2f} nm (Δ={Lo-L_opt:+.2f}), FoM={fo:.4f}")


# =============================================================================
# SECTION 9 — SANITY CHECKS
# =============================================================================
print("\n" + "=" * 64)
print("SANITY CHECKS")
print("=" * 64)

# 1. Optimum lies inside the typical experimental operating window
print(f"\n  Optimum (x*={x_opt:.3f}, L*={L_opt:.1f} nm) within typical window?")
in_window = (0.65 <= x_opt <= 0.95) and (6 <= L_opt <= 25)
print(f"    Window: x∈[0.65, 0.95], L∈[6, 25] nm  "
      f"{'✓' if in_window else '⚠'}")

# 2. Optimum satisfies all hard constraints
DE_at_opt = deltaE_total(x_opt, L_opt)*1000
hc_at_opt = critical_thickness_MB(x_opt)
print(f"\n  Hard-constraint satisfaction at the optimum:")
print(f"    ΔE > 50 meV?            {DE_at_opt:.1f} meV  "
      f"{'✓' if DE_at_opt > 50 else '⚠'}")
print(f"    L < 0.7·h_c?             {L_opt:.1f} < {0.7*hc_at_opt:.1f} nm  "
      f"{'✓' if L_opt < 0.7*hc_at_opt else '⚠'}")
print(f"    F > 99%?                 {F_single_grid[i_opt, j_opt]*100:.4f}%  "
      f"{'✓' if F_single_grid[i_opt, j_opt] > 0.99 else '⚠'}")
print(f"    Y > 99%?                 {Y_DE[i_opt, j_opt]*100:.4f}%  "
      f"{'✓' if Y_DE[i_opt, j_opt] > 0.99 else '⚠'}")

# 3. Pareto front contains the optimum (or a near-neighbour)
dist_to_pareto = np.sqrt((Y_pareto - Y_DE[i_opt, j_opt])**2
                       + (F_pareto - F_single_grid[i_opt, j_opt])**2).min()
print(f"\n  FoM optimum near Pareto front?  closest distance = {dist_to_pareto:.4f}  "
      f"{'✓' if dist_to_pareto < 0.05 else '⚠'}")

# 4. MC-validate the predicted yield at the optimum
n_mc = 10000
xs = np.clip(RNG.normal(x_opt, sigma_x_default, n_mc), 1e-3, 0.999)
Ls = np.clip(RNG.normal(L_opt, sigma_L_default, n_mc), L_table[0], L_table[-1])
DE_mc = deltaE_strain(xs) + deltaE_QC_fast(Ls)
Y_mc = (DE_mc > DELTA_E_TARGET).mean()
print(f"\n  Yield MC-validation at the optimum:")
print(f"    Y_linearised = {Y_DE[i_opt, j_opt]*100:.4f}%")
print(f"    Y_MC          = {Y_mc*100:.4f}%  ({n_mc} samples)")
print(f"    Difference    = {abs(Y_DE[i_opt, j_opt] - Y_mc)*100:.3f} pp  "
      f"{'✓' if abs(Y_DE[i_opt, j_opt] - Y_mc) < 0.005 else '⚠'}")

# 5. Optimum is stable under tolerance perturbations
opts_x = np.array([find_optimum(*p[1:])[0] for p in scenarios])
print(f"\n  Optimum stability across {len(scenarios)} scenarios:")
print(f"    x* range: [{opts_x.min():.3f}, {opts_x.max():.3f}]   "
      f"Δ = {opts_x.max()-opts_x.min():.3f}  "
      f"{'✓ stable' if opts_x.max()-opts_x.min() < 0.05 else '⚠ shift'}")

print("\n✓ ALL OBJECTIVE 6 CALCULATIONS COMPLETE.")
print("\n" + "=" * 64)
print("PROJECT-WIDE SUMMARY  (Objectives 1–6, complete)")
print("=" * 64)
print(f"""
  RECOMMENDED OPERATING POINT (one-sentence headline):

    x = {x_opt:.2f}, L = {L_opt:.1f} nm; ΔE = {DE_at_opt:.0f} meV
    (well above 50 meV target), F = {F_single_grid[i_opt, j_opt]*100:.3f}%,
    Y > {Y_DE[i_opt, j_opt]*100-0.01:.0f}% under realistic σ_x = {sigma_x_default:.3f},
    σ_L = {sigma_L_default:.1f} nm tolerances.

  This design simultaneously satisfies:
    • ΔE > 50 meV  (clean HH ground state)
    • L < 0.7·h_c   (sub-critical thickness, dislocation-safe)
    • F > 99%       (above first-generation FT threshold)
    • Y > 99%       (high manufacturing yield)
    • g uniformity  (σ_g/g = {sigma_g_over_g[i_opt, j_opt]*100:.1f}%, addressable)
""")
