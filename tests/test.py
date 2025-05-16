import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

# Constants
k = 1.381e-23  # Boltzmann constant [J/K]
q = 1.602e-19  # Elementary charge [C]

# PV module datasheet parameters (example: ARCO M-52)
Isc_stc = 7.811  # Short-circuit current at STC [A]
Voc_stc = 6.808  # Open-circuit voltage at STC [V]
Impp_stc = 7.183  # Current at maximum power point at STC [A]
Vmpp_stc = 5.389  # Voltage at maximum power point at STC [V]
Pmpp_stc = Impp_stc * Vmpp_stc  # Maximum power at STC [W]
Tcell_stc = 25 + 273.15  # Cell temperature at STC [K]

# Estimated number of cells in series (typical for this voltage range)
ns = round(Voc_stc / 0.6)

print("=== PV Module Parameter Extraction - Simplified Explicit Method ===")
print(f"Module data:")
print(f"Isc = {Isc_stc} A")
print(f"Voc = {Voc_stc} V")
print(f"Impp = {Impp_stc} A")
print(f"Vmpp = {Vmpp_stc} V")
print(f"Pmpp = {Pmpp_stc:.2f} W")
print(f"Estimated cells in series: {ns}")
print()

# According to the simplified explicit method, we make these assumptions:
# 1. IL ≈ Isc (light current equals short-circuit current)
# 2. The "-1" terms in exponentials can be omitted as they are much smaller than the exponential terms
# 3. We introduce A = q/(γ*k*Tc) to simplify notation

print("Step 1: Simplifications and objective functions")
print("Following Sera et al., we make the following simplifications:")
print("- IL = Isc (light current equals short-circuit current)")
print("- Drop '-1' terms from exponentials (they are negligible)")
print("- Define A = q/(γ*k*Tc) for cleaner notation")
print()

# Define the simplified objective functions as lambda expressions
# F1: IL = Isc (trivial, already satisfied by our assumption)
F1 = lambda IL, Isc: IL - Isc

# F2: At open circuit (I=0, V=Voc)
# 0 = IL - Io*exp(A*Voc)
# Rearranged: Io = IL*exp(A*Voc) = Isc*exp(A*Voc)
F2 = lambda A, Voc, Isc: Isc - Isc * np.exp(A * Voc)

# F3: At maximum power point
# Impp = IL - Io*exp(A*(Vmpp + Impp*Rs))
# Substituting IL=Isc and Io from F2:
# Impp = Isc - Isc*exp(-A*Voc)*exp(A*(Vmpp + Impp*Rs))
# Impp = Isc[1 - exp(A*(Vmpp - Voc + Impp*Rs))]
F3 = lambda A, Rs, Isc, Impp, Voc, Vmpp: -Impp + Isc * (
    1 - np.exp(A * (Vmpp - Voc + Impp * Rs))
)

# F4: dP/dV = 0 at maximum power point
# After differentiation and simplification (see paper for full derivation):
# [1 - exp(A*(Vmpp - Voc + Impp*Rs))] * Vmpp + A*Vmpp / [1 + A*Rs*Isc*exp(A*(Vmpp - Voc + Impp*Rs))] = 0
F4 = lambda A, Rs, Isc, Impp, Voc, Vmpp: (
    1
    - (np.exp(A * (Vmpp - Voc + Impp * Rs)))
    * (1 + (A * Vmpp) / (1 + A * Rs * Isc * np.exp(A * (Vmpp - Voc + Impp * Rs))))
    - 0
)

print("Objective functions defined as lambda expressions:")
print("F1: IL = Isc (trivial)")
print("F2: Io = Isc * exp(-A * Voc)")
print("F3: Impp = Isc * [1 - exp(A * (Vmpp - Voc + Impp * Rs))]")
print("F4: dP/dV = 0 at MPP (see paper for full derivation)")
print()

# Step 2: Solve for A from the simplified system
print("Step 2: Solve for parameter A")
print("From the paper's equation (2.77), we can solve for A explicitly:")

# From equation (2.77) in the paper:
numerator = Isc_stc / (Isc_stc - Impp_stc) + np.log(1 - Impp_stc / Isc_stc)
denominator = 2 * (Vmpp_stc - Voc_stc)
A_solved = numerator / denominator

print(f"A = {A_solved:.6f}")

# Step 3: Solve for Rs using the MPP constraint
print("\nStep 3: Solve for series resistance Rs")
print("Using F3 rearranged to solve for Rs:")

# From F3, solve for Rs:
# Impp = Isc[1 - exp(A*(Vmpp - Voc + Impp*Rs))]
# exp(A*(Vmpp - Voc + Impp*Rs)) = 1 - Impp/Isc
# A*(Vmpp - Voc + Impp*Rs) = ln(1 - Impp/Isc)
# Rs = [ln(1 - Impp/Isc)/A - (Vmpp - Voc)] / Impp

Rs_solved = (
    np.log(1 - Impp_stc / Isc_stc) / A_solved - (Vmpp_stc - Voc_stc)
) / Impp_stc

print(f"Rs = {Rs_solved:.6f} Ω")

# Step 4: Calculate remaining parameters
print("\nStep 4: Calculate remaining parameters")

# IL = Isc (by assumption)
IL_solved = Isc_stc
print(f"IL = {IL_solved:.6f} A")

# Io = Isc * exp(-A * Voc)
Io_solved = Isc_stc * np.exp(-A_solved * Voc_stc)
print(f"Io = {Io_solved:.2e} A")

# γ = q / (A * k * Tc)
gamma_solved = q / (A_solved * k * Tcell_stc)
print(f"γ = {gamma_solved:.6f}")

# Ideality factor n = γ / ns
n_solved = gamma_solved / ns
print(f"Ideality factor n = {n_solved:.6f}")

print(f"\n=== Final Parameters ===")
print(f"Light current (IL): {IL_solved:.6f} A")
print(f"Dark saturation current (Io): {Io_solved:.2e} A")
print(f"Series resistance (Rs): {Rs_solved:.6f} Ω")
print(f"Diode factor (γ): {gamma_solved:.6f}")
print(f"Ideality factor (n): {n_solved:.6f}")

# Step 5: Verification - check if the solved parameters satisfy the original equations
print(f"\n=== Verification ===")


# Define the single-diode model current equation
def single_diode_current(V, IL, Io, Rs, gamma, k_T):
    """Calculate current using the single-diode model"""
    return IL - Io * (np.exp((V + IL * Rs) * gamma / (k_T * ns)) - 1)


# Check the three key points
Vt_stc = k * Tcell_stc / q  # Thermal voltage

# At short circuit (V=0)
I_at_sc = single_diode_current(
    0, IL_solved, Io_solved, Rs_solved, gamma_solved, k * Tcell_stc
)
print(f"At V=0: I = {I_at_sc:.6f} A (should be ≈ {Isc_stc} A)")

# At open circuit
# We need to solve for I when V=Voc
from scipy.optimize import fsolve


def current_equation(I, V):
    return I - (
        IL_solved
        - Io_solved
        * (np.exp((V + I * Rs_solved) * gamma_solved / (k * Tcell_stc * ns)) - 1)
    )


I_at_oc = fsolve(lambda I: current_equation(I, Voc_stc), 0)[0]
print(f"At V={Voc_stc}V: I = {I_at_oc:.6f} A (should be ≈ 0 A)")

# At maximum power point
I_at_mpp = fsolve(lambda I: current_equation(I, Vmpp_stc), Impp_stc)[0]
print(f"At V={Vmpp_stc}V: I = {I_at_mpp:.6f} A (should be ≈ {Impp_stc} A)")

# Calculate power at MPP
P_at_mpp = Vmpp_stc * I_at_mpp
print(f"Power at MPP: {P_at_mpp:.2f} W (should be ≈ {Pmpp_stc:.2f} W)")

# Step 6: Generate and plot I-V curve
print(f"\n=== I-V Curve Generation ===")

# Voltage range from 0 to slightly above Voc
V_range = np.linspace(0, Voc_stc * 1.1, 500)
I_curve = np.zeros_like(V_range)

# Solve for current at each voltage point
for i, V in enumerate(V_range):
    if i == 0:
        I_guess = IL_solved
    else:
        I_guess = I_curve[i - 1]

    try:
        I_solution = fsolve(lambda I: current_equation(I, V), I_guess)[0]
        I_curve[i] = max(0, I_solution)  # No negative current
    except:
        I_curve[i] = 0

# Calculate power curve
P_curve = V_range * I_curve

# Plot I-V and P-V curves
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# I-V curve
ax1.plot(V_range, I_curve, "b-", linewidth=2, label="I-V curve")
ax1.plot(0, Isc_stc, "ro", markersize=8, label=f"Isc = {Isc_stc} A")
ax1.plot(Voc_stc, 0, "go", markersize=8, label=f"Voc = {Voc_stc} V")
ax1.plot(
    Vmpp_stc, Impp_stc, "mo", markersize=8, label=f"MPP ({Vmpp_stc}V, {Impp_stc}A)"
)
ax1.set_xlabel("Voltage [V]")
ax1.set_ylabel("Current [A]")
ax1.set_title("I-V Characteristic")
ax1.grid(True, alpha=0.3)
ax1.legend()

# P-V curve
ax2.plot(V_range, P_curve, "r-", linewidth=2, label="P-V curve")
ax2.plot(Vmpp_stc, Pmpp_stc, "mo", markersize=8, label=f"MPP ({Pmpp_stc:.2f}W)")
ax2.set_xlabel("Voltage [V]")
ax2.set_ylabel("Power [W]")
ax2.set_title("P-V Characteristic")
ax2.grid(True, alpha=0.3)
ax2.legend()

plt.tight_layout()
plt.show()

print("Parameter extraction completed successfully!")
