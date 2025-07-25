# -*- coding: utf-8 -*-
"""
Full Python Script for the Analysis of the Fermi-Dirac Distribution
and its Tanh Approximation.
"""

# ===============================
#  SETUP: IMPORTS & CONSTANTS
# ===============================
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import linregress

# Use a professional plotting style
plt.style.use('seaborn-v0_8-whitegrid')

# Boltzmann constant in eV/K
k_B = 8.617e-5

# ===============================
#  CORE FUNCTIONS
# ===============================

def fermi_dirac(E, E_F, T):
    """Computes the exact Fermi-Dirac distribution."""
    T_eff = T if T > 0 else 1e-12
    # Cap exponent to avoid overflow, crucial for numerical stability
    exponent = np.clip((E - E_F) / (k_B * T_eff), -700, 700)
    return 1 / (np.exp(exponent) + 1)

def tanh_fit_model(E, E_F, alpha):
    """Differentiable tanh model used for fitting."""
    return 0.5 * (1 - np.tanh((E - E_F) / (2 * alpha)))

def fermi_derivatives(E, E_F, T):
    """Computes the first and second derivatives of the F-D function."""
    delta = k_B * T
    exp_term = np.exp((E - E_F) / delta)
    denom_sq = (exp_term + 1)**2
    first_deriv = -exp_term / (delta * denom_sq)
    second_deriv = (exp_term * (exp_term - 1)) / (delta**2 * (exp_term + 1)**3)
    return first_deriv, second_deriv

# ===============================
#  PLOTTING FUNCTIONS
# ===============================

def plot_f0_vs_temperature(temperature_list, E_F):
    """Plots f(E) for different temperatures."""
    E_vals = np.linspace(E_F - 0.2, E_F + 0.2, 1000)
    plt.figure(figsize=(10, 6))
    for T in temperature_list:
        plt.plot(E_vals, fermi_dirac(E_vals, E_F, T), label=f"$T = {T}$ K")
    plt.axvline(E_F, color='r', linestyle='--', label="$E_F$")
    plt.xlabel("Energy (eV)"); plt.ylabel("$f(E)$")
    plt.title("Fermi-Dirac Distribution at Different Temperatures")
    plt.legend(); plt.tight_layout(); plt.show()

def plot_derivatives(E_F, T):
    """Plots the first and second derivatives."""
    E_vals = np.linspace(E_F - 0.2, E_F + 0.2, 1000)
    f1, f2 = fermi_derivatives(E_vals, E_F, T)
    plt.figure(figsize=(10, 6))
    plt.plot(E_vals, f1, label="1st Derivative")
    plt.plot(E_vals, f2, label="2nd Derivative")
    plt.axvline(E_F, color='r', linestyle='--', label="$E_F$")
    plt.axhline(0, color='gray', linestyle='--', alpha=0.6)
    plt.xlabel("Energy (eV)"); plt.ylabel("Derivative")
    plt.title(f"Derivatives of $f(E)$ at T = {T} K")
    plt.legend(); plt.tight_layout(); plt.show()

# ===============================
#  MAIN ANALYSIS FUNCTION
# ===============================

def perform_error_analysis(temperatures, E_F):
    """
    Performs the full fitting and error analysis workflow.
    """
    E_fit_range = np.linspace(E_F - 0.5, E_F + 0.5, 1000)
    alphas = []
    errors = []

    for T in temperatures:
        f_exact = fermi_dirac(E_fit_range, E_F, T)
        
        # Fit the model using curve_fit
        # The lambda function fixes E_F, so only alpha is fitted.
        # p0 is the initial guess for alpha, crucial for convergence.
        popt, _ = curve_fit(lambda E, alpha: tanh_fit_model(E, E_F, alpha),
                            E_fit_range, f_exact, p0=[k_B * T])
        alpha_fit = popt[0]
        alphas.append(alpha_fit)
        
        f_fitted = tanh_fit_model(E_fit_range, E_F, alpha_fit)
        rmse = np.sqrt(np.mean((f_exact - f_fitted)**2))
        errors.append(rmse)

        # Plot each individual comparison for verification
        plt.figure(figsize=(8, 5))
        plt.plot(E_fit_range, f_exact, label=f'Fermi-Dirac (T={T}K)')
        plt.plot(E_fit_range, f_fitted, '--', label=f'Tanh Fit ($\\alpha$={alpha_fit:.2e} eV)')
        plt.title(f'Tanh Fit vs Fermi-Dirac (T={T}K)')
        plt.xlabel('Energy (eV)'); plt.ylabel('Occupancy')
        plt.legend(); plt.grid(True); plt.tight_layout(); plt.show()
    
    # After the loop, plot summary graphs
    plt.figure(figsize=(8, 5))
    plt.plot(temperatures, alphas, 's--', color='darkorange', label='Fitted $\\alpha$')
    plt.xlabel('Temperature (K)'); plt.ylabel('$\\alpha$ (eV)')
    plt.title('Fitted Parameter $\\alpha$ vs Temperature')
    plt.grid(True); plt.legend(); plt.tight_layout(); plt.show()

    plt.figure(figsize=(8, 5))
    plt.plot(temperatures, errors, 'o-', label='RMSE')
    plt.xlabel('Temperature (K)'); plt.ylabel('Fit Error (RMSE)')
    plt.title('Fit Error vs Temperature')
    plt.grid(True); plt.legend(); plt.tight_layout(); plt.show()
    return errors, np.array(alphas)

# ===============================
#  MAIN EXECUTION BLOCK
# ===============================
if __name__ == '__main__':
    # Define primary parameters for the study
    E_F_main = 0.5
    
    # --- Part 1: Foundational Visualization ---
    print("--- Generating Foundational Plots ---")
    plot_f0_vs_temperature([10, 50, 100, 200, 400], E_F_main)
    plot_derivatives(E_F_main, T=100)
    print("Done.\n")

    # --- Part 2: Fitting Analysis ---
    print("--- Performing Fitting and Error Analysis ---")
    fit_temps = [10, 50, 100, 150, 200]
    err_vals, alpha_vals = perform_error_analysis(fit_temps, E_F_main)
    print("Done.\n")

    # --- Part 3: Quantitative Validation ---
    print("--- Quantitative Validation via Linear Regression ---")
    slope, intercept, r_val, p_val, std_err = linregress(fit_temps, alpha_vals)
    print(f"Slope (Fitted k_B) : {slope:.4e} +/- {std_err:.2e} eV/K")
    print(f"Y-intercept          : {intercept:.4e} eV")
    print(f"R-squared            : {r_val**2:.6f}")
    print(f"Actual k_B           : {k_B:.4e} eV/K")
    print("--- Analysis Complete ---")