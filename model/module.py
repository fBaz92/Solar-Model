import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize


class PVModule:
    """
    A PV module model based on the single-diode five-parameter model.
    Implements the methodology from Sera et al. "PV Panel Model Based on Datasheet Values"
    """

    def __init__(
        self, name, Isc, Voc, Impp, Vmpp, Pmpp, ki, kv, kp=None, ns=None, Tcell=25
    ):
        """
        Initialize the PV module with datasheet parameters.

        Parameters:
        -----------
        name : str
            Name of the PV module
        Isc : float
            Short-circuit current at STC [A]
        Voc : float
            Open-circuit voltage at STC [V]
        Impp : float
            Current at maximum power point at STC [A]
        Vmpp : float
            Voltage at maximum power point at STC [V]
        Pmpp : float
            Maximum power at STC [W]
        ki : float
            Temperature coefficient of short-circuit current [%/°C]
        kv : float
            Temperature coefficient of open-circuit voltage [V/°C]
        kp : float, optional
            Temperature coefficient of power [%/°C]
        ns : int, optional
            Number of cells in series in the module
        Tcell : float, optional
            Cell temperature at STC [°C], default is 25°C
        """
        # Store datasheet parameters
        self.name = name
        self.Isc_stc = Isc
        self.Voc_stc = Voc
        self.Impp_stc = Impp
        self.Vmpp_stc = Vmpp
        self.Pmpp_stc = Pmpp
        self.ki = ki / 100  # Convert from %/°C to decimal form
        self.kv = kv
        self.kp = kp / 100 if kp is not None else None
        self.Tcell_stc = Tcell + 273.15  # Convert to Kelvin

        # Constants
        self.k = 1.381e-23  # Boltzmann constant [J/K]
        self.q = 1.602e-19  # Elementary charge [C]

        # Determine number of cells if not provided
        if ns is None:
            # Estimate based on typical cell voltage (~0.6V at MPP)
            self.ns = round(self.Voc_stc / 0.6)
        else:
            self.ns = ns

        # Extract the five parameters of the model
        self.extract_parameters()

    def extract_parameters(self):
        """
        Extract the five parameters of the single-diode model following exactly
        the flowchart in Fig. 2 of the paper by Sera et al.
        """
        # Initialize parameters
        Rs = 0.1  # Initial guess for series resistance
        Rsh = 1000  # Initial guess for shunt resistance
        A = 1.3  # Initial guess for ideality factor

        # Maximum iterations and tolerances
        MAX_ITERATIONS = 100
        TOL_DPDV = 1e-6
        TOL_DIDV = 1e-4  # Increased tolerance for dI/dV at Isc

        # Function to calculate Io and Iph from Rs, Rsh, and Vt
        def calculate_Io_Iph(Rs, Rsh, Vt):
            numerator = self.Isc_stc - (self.Voc_stc - self.Isc_stc * Rs) / Rsh
            denominator = np.exp(self.Voc_stc / (self.ns * Vt))
            Io = numerator / denominator
            Iph = Io * np.exp(self.Voc_stc / (self.ns * Vt)) + self.Voc_stc / Rsh
            return Io, Iph

        # Newton-Raphson method for finding Vt (and thus A) from Impp
        def find_Vt_from_Impp(Rs, Rsh, Vt_guess):
            Vt = Vt_guess
            for _ in range(20):  # Max 20 iterations for this inner loop
                # Calculate Io and Iph
                Io, Iph = calculate_Io_Iph(Rs, Rsh, Vt)

                # Calculate Impp using Eq. (12)
                Impp_calc = (
                    Iph
                    - Io * np.exp((self.Vmpp_stc + self.Impp_stc * Rs) / (self.ns * Vt))
                    - (self.Vmpp_stc + self.Impp_stc * Rs) / Rsh
                )

                # If close enough, return
                if abs(Impp_calc - self.Impp_stc) < 1e-10:
                    break

                # Calculate derivative of Impp with respect to Vt (simplified approximation)
                delta_Vt = Vt * 0.001
                Vt_plus = Vt + delta_Vt

                Io_plus, Iph_plus = calculate_Io_Iph(Rs, Rsh, Vt_plus)
                Impp_calc_plus = (
                    Iph_plus
                    - Io_plus
                    * np.exp((self.Vmpp_stc + self.Impp_stc * Rs) / (self.ns * Vt_plus))
                    - (self.Vmpp_stc + self.Impp_stc * Rs) / Rsh
                )

                dImpp_dVt = (Impp_calc_plus - Impp_calc) / delta_Vt

                # Newton-Raphson update
                delta_Vt = (self.Impp_stc - Impp_calc) / dImpp_dVt

                # Limit step size
                delta_Vt = max(min(delta_Vt, Vt * 0.1), -Vt * 0.1)

                Vt += delta_Vt

                # Check convergence
                if abs(delta_Vt) < 1e-10:
                    break

            return Vt

        # Bisection method for finding Rsh from dP/dV at MPP
        def find_Rsh_from_dPdV(Rs, Vt, Rsh_min=50, Rsh_max=5000):
            Rsh_low = Rsh_min
            Rsh_high = Rsh_max

            for _ in range(30):  # Max 30 iterations
                Rsh_mid = (Rsh_low + Rsh_high) / 2

                # Calculate Io and Iph
                Io, Iph = calculate_Io_Iph(Rs, Rsh_mid, Vt)

                # Calculate dP/dV at MPP using numerical differentiation for stability
                delta_V = 0.001 * self.Vmpp_stc

                # Calculate power at Vmpp + delta_V
                V_plus = self.Vmpp_stc + delta_V
                I_plus = (
                    Iph
                    - Io * np.exp((V_plus + self.Impp_stc * Rs) / (self.ns * Vt))
                    - (V_plus + self.Impp_stc * Rs) / Rsh_mid
                )
                P_plus = V_plus * I_plus

                # Calculate power at Vmpp - delta_V
                V_minus = self.Vmpp_stc - delta_V
                I_minus = (
                    Iph
                    - Io * np.exp((V_minus + self.Impp_stc * Rs) / (self.ns * Vt))
                    - (V_minus + self.Impp_stc * Rs) / Rsh_mid
                )
                P_minus = V_minus * I_minus

                # Calculate dP/dV
                dPdV = (P_plus - P_minus) / (2 * delta_V)

                # Bisection update
                if dPdV > 0:
                    Rsh_low = Rsh_mid
                else:
                    Rsh_high = Rsh_mid

                # Check convergence
                if abs(dPdV) < TOL_DPDV or (Rsh_high - Rsh_low) < 1:
                    break

            return Rsh_mid

        # Calculate dI/dV at Isc
        def calculate_dIdV_at_Isc(Rs, Rsh, Vt):
            # Calculate using numerical differentiation for stability
            Io, Iph = calculate_Io_Iph(Rs, Rsh, Vt)

            delta_V = 0.001  # Small voltage step

            # Get I at V = 0 (short circuit)
            I_sc = (
                Iph
                - Io * np.exp(self.Isc_stc * Rs / (self.ns * Vt))
                - (self.Isc_stc * Rs) / Rsh
            )

            # Get I at V = delta_V
            I_delta = (
                Iph
                - Io * np.exp((delta_V + self.Isc_stc * Rs) / (self.ns * Vt))
                - (delta_V + self.Isc_stc * Rs) / Rsh
            )

            # Calculate dI/dV
            dIdV = (I_delta - I_sc) / delta_V

            return dIdV

        # Main iteration loop
        converged = False
        for iteration in range(MAX_ITERATIONS):
            # Step 1: Find Vt (and thus A) from Impp
            Vt = find_Vt_from_Impp(Rs, Rsh, A * self.k * self.Tcell_stc / self.q)
            A = Vt * self.q / (self.k * self.Tcell_stc)

            # Make sure A stays within reasonable bounds
            A = max(1.0, min(A, 2.0))
            Vt = A * self.k * self.Tcell_stc / self.q

            # Step 2: Find Rsh from dP/dV at MPP
            Rsh_new = find_Rsh_from_dPdV(Rs, Vt)

            # Step 3: Calculate dI/dV at Isc
            dIdV_Isc = calculate_dIdV_at_Isc(Rs, Rsh_new, Vt)
            expected_dIdV = -1 / Rsh_new

            # Step 4: Check convergence
            if abs(dIdV_Isc - expected_dIdV) < TOL_DIDV:
                Rsh = Rsh_new
                converged = True
                break

            # Step 5: Update Rs based on dI/dV at Isc
            # If dI/dV is more negative than expected, decrease Rs
            # If dI/dV is less negative than expected, increase Rs
            if dIdV_Isc < expected_dIdV:  # More negative
                Rs_new = Rs * 0.9  # Decrease Rs
            else:  # Less negative
                Rs_new = Rs * 1.1  # Increase Rs

            # Update Rs and Rsh
            Rs = max(0.01, min(Rs_new, 2.0))  # Keep Rs within reasonable bounds
            Rsh = Rsh_new

            # Print progress every 10 iterations
            if iteration % 10 == 0:
                print(
                    f"Iteration {iteration}: Rs={Rs:.4f}, Rsh={Rsh:.1f}, A={A:.4f}, dI/dV={dIdV_Isc:.6f}, expected={expected_dIdV:.6f}"
                )

        # Calculate final Io and Iph
        Io, Iph = calculate_Io_Iph(Rs, Rsh, Vt)

        # Store the final parameters
        self.Rs = Rs
        self.Rsh = Rsh
        self.A = A
        self.Vt_stc = Vt
        self.Io_stc = Io
        self.Iph_stc = Iph

        print(f"Extracted parameters for {self.name}:")
        print(f"Series resistance (Rs): {self.Rs:.4f} Ω")
        print(f"Shunt resistance (Rsh): {self.Rsh:.1f} Ω")
        print(f"Ideality factor (A): {self.A:.4f}")
        print(f"Dark saturation current (Io): {self.Io_stc:.2e} A")
        print(f"Photocurrent (Iph): {self.Iph_stc:.4f} A")
        print(f"Iterations required: {iteration + 1}")
        print(f"Converged: {converged}")

    def get_iv_curve(self, voltage_range=None, irradiance=1000, temperature=25):
        """
        Calculate the I-V curve of the module at given irradiance and temperature.

        Parameters:
        -----------
        voltage_range : array-like, optional
            Array of voltage values to calculate current for
        irradiance : float, optional
            Irradiance in W/m² (default 1000 W/m² = 1 sun)
        temperature : float, optional
            Cell temperature in °C (default 25°C)

        Returns:
        --------
        V : array
            Voltage values [V]
        I : array
            Current values [A]
        """
        # Convert temperature to Kelvin
        T = temperature + 273.15

        # Normalized irradiance
        G = irradiance / 1000

        # If voltage range not provided, create one
        if voltage_range is None:
            # Estimate Voc at the given conditions
            Voc_est = self.Voc_stc + self.kv * (temperature - 25)
            voltage_range = np.linspace(0, Voc_est * 1.1, 500)

        # Update parameters for new conditions
        # Temperature-dependent thermal voltage
        Vt = self.A * self.k * T / self.q

        # Temperature-dependent short-circuit current
        Isc_T = self.Isc_stc * (1 + self.ki * (temperature - 25))

        # Temperature-dependent open-circuit voltage
        Voc_T = self.Voc_stc + self.kv * (temperature - 25)

        # Irradiance and temperature dependent photo-current
        Iph = Isc_T * G

        # Calculate dark saturation current for new temperature (using the paper's recommended method)
        numerator = Isc_T - (Voc_T - Isc_T * self.Rs) / self.Rsh
        denominator = np.exp(Voc_T / (self.ns * Vt))
        Io = numerator / denominator

        # Calculate current for each voltage point using the single-diode model
        current = np.zeros_like(voltage_range)
        for i, v in enumerate(voltage_range):
            # Initial guess for current
            if i == 0:
                I_guess = Iph
            else:
                I_guess = current[i - 1]

            # Define function to solve for current
            def current_equation(I):
                return I - (
                    Iph
                    - Io * np.exp((v + I * self.Rs) / (self.ns * Vt))
                    - (v + I * self.Rs) / self.Rsh
                )

            # Solve for current
            solution = optimize.root_scalar(
                current_equation, x0=I_guess, method="secant", bracket=[0, Iph * 1.1]
            )
            if solution.converged:
                current[i] = max(0, solution.root)  # No negative current
            else:
                current[i] = 0

        return voltage_range, current

    def get_pv_curve(self, voltage_range=None, irradiance=1000, temperature=25):
        """
        Calculate the P-V curve of the module at given irradiance and temperature.

        Parameters:
        -----------
        voltage_range : array-like, optional
            Array of voltage values to calculate power for
        irradiance : float, optional
            Irradiance in W/m² (default 1000 W/m² = 1 sun)
        temperature : float, optional
            Cell temperature in °C (default 25°C)

        Returns:
        --------
        V : array
            Voltage values [V]
        P : array
            Power values [W]
        """
        V, I = self.get_iv_curve(voltage_range, irradiance, temperature)
        P = V * I
        return V, P

    def get_mpp(self, irradiance=1000, temperature=25):
        """
        Calculate the maximum power point (MPP) at given irradiance and temperature.

        Parameters:
        -----------
        irradiance : float, optional
            Irradiance in W/m² (default 1000 W/m² = 1 sun)
        temperature : float, optional
            Cell temperature in °C (default 25°C)

        Returns:
        --------
        V_mpp : float
            Voltage at maximum power point [V]
        I_mpp : float
            Current at maximum power point [A]
        P_mpp : float
            Maximum power [W]
        """
        V, P = self.get_pv_curve(irradiance=irradiance, temperature=temperature)
        idx_mpp = np.argmax(P)
        V_mpp = V[idx_mpp]
        P_mpp = P[idx_mpp]
        _, I = self.get_iv_curve(
            voltage_range=[V_mpp], irradiance=irradiance, temperature=temperature
        )
        I_mpp = I[0]
        return V_mpp, I_mpp, P_mpp

    def plot_iv_curve(self, irradiances=None, temperatures=None, figsize=(10, 6)):
        """
        Plot the I-V curve for different irradiance or temperature values.

        Parameters:
        -----------
        irradiances : list, optional
            List of irradiance values in W/m² to plot
        temperatures : list, optional
            List of temperature values in °C to plot
        figsize : tuple, optional
            Figure size (width, height) in inches

        Returns:
        --------
        fig : matplotlib.figure.Figure
            The created figure
        """
        fig, ax = plt.subplots(figsize=figsize)

        if irradiances is not None and temperatures is None:
            # Plot I-V curves for different irradiance levels
            for G in irradiances:
                V, I = self.get_iv_curve(irradiance=G)
                ax.plot(V, I, label=f"G = {G} W/m²")
            ax.set_title(
                f"I-V Characteristics at Different Irradiance Levels (T = 25°C)"
            )

        elif temperatures is not None and irradiances is None:
            # Plot I-V curves for different temperatures
            for T in temperatures:
                V, I = self.get_iv_curve(temperature=T)
                ax.plot(V, I, label=f"T = {T}°C")
            ax.set_title(
                f"I-V Characteristics at Different Temperatures (G = 1000 W/m²)"
            )

        else:
            # Default: plot I-V curve at STC
            V, I = self.get_iv_curve()
            ax.plot(V, I, label="STC (1000 W/m², 25°C)")
            ax.set_title(f"I-V Characteristic at STC")

        ax.set_xlabel("Voltage [V]")
        ax.set_ylabel("Current [A]")
        ax.grid(True)
        ax.legend()
        plt.tight_layout()

        return fig

    def plot_pv_curve(self, irradiances=None, temperatures=None, figsize=(10, 6)):
        """
        Plot the P-V curve for different irradiance or temperature values.

        Parameters:
        -----------
        irradiances : list, optional
            List of irradiance values in W/m² to plot
        temperatures : list, optional
            List of temperature values in °C to plot
        figsize : tuple, optional
            Figure size (width, height) in inches

        Returns:
        --------
        fig : matplotlib.figure.Figure
            The created figure
        """
        fig, ax = plt.subplots(figsize=figsize)

        if irradiances is not None and temperatures is None:
            # Plot P-V curves for different irradiance levels
            for G in irradiances:
                V, P = self.get_pv_curve(irradiance=G)
                ax.plot(V, P, label=f"G = {G} W/m²")
                # Mark MPP
                V_mpp, _, P_mpp = self.get_mpp(irradiance=G)
                ax.plot(V_mpp, P_mpp, "o", color="red")

            ax.set_title(
                f"P-V Characteristics at Different Irradiance Levels (T = 25°C)"
            )

        elif temperatures is not None and irradiances is None:
            # Plot P-V curves for different temperatures
            for T in temperatures:
                V, P = self.get_pv_curve(temperature=T)
                ax.plot(V, P, label=f"T = {T}°C")
                # Mark MPP
                V_mpp, _, P_mpp = self.get_mpp(temperature=T)
                ax.plot(V_mpp, P_mpp, "o", color="red")

            ax.set_title(
                f"P-V Characteristics at Different Temperatures (G = 1000 W/m²)"
            )

        else:
            # Default: plot P-V curve at STC
            V, P = self.get_pv_curve()
            ax.plot(V, P, label="STC (1000 W/m², 25°C)")
            # Mark MPP
            V_mpp, _, P_mpp = self.get_mpp()
            ax.plot(V_mpp, P_mpp, "o", color="red", label="MPP")

            ax.set_title(f"P-V Characteristic at STC")

        ax.set_xlabel("Voltage [V]")
        ax.set_ylabel("Power [W]")
        ax.grid(True)
        ax.legend()
        plt.tight_layout()

        return fig
