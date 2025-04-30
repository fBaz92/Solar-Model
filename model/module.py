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
        Extract the five parameters of the single-diode model:
        - A (diode ideality factor)
        - Rs (series resistance)
        - Rsh (shunt resistance)
        - Io (dark saturation current)
        - Iph (photocurrent)
        """
        # Initial guesses
        A_initial = 1.3  # Common ideality factor value
        Rs_initial = 0.1  # Initial guess for series resistance
        Rsh_initial = 1000  # Initial guess for shunt resistance

        # Define objective function to minimize for parameter extraction
        def objective_function(params):
            Rs, Rsh, A = params

            # Calculate thermal voltage
            Vt = A * self.k * self.Tcell_stc / self.q

            # Calculate Io based on Rs and Rsh
            numerator = self.Isc_stc - (self.Voc_stc - self.Isc_stc * Rs) / Rsh
            denominator = np.exp(self.Voc_stc / (self.ns * Vt))
            Io = numerator / denominator

            # Calculate Iph
            Iph = Io * np.exp(self.Voc_stc / (self.ns * Vt)) + self.Voc_stc / Rsh

            # Calculate I-V at MPP
            I_mpp_calc = (
                Iph
                - Io * np.exp((self.Vmpp_stc + self.Impp_stc * Rs) / (self.ns * Vt))
                - (self.Vmpp_stc + self.Impp_stc * Rs) / Rsh
            )

            # Calculate dP/dV at MPP (should be zero)
            # This is a simplified approximation
            dP_dV = self.Impp_stc + self.Vmpp_stc * (
                -Io / Vt * np.exp((self.Vmpp_stc + self.Impp_stc * Rs) / (self.ns * Vt))
                - 1 / Rsh
            )

            # Calculate dI/dV at Isc (should be -1/Rsh)
            dI_dV_Isc = -1 / Rsh - Io * Rs / (self.ns * Vt) * np.exp(
                self.Isc_stc * Rs / (self.ns * Vt)
            )

            # Return sum of squared errors
            return (
                (I_mpp_calc - self.Impp_stc) ** 2
                + dP_dV**2
                + (dI_dV_Isc + 1 / Rsh) ** 2
            )

        # Perform optimization
        bounds = [(0.001, 5), (10, 10000), (1, 2)]  # (Rs, Rsh, A)
        result = optimize.minimize(
            objective_function,
            [Rs_initial, Rsh_initial, A_initial],
            bounds=bounds,
            method="L-BFGS-B",
        )

        # Store results
        self.Rs, self.Rsh, self.A = result.x

        # Calculate thermal voltage
        self.Vt_stc = self.A * self.k * self.Tcell_stc / self.q

        # Calculate Io and Iph at STC
        numerator = self.Isc_stc - (self.Voc_stc - self.Isc_stc * self.Rs) / self.Rsh
        denominator = np.exp(self.Voc_stc / (self.ns * self.Vt_stc))
        self.Io_stc = numerator / denominator
        self.Iph_stc = (
            self.Io_stc * np.exp(self.Voc_stc / (self.ns * self.Vt_stc))
            + self.Voc_stc / self.Rsh
        )

        print(f"Extracted parameters for {self.name}:")
        print(f"Series resistance (Rs): {self.Rs:.4f} Ω")
        print(f"Shunt resistance (Rsh): {self.Rsh:.1f} Ω")
        print(f"Ideality factor (A): {self.A:.4f}")
        print(f"Dark saturation current (Io): {self.Io_stc:.2e} A")
        print(f"Photocurrent (Iph): {self.Iph_stc:.4f} A")

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
