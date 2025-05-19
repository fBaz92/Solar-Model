import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
from dataclasses import dataclass
from typing import Tuple, Optional


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

    def _newton_raphson(self, f, f_prime, x0, tol=1e-4, max_iter=100):
        """
        Newton-Raphson method for solving non-linear equations.

        Args:
            f (function): The function to solve
            f_prime (function): The derivative of the function
            x0 (float): The initial guess
            tol (float, optional): The tolerance. Defaults to 1e-4.
            max_iter (int, optional): The maximum number of iterations. Defaults to 100.
        """
        value = f(x0)

        x_new = x0 - value / f_prime(x0)

        if abs(x_new - x0) < tol:
            return x_new

        return self._newton_raphson(f, f_prime, x_new, tol, max_iter)

    def extract_parameters(self):
        """This function extracts the four parameters of the single-diode model.

        Returns:
            None
        """

        import math

        # Initialize parameters with more reasonable values based on datasheet
        # Make initial estimate for Rs based on approximate voltage drop
        Rs_max = (self.Voc_stc - self.Vmpp_stc) / self.Impp_stc
        Rs_min = 1e-4

        Rsh_min = self.Vmpp_stc / (self.Isc_stc - self.Impp_stc) - Rs_max
        Rsh_max = 100 * Rsh_min

        A_min = 1.0
        A_max = 2.0
        I0_min = 1e-11
        I0_max = 1e-6

        # Maximum iterations and tolerances
        TOL = 1e-4

        def f(Iph, I0, rs, A, rsh, v, i):
            """function that describes the output current in function to the electrical parameters of a cell

            Args:
                Iph (float): _description_
                I0 (float): _description_
                rs (float): _description_
                A (float): _description_
                rsh (float): _description_
                v (float): _description_
                i (float): _description_

            Returns:
                error: returns the diffence between the expected current and the current that comes from the given electrical parameters
            """
            return (
                Iph
                - I0 * (math.exp((v + i * rs) / (self.ns * A * self.Vt)) - 1)
                - i
                - (v + i * rs) / rsh
            )

        def f_prime(Iph, I0, rs, A, rsh, v, i):
            """Function that returns the derivative of the function f with respect to the current

            Args:
                I0 (float): Dark saturation current
                rs (float): Series resistance
                A (float): Ideality factor
                rsh (float): Shunt resistance
                v (float): Voltage
                i (float): Current

            Returns:
                float: Derivative of the function f with respect to the current
            """

            return (
                -(I0 * rs / (self.ns * A * self.Vt))
                * (math.exp((v + i * rs) / (self.ns * A * self.Vt)))
                - 1
                - rs / rsh
            )

        interesting_v_points = [0, self.Vmpp_stc, self.Voc_stc]
        interesting_i_points = [self.Isc_stc, self.Impp_stc, 0]

        Iph = self.Isc_stc  # approximation that makes sense

        best_set_solutions = {"rs": None, "rsh": None, "a": None, "io": None}
        error_best_set_solutions = float("inf")

        for rs in np.linspace(Rs_min, Rs_max, 100):
            for rsh in np.linspace(Rsh_min, Rsh_max, 100):
                for a in np.linspace(A_min, A_max, 20):
                    for io in np.linspace(I0_min, I0_max, 100):
                        # define the function

                        solutions = []

                        for v, i_target in zip(
                            interesting_v_points, interesting_i_points
                        ):
                            f_ = lambda i: f(Iph, io, rs, a, rsh, v, i)
                            f_prime_ = lambda i: f_prime(Iph, io, rs, a, rsh, v, i)

                            i_found = self._newton_raphson(f_, f_prime_, i_target)

                            solutions.append(i_found)

                        error = sum(abs(solutions - interesting_i_points))

                        if error < error_best_set_solutions:
                            error_best_set_solutions = error
                            best_set_solutions = {
                                "rs": rs,
                                "rsh": rsh,
                                "a": a,
                                "io": io,
                            }

        print(f"Best set of solutions: {best_set_solutions}")
        print(f"Error: {error_best_set_solutions}")

        # # Store the final parameters
        # self.Rs = Rs
        # self.Rsh = Rsh
        # self.A = A
        # self.Vt_stc = Vt
        # self.Io_stc = Io
        # self.Iph_stc = Iph

        # print(f"Extracted parameters for {self.name}:")
        # print(f"Series resistance (Rs): {self.Rs:.4f} Ω")
        # print(f"Shunt resistance (Rsh): {self.Rsh:.1f} Ω")
        # print(f"Ideality factor (A): {self.A:.4f}")
        # print(f"Dark saturation current (Io): {self.Io_stc:.2e} A")
        # print(f"Photocurrent (Iph): {self.Iph_stc:.4f} A")
        # print(f"Iterations required: {iteration + 1}")
        # print(f"Converged: {converged}")

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


@dataclass
class PVParameters:
    """Data class to hold PV model parameters"""

    Rs: float  # Series resistance [Ω]
    Rsh: float  # Shunt resistance [Ω]
    A: float  # Ideality factor
    Io: float  # Dark saturation current [A]
    Iph: float  # Photocurrent [A]
    error: float = 0.0  # Final optimization error


class OptimizedPVModule:
    """
    Optimized PV module model with efficient parameter extraction.
    Uses a combination of genetic algorithm initialization and gradient-based optimization.
    """

    def __init__(
        self, name, Isc, Voc, Impp, Vmpp, Pmpp, ki, kv, kp=None, ns=None, Tcell=25
    ):
        """Initialize PV module with datasheet parameters."""
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
            self.ns = round(self.Voc_stc / 0.6)
        else:
            self.ns = ns

        # Extract parameters using optimized method
        self.parameters = self.extract_parameters_optimized()

    def single_diode_model(self, V: float, I: float, params: PVParameters) -> float:
        """
        Single diode model equation.
        Returns the error (should be zero when correct).
        """
        Vt = self.k * self.Tcell_stc / self.q / self.ns  # Thermal voltage per cell

        return (
            params.Iph
            - params.Io * (np.exp((V + I * params.Rs) / (params.A * Vt)) - 1)
            - (V + I * params.Rs) / params.Rsh
            - I
        )

    def objective_function(self, x: np.ndarray) -> float:
        """
        Objective function for optimization.
        Minimizes the sum of squared errors at the three key points.
        """
        Rs, Rsh, A, Io = x
        Iph = self.Isc_stc  # Approximation

        params = PVParameters(Rs=Rs, Rsh=Rsh, A=A, Io=Io, Iph=Iph)

        # Three key points: (V, I)
        points = [
            (0, self.Isc_stc),  # Short circuit
            (self.Voc_stc, 0),  # Open circuit
            (self.Vmpp_stc, self.Impp_stc),  # Maximum power point
        ]

        error = 0
        for V, I in points:
            model_error = self.single_diode_model(V, I, params)
            error += model_error**2

        return error

    def get_parameter_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get reasonable bounds for parameters based on datasheet values."""
        # Series resistance bounds
        Rs_max = (self.Voc_stc - self.Vmpp_stc) / self.Impp_stc
        Rs_min = 1e-6

        # Shunt resistance bounds
        Rsh_min = self.Vmpp_stc / (self.Isc_stc - self.Impp_stc) - Rs_max
        Rsh_max = 1000 * Rsh_min

        # Ideality factor bounds
        A_min = 0.8
        A_max = 2.0

        # Dark saturation current bounds
        Io_min = 1e-12
        Io_max = 1e-6

        lower_bounds = np.array([Rs_min, Rsh_min, A_min, Io_min])
        upper_bounds = np.array([Rs_max, Rsh_max, A_max, Io_max])

        return lower_bounds, upper_bounds

    def genetic_algorithm_init(
        self, population_size: int = 50, generations: int = 100
    ) -> np.ndarray:
        """
        Use a simple genetic algorithm to find a good initial guess.
        """
        lower_bounds, upper_bounds = self.get_parameter_bounds()

        # Initialize population
        population = np.random.uniform(lower_bounds, upper_bounds, (population_size, 4))

        best_individual = None
        best_fitness = float("inf")

        for generation in range(generations):
            # Evaluate fitness
            fitness = np.array([self.objective_function(ind) for ind in population])

            # Track best individual
            min_idx = np.argmin(fitness)
            if fitness[min_idx] < best_fitness:
                best_fitness = fitness[min_idx]
                best_individual = population[min_idx].copy()

            # Early stopping if error is low enough
            if best_fitness < 1e-6:
                break

            # Selection (tournament selection)
            new_population = []
            for _ in range(population_size):
                # Tournament selection
                tournament_size = 3
                tournament_indices = np.random.choice(population_size, tournament_size)
                tournament_fitness = fitness[tournament_indices]
                winner_idx = tournament_indices[np.argmin(tournament_fitness)]
                new_population.append(population[winner_idx].copy())

            population = np.array(new_population)

            # Mutation
            mutation_rate = 0.1
            mutation_strength = 0.1
            for i in range(population_size):
                if np.random.random() < mutation_rate:
                    # Add Gaussian noise
                    noise = np.random.normal(0, mutation_strength, 4)
                    population[i] += noise * (upper_bounds - lower_bounds)
                    # Ensure bounds
                    population[i] = np.clip(population[i], lower_bounds, upper_bounds)

        return best_individual

    def extract_parameters_optimized(self) -> PVParameters:
        """
        Extract PV model parameters using optimized approach:
        1. Genetic algorithm for initial guess
        2. Gradient-based optimization for fine-tuning
        """
        print(f"Optimizing parameters for {self.name}...")

        # Step 1: Get good initial guess using genetic algorithm
        print("Step 1: Genetic algorithm initialization...")
        initial_guess = self.genetic_algorithm_init(population_size=30, generations=50)
        initial_error = self.objective_function(initial_guess)
        print(f"GA initial error: {initial_error:.2e}")

        # Step 2: Gradient-based optimization for fine-tuning
        print("Step 2: Gradient-based optimization...")
        lower_bounds, upper_bounds = self.get_parameter_bounds()

        # Create bounds for scipy optimization
        bounds = list(zip(lower_bounds, upper_bounds))

        # Use multiple optimization methods for robustness
        methods = ["L-BFGS-B", "TNC", "SLSQP"]
        best_result = None
        best_error = float("inf")

        for method in methods:
            try:
                # Set tolerances for precision
                options = {"ftol": 1e-9, "gtol": 1e-8} if method == "L-BFGS-B" else {}

                result = optimize.minimize(
                    self.objective_function,
                    initial_guess,
                    method=method,
                    bounds=bounds,
                    options=options,
                )

                if result.success and result.fun < best_error:
                    best_result = result
                    best_error = result.fun

            except Exception as e:
                print(f"Optimization with {method} failed: {e}")
                continue

        if best_result is None:
            # Fallback to Nelder-Mead if bounded methods fail
            print("Using Nelder-Mead as fallback...")
            best_result = optimize.minimize(
                self.objective_function,
                initial_guess,
                method="Nelder-Mead",
                options={"xatol": 1e-8, "fatol": 1e-9},
            )

        # Extract final parameters
        Rs_opt, Rsh_opt, A_opt, Io_opt = best_result.x
        Iph_opt = self.Isc_stc  # Keep this approximation

        final_error = best_result.fun
        print(f"Final optimization error: {final_error:.2e}")
        print(f"Converged: {best_result.success}")

        # Create parameter object
        params = PVParameters(
            Rs=Rs_opt, Rsh=Rsh_opt, A=A_opt, Io=Io_opt, Iph=Iph_opt, error=final_error
        )

        self._print_parameters(params)

        return params

    def _print_parameters(self, params: PVParameters):
        """Print extracted parameters in a nice format."""
        print(f"\nExtracted parameters for {self.name}:")
        print(f"Series resistance (Rs): {params.Rs:.6f} Ω")
        print(f"Shunt resistance (Rsh): {params.Rsh:.1f} Ω")
        print(f"Ideality factor (A): {params.A:.6f}")
        print(f"Dark saturation current (Io): {params.Io:.2e} A")
        print(f"Photocurrent (Iph): {params.Iph:.6f} A")
        print(f"Final error: {params.error:.2e}")

    def get_iv_curve(self, voltage_range=None, irradiance=1000, temperature=25):
        """Calculate I-V curve using extracted parameters."""
        # Convert temperature to Kelvin
        T = temperature + 273.15
        G = irradiance / 1000  # Normalized irradiance

        # If voltage range not provided, create one
        if voltage_range is None:
            # Estimate Voc at the given conditions
            Voc_est = self.Voc_stc + self.kv * (temperature - 25)
            voltage_range = np.linspace(0, Voc_est * 1.1, 500)

        # Update parameters for new conditions
        Vt = self.k * T / self.q / self.ns  # Thermal voltage per cell

        # Temperature-dependent short-circuit current
        Isc_T = self.Isc_stc * (1 + self.ki * (temperature - 25))

        # Irradiance and temperature dependent photo-current
        Iph = Isc_T * G

        # Temperature-dependent dark saturation current
        # Using the relationship: Io(T) = Io(Tref) * (T/Tref)^3 * exp(Eg/k * (1/Tref - 1/T))
        # Simplified version for silicon (Eg ≈ 1.12 eV)
        Eg = 1.12 * self.q  # eV to Joules
        Io_T = (
            self.parameters.Io
            * (T / self.Tcell_stc) ** 3
            * np.exp(Eg / self.k * (1 / self.Tcell_stc - 1 / T))
        )

        # Calculate current for each voltage point
        current = np.zeros_like(voltage_range)

        for i, v in enumerate(voltage_range):
            # Define current equation to solve
            def current_equation(I):
                return I - (
                    Iph
                    - Io_T
                    * (
                        np.exp((v + I * self.parameters.Rs) / (self.parameters.A * Vt))
                        - 1
                    )
                    - (v + I * self.parameters.Rs) / self.parameters.Rsh
                )

            # Initial guess
            if i == 0:
                I_guess = Iph
            else:
                I_guess = current[i - 1]

            # Solve for current
            try:
                solution = optimize.root_scalar(
                    current_equation,
                    x0=I_guess,
                    method="brentq",
                    bracket=[0, Iph * 1.1],
                )
                current[i] = max(0, solution.root)
            except:
                current[i] = 0

        return voltage_range, current

    def get_pv_curve(self, voltage_range=None, irradiance=1000, temperature=25):
        """Calculate P-V curve."""
        V, I = self.get_iv_curve(voltage_range, irradiance, temperature)
        P = V * I
        return V, P

    def verify_extraction(self) -> dict:
        """Verify parameter extraction by checking key points."""
        verification = {}

        # Test at three key points
        test_points = [
            ("Short circuit", 0, self.Isc_stc),
            ("Open circuit", self.Voc_stc, 0),
            ("Maximum power point", self.Vmpp_stc, self.Impp_stc),
        ]

        print(f"\nVerification for {self.name}:")
        print("-" * 50)

        for name, V_test, I_expected in test_points:
            # Calculate current at this voltage
            V_array, I_array = self.get_iv_curve(voltage_range=[V_test])
            I_calculated = I_array[0]

            error = abs(I_calculated - I_expected)
            error_percent = (error / max(I_expected, 0.001)) * 100

            print(
                f"{name:20s}: Expected={I_expected:.4f}A, "
                f"Calculated={I_calculated:.4f}A, "
                f"Error={error:.4f}A ({error_percent:.2f}%)"
            )

            verification[name] = {
                "expected": I_expected,
                "calculated": I_calculated,
                "error": error,
                "error_percent": error_percent,
            }

        # Calculate power at MPP
        P_calculated = self.Vmpp_stc * verification["Maximum power point"]["calculated"]
        P_error = abs(P_calculated - self.Pmpp_stc)
        P_error_percent = (P_error / self.Pmpp_stc) * 100

        print(
            f"{'Power at MPP':20s}: Expected={self.Pmpp_stc:.2f}W, "
            f"Calculated={P_calculated:.2f}W, "
            f"Error={P_error:.2f}W ({P_error_percent:.2f}%)"
        )

        verification["Power at MPP"] = {
            "expected": self.Pmpp_stc,
            "calculated": P_calculated,
            "error": P_error,
            "error_percent": P_error_percent,
        }

        return verification

    def plot_iv_curve(self, irradiances=None, temperatures=None, figsize=(12, 6)):
        """Plot I-V and P-V curves with datasheet points marked."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

        if irradiances is not None:
            for G in irradiances:
                V, I = self.get_iv_curve(irradiance=G)
                V_pv, P = self.get_pv_curve(irradiance=G)
                ax1.plot(V, I, label=f"G = {G} W/m²")
                ax2.plot(V_pv, P, label=f"G = {G} W/m²")
        elif temperatures is not None:
            for T in temperatures:
                V, I = self.get_iv_curve(temperature=T)
                V_pv, P = self.get_pv_curve(temperature=T)
                ax1.plot(V, I, label=f"T = {T}°C")
                ax2.plot(V_pv, P, label=f"T = {T}°C")
        else:
            # Default STC conditions
            V, I = self.get_iv_curve()
            V_pv, P = self.get_pv_curve()
            ax1.plot(V, I, "b-", linewidth=2, label="Modeled")
            ax2.plot(V_pv, P, "r-", linewidth=2, label="Modeled")

            # Mark datasheet points
            ax1.plot(
                0, self.Isc_stc, "ro", markersize=8, label=f"Isc = {self.Isc_stc} A"
            )
            ax1.plot(
                self.Voc_stc, 0, "go", markersize=8, label=f"Voc = {self.Voc_stc} V"
            )
            ax1.plot(
                self.Vmpp_stc,
                self.Impp_stc,
                "mo",
                markersize=8,
                label=f"MPP ({self.Vmpp_stc}V, {self.Impp_stc}A)",
            )

            ax2.plot(
                self.Vmpp_stc,
                self.Pmpp_stc,
                "mo",
                markersize=8,
                label=f"MPP = {self.Pmpp_stc} W",
            )

        ax1.set_xlabel("Voltage [V]")
        ax1.set_ylabel("Current [A]")
        ax1.set_title(f"I-V Characteristic - {self.name}")
        ax1.grid(True, alpha=0.3)
        ax1.legend()

        ax2.set_xlabel("Voltage [V]")
        ax2.set_ylabel("Power [W]")
        ax2.set_title(f"P-V Characteristic - {self.name}")
        ax2.grid(True, alpha=0.3)
        ax2.legend()

        plt.tight_layout()
        return fig
