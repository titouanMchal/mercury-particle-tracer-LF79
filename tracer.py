import numpy as np
import matplotlib.pyplot as plt
import LF_field as lf
import functions as f


plt.rcParams.update(
    {
        "font.family": "DejaVu Serif",
        "font.size": 12,
        "axes.labelsize": 14,
        "grid.alpha": 0.3,
    }
)


RM = f.RM  # Mercury's radius (m)


class Ion:
    def __init__(self, name, charge, mass):
        """
        Instantiates an ion species

        Inputs
        ------
        name
            name of the ion (proton for example)
        charge (C)
        mass (kg)
        """
        self.name = name
        self.charge = charge
        self.mass = mass


class _StopSimulation(Exception):
    """Internal stopping criteria"""

    def __init__(self, reason: str, iteration: int):
        super().__init__(reason)
        self.reason = reason
        self.iteration = iteration


class Shoot:
    def __init__(self, ion, N: int, x0: np.ndarray, v0: np.ndarray, dt: float):
        """
        Instantiates a Shoot object (ion + trajectory + physical diagnostics)

        Inputs
        ------
        ion
            Ion object
        N
            Number of iterations
        x0
            Initial position (m)
        v0
            Initial velocity (m/s)
        dt
            Time step (s) if not adaptive
        """
        self.ion = ion
        self.N = N
        self.dt = dt
        self.t = np.zeros(N)
        self.x = np.zeros((N, 3))
        self.v = np.zeros((N, 3))
        self.x[0] = x0
        self.v[0] = v0
        self.x2 = np.zeros((N, 3))
        self.x3 = np.zeros((N, 3))
        self.e1 = 0
        self.e2 = 0
        self.Tc = np.zeros(N)
        self.list_dt = np.zeros(N)
        self.p = 0
        self.Lp = np.zeros(N)
        self.mu = np.zeros(N)
        self.pitch = np.zeros(N)
        self.B = np.zeros((N, 3))
        self.E = np.zeros((N, 3))
        self.latp = np.zeros(N)
        self.scalar = np.zeros(N)
        self.etot = 0  # total energy (should remain constant)
        self.E_para = np.zeros(N)
        self.E_ortho = np.zeros(N)

    def _boris_step(self, i, B0, E0, direction):
        """
        Computes one iteration with Boris method

        Inputs
        ------
        i : iteration number
        B0 : B field vector
        E0 : E field vector
        direction : +1 / -1  (forward, backward)
        """
        self.v[i] = (
            self.v[i - 1]
            + direction * 0.5 * self.ion.charge * E0 / self.ion.mass * self.dt
        )
        t = self.ion.charge * self.dt * B0 / 2 / self.ion.mass
        v1 = self.v[i] + direction * f.cross(self.v[i], t)
        self.v[i] = self.v[i] + direction * f.cross(
            v1, 2 / (1 + np.linalg.norm(t) ** 2) * t
        )
        self.v[i] = (
            self.v[i]
            + direction * self.dt * 1 / 2 * self.ion.charge * E0 / self.ion.mass
        )
        self.x[i] = self.x[i - 1] + direction * self.dt * self.v[i]

    def _print_progress(self, i, kinetic_energy, nB0):
        t_min = self.t[i] / 60
        keV = kinetic_energy / 1.6e-19 / 1e3
        B_nT = nB0 * 1e9

        text = (
            f"Time of flight: {t_min:.2f} min ; "
            f"Niter = {i}/{self.N} ; "
            f"Kinetic energy = {keV:.3f} keV ; "
            f"B = {B_nT:.0f} nT"
        )
        print(f"\r\033[K{text}", end="", flush=True)

    def compute_trajectory_boris(
        self,
        E: callable = lf.E_LF,
        B: callable = lf.B_LF,
        precision=360,
        adaptative_dt=False,
        direction=1,
        shift=True,
        dse=1e4,
        pdrop=10e3,
        energy_corrector=False,
    ):
        """
        Integration routine to compute the trajectory of a charged particle using the Boris-Buneman integrator.

        This method propagates the position and velocity of the particle over time in an electromagnetic field
        using a symplectic and energy-conserving scheme.

        Parameters
        ----------
        E : bool or callable
            If False, the electric field is ignored. If callable, it should return the electric field at a given position.
        B : callable
            Function that takes a 3D position and returns the magnetic field vector.
        precision : int, optional
            Number of time steps per gyration period if "adaptative_dt" is True. Default is 360.
        adaptative_dt : bool, optional
            Whether to adapt the time step based on local magnetic field strength. Default is False. Not recommended with Boris algorithm
        direction : int, optional
            Integration direction: +1 (forward in time), or -1 (backward). Default is 1.
        shift : bool, optional
            True if B field is shifted to the North (more realistic)
        dse : float, optional
            Step size (in meters) for electric field computation. Default is 1E4.
        pdrop : float, optional
            Electrostatic potential drop (in volts) used in electric field model. Default is 10E3.
        energy_corrector : bool, optional
            Whether to apply an energy conservation correction at each electric field update. Default is False.
        dt : float, optional
            Fixed time step for integration (in seconds). Ignored if `adaptative_dt` is True. Default is 1E-3. Recommended for Boris method

        Notes
        -----
        - The method modifies the following internal arrays: self.x, self.v, self.t, self.B, self.E, self.mu, etc.
        - It stops the simulation early if the particle crosses boundaries or violates energy conservation.
        """
        curvilinear_distance = dse

        n_print = np.floor(self.N / 400)

        if E != False:
            self.p, self.e1, self.e2, E_0 = lf.E_LF(self.x[0], shift=shift, pdrop=pdrop)

        self.etot = (
            self.p * self.ion.charge
            + 1 / 2 * self.ion.mass * np.linalg.norm(self.v[0]) ** 2
        )

        for i in range(1, self.N):

            self.B[i - 1] = B(self.x[i - 1], shift=shift)

            nB0 = np.linalg.norm(self.B[i - 1])

            if adaptative_dt:
                dt = 2 * np.pi * self.ion.mass / self.ion.charge / nB0 / precision
                if dt < 1e-2:
                    self.dt = dt

            self.t[i] = self.t[i - 1] + self.dt

            self.list_dt[i - 1] = self.dt

            if E != False:
                if curvilinear_distance >= dse:

                    result = lf.E_LF(self.x[i - 1], shift=shift, pdrop=pdrop)

                    if result is None:
                        print("\nSimulation ended : particle reached MP")
                        break

                    self.p, self.e1, self.e2, E0 = result
                    curvilinear_distance = 0

                if curvilinear_distance < dse:
                    u1, u2 = f.vec_ortho(self.B[i - 1])
                    self.E[i - 1] = self.e1 * u1 + self.e2 * u2

            else:
                self.E[i - 1] = np.zeros(3)

            if energy_corrector and curvilinear_distance == 0:
                kinetic_energy = (
                    1 / 2 * self.ion.mass * (np.linalg.norm(self.v[i - 1])) ** 2
                )
                delta_et = kinetic_energy + self.p * self.ion.charge - self.etot
                vnew = np.sqrt(np.abs(2 * (kinetic_energy - delta_et) / self.ion.mass))
                self.v[i - 1] = vnew * self.v[i - 1] / np.linalg.norm(self.v[i - 1])

            self._boris_step(i, B0=self.B[i - 1], E0=self.E[i - 1], direction=direction)

            v_drift = np.cross(self.E[i - 1], self.B[i - 1]) / nB0**2
            u1, u2 = f.vec_ortho(self.B[i - 1])
            ec_ortho = (
                1
                / 2
                * self.ion.mass
                * (
                    np.dot(self.v[i - 1] - v_drift, u1) ** 2
                    + np.dot(self.v[i - 1] - v_drift, u2) ** 2
                )
            )
            self.mu[i - 1] = ec_ortho / nB0

            self.Lp[i - 1] = self.p
            self.Tc[i - 1] = 2 * np.pi * self.ion.mass / self.ion.charge / nB0

            if np.linalg.norm(self.x[i]) < RM:
                print("\nSimulation ended : particle reached planet's surface")
                break

            if self.x[i][0] > 6 * RM:

                print("\nSimulation ended: x > 6 RM ")
                print(
                    "If you want to continue integration : change stopping criteria in compute_trajectory_boris"
                )
                break

            kinetic_energy = (
                1 / 2 * self.ion.mass * (np.linalg.norm(self.v[i - 1])) ** 2
            )
            delta_e = kinetic_energy + self.p * self.ion.charge - self.etot

            if np.abs(delta_e) / (self.ion.charge * 20e3) > 0.01:
                print("\nSimulation ended : total energy error exceeds 1% ")
                print(
                    "If you want to continue integration : try ajusting dse parameter, use energy corrector or relax tolerance in compute_trajectory_boris"
                )
                break

            if i % (self.N / n_print) == 0:
                self._print_progress(i, kinetic_energy=kinetic_energy, nB0=nB0)

            curvilinear_distance += np.linalg.norm(self.x[i] - self.x[i - 1])

        index = np.any(self.x != 0, axis=1)
        for attr in ["x", "t", "v", "Tc", "list_dt", "Lp", "mu", "B", "E"]:
            setattr(self, attr, getattr(self, attr)[index])

    def _prepare_ax(self, ax, xlabel, ylabel, title=None, logy=False):
        if ax is None:
            fig, ax = plt.subplots()
        if logy:
            ax.set_yscale("log")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        if title is not None:
            ax.set_title(title)
        ax.grid(True)
        return ax

    def plot_trajectory(self, ax=None, color="red", label=None, projection="XZ", w=1):
        if projection == "XZ":
            ax.plot(
                self.x[:, 0] / f.RM,
                self.x[:, 2] / f.RM,
                color=color,
                zorder=4,
                label=label,
                linewidth=w,
            )
        if projection == "XY":
            ax.plot(
                self.x[:, 0] / f.RM,
                self.x[:, 1] / f.RM,
                color=color,
                zorder=4,
                label=label,
                linewidth=w,
            )
        if label is not None:
            ax.legend()
        return ax

    def plot_kinetic_energy(self, ax=None, color="red", label=None, w=1, title=None):
        t_min = self.t / 60
        ec_keV = (
            0.5 * self.ion.mass * np.linalg.norm(self.v, axis=1) ** 2 / 1.6e-19 / 1e3
        )
        ax = self._prepare_ax(ax, "t (min)", "T (keV)", logy=True, title=title)
        ax.plot(t_min, ec_keV, color=color, label=label, linewidth=w)
        if label is not None:
            ax.legend()
        return ax

    def plot_total_energy(self, ax=None, w=1, title=None):
        t_min = self.t / 60
        kinetic_energy_keV = (
            0.5 * self.ion.mass * np.linalg.norm(self.v, axis=1) ** 2 / 1.6e-19 / 1e3
        )
        ep_keV = self.ion.charge * self.Lp / 1.6e-19 / 1e3
        ax = self._prepare_ax(ax, "t (min)", "E (keV)", title=title)
        ax.plot(t_min, kinetic_energy_keV, color="red", label="kinetic", linewidth=w)
        ax.plot(t_min[:-1], ep_keV[:-1], color="blue", label="potential", linewidth=w)
        ax.plot(
            self.t[:-1] / 60,
            (ep_keV[:-1] + kinetic_energy_keV[:-1]),
            color="gray",
            label="total energy",
            linewidth=w,
        )
        ax.legend()
        return ax

    def plot_mu(self, ax=None, color="red", label=None, w=1, title=None):
        t_min = self.t[:-1] / 60
        mu_rel = self.mu[:-1] / self.mu[0]
        ax = self._prepare_ax(ax, "t (min)", r"$\mu/\mu_0$", logy=True, title=title)
        ax.plot(t_min, mu_rel, color=color, label=label, linewidth=w)
        if label is not None:
            ax.legend()
        return ax

    def plot_cyclotron_period(self, ax=None, color="red", label=None, w=1, title=None):
        t = self.t[:-1] / 60
        Tc = self.Tc[:-1]
        ax = self._prepare_ax(ax, "t (min)", "Tc (s)", title=title)
        ax.plot(t, Tc, color=color, label=label, linewidth=w)
        if label:
            ax.legend()
        return ax

    def plot_dt(self, ax=None, color="red", label=None, w=1, title=None):
        t = self.t[:-1] / 60
        dt = self.list_dt[:-1]
        ax = self._prepare_ax(ax, "t (min)", "dt (s)")
        ax.plot(t, dt, color=color, label=label, linewidth=w, title=title)
        if label:
            ax.legend()
        return ax

    def plot_potential(self, ax=None, color="red", label=None, w=1, title=None):
        t = self.t[:-1] / 60
        phi = self.Lp[:-1] / 1e3
        ax = self._prepare_ax(ax, "t (min)", r"$\phi$ (kV)", title=title)
        ax.plot(t, phi, color=color, label=label, linewidth=w)
        if label:
            ax.legend()
        return ax

    def plot_B(self, axes=None, color="red", w=1.2):
        t = self.t[:-1] / 60
        data = self.B[:-1] * 1e9
        if axes is None:
            fig, axes = plt.subplots(3, 1, sharex=True, figsize=(6, 8))
        else:
            fig = axes[0].get_figure()
        labels = ["Bx (nT)", "By (nT)", "Bz (nT)"]
        for i, ax in enumerate(axes):
            ax.plot(t, data[:, i], color=color, linewidth=w)
            ax.set_ylabel(labels[i])
            ax.grid(True)
        axes[-1].set_xlabel("t (min)")
        return fig, axes

    def plot_E(self, axes=None, color="red", w=1.3):
        t = self.t[:-1] / 60
        data = self.E[:-1] * 1e3
        if axes is None:
            fig, axes = plt.subplots(3, 1, sharex=True, figsize=(6, 8))
        else:
            fig = axes[0].get_figure()
        labels = ["Ex (mV/m)", "Ey (mV/m)", "Ez (mV/m)"]
        for i, ax in enumerate(axes):
            ax.plot(t, data[:, i], color=color, linewidth=w)
            ax.set_ylabel(labels[i])
            ax.grid(True)
        axes[-1].set_xlabel("t (min)")
        return fig, axes


def push(
    particle: Ion,
    x0_RM: np.ndarray,
    kinetic_energy_eV: float,
    pitch_deg: float,
    phase_deg: float = 0,
    E_field: bool = False,
    dse: float = 5e4,
    shift: bool = True,
    dt: float = 1e-3,
    n_iter: int = 2000000,
    phi_drop_V: float = 20e3,
    adaptive_dt: bool = False,
    direction: int = 1,
    energy_corrector: bool = False,
) -> Shoot:
    """
    Initialize and run a Boris integrator for an ion in Mercury's magnetosphere.

    Parameters
    ----------
    particle
        An Ion instance (with attributes: name, charge [C], mass [kg]).
    x0_RM
        Initial position in Mercury radii (RM).
    kinetic_energy_eV
        Initial kinetic energy in electron-volts.
    pitch_deg
        Pitch angle in degrees (angle between velocity and magnetic field).

    Other Parameters
    ----------------
    phase_deg : float, optional
        Initial gyrophase in degrees. Default is 0.
    E_field : bool, optional
        Whether to include the electric field from LF_field.E_LF. Default is False.
    dse : float, optional
        Step size (in meters) used when computing the electric field. Default is 5e4.
    z_shift : float, optional
        Z-offset parameter for the LF_field model. Default is 0.196.
    dt : float, optional
        Fixed time step in seconds (ignored if adaptive_dt=True). Default is 1e-3.
    n_iter : int, optional
        Maximum number of integration steps. Default is 2,000,000.
    phi_drop_V : float, optional
        Potential drop parameter for LF_field, in volts. Default is 20e3.
    adaptive_dt : bool, optional
        If True, the time step is adjusted each iteration based on the cyclotron period. Default is False. Should not be used here.
    direction : int, optional
        +1 to integrate forward in time, -1 to integrate backward. Default is +1.
    energy_corrector : bool, optional
        If True, applies an energy correction at each step. Default is False.

    Returns
    -------
    Shoot
        A Shoot object containing the time history of position, velocity, and derived quantities.
    """

    x0 = x0_RM * f.RM
    phase = np.deg2rad(phase_deg)
    pitch = np.deg2rad(pitch_deg)
    kinetic_energy = kinetic_energy_eV * 1.6e-19

    b0 = lf.B_LF(x0_RM, shift=shift)
    u1, u2 = f.vec_ortho(lf.B_LF(x0_RM, shift=shift))
    vpar = np.sqrt(2 * kinetic_energy / particle.mass) * np.cos(pitch)
    vper = np.sqrt(2 * kinetic_energy / particle.mass) * np.sin(pitch)
    v1 = -vper * np.sin(phase)
    v2 = -vper * np.cos(phase)
    v0 = v1 * u1 + v2 * u2 + vpar * b0 / np.linalg.norm(b0)

    shoot = Shoot(ion=particle, N=n_iter, x0=x0, v0=v0, dt=dt)

    shoot.compute_trajectory_boris(
        E=E_field,
        B=lf.B_LF,
        precision=360,
        adaptative_dt=adaptive_dt,
        shift=shift,
        dse=dse,
        pdrop=phi_drop_V / 2,
        energy_corrector=energy_corrector,
        direction=direction,
    )

    return shoot
