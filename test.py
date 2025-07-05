import tracer as tr
import matplotlib.pyplot as plt
import numpy as np
import LF_field as lf
import functions as f

# Test particle tracing code adapted from D. Delcourt Fortran code
# Uses Luhman-Friesen magnetic field model (1979, adapted to Mercury) and a Volland-Stern distribution for the electrostatic potential
# This model is very simplified and will be replaced in future versions.

# This script simulates charged‑particle trajectories in Mercury’s magnetosphere using the Boris–Buneman integrator
# Warning : in the simulation frame : X axis points in anti-solar direction, Z axis northwards !
# You can trace particles (e.g. Na⁺, protons) from an initial position and pitch angle,
# then visualize their paths and diagnostics (energy, magnetic moment, electric field).

# Required python packages: numpy, scipy, matplotlib

# Electric field calculations may fail when reaching magnetopause --> halts the simulation
# If the integrator stops early, check the console messages for boundary crossings or energy errors
# Adjust dt, dse, n_iter to balance accuracy vs. run time
# Important : the use of an adaptive time step is not recommended when using the Boris-Buneman algorithm : set adaptive_dt to False

# Quick user guide :
#   1) Instantiate your particles
#   2) Call push to compute a trajectory
#   3) Plots trajectories and physical diagnostics

# Here is a minimalist example :

sodium = tr.Ion(name="Na+", charge=1.6e-19, mass=23 * 1.67252e-27)
proton = tr.Ion(name="proton", charge=1.6e-19, mass=1.672649e-27)

# display options
plot_trajectory = True
plot_physical_diagnostic = True


# sodium ion from the cusp
shoot1 = tr.push(
    particle=sodium,
    x0_RM=np.array(f.spherical_to_cartesian(1.001, np.pi, np.deg2rad(75))),
    kinetic_energy_eV=10,
    pitch_deg=170,
    dt=1e-2,
    adaptive_dt=False,
    direction=1,
    shift=True,
    n_iter=300000,
    E_field=True,
    phi_drop_V=20e3,
    dse=1e4,
    energy_corrector=True,
    phase_deg=0,
)

# proton in the ring current - no electric field
shoot2 = tr.push(
    particle=proton,
    x0_RM=np.array([1.2, 0, 0.196]),
    kinetic_energy_eV=10e3,
    pitch_deg=95,
    dt=1e-2,
    adaptive_dt=False,
    direction=1,
    shift=True,
    n_iter=27000,
    E_field=False,
    phi_drop_V=20e3,
    dse=2e3,
    energy_corrector=False,
    phase_deg=0,
)

# sodium ion from the cusp - no electric field
shoot3 = tr.push(
    particle=sodium,
    x0_RM=np.array(f.spherical_to_cartesian(1.001, np.pi, np.deg2rad(75))),
    kinetic_energy_eV=10,
    pitch_deg=170,
    dt=1e-2,
    adaptive_dt=False,
    direction=1,
    shift=True,
    n_iter=300000,
    E_field=False,
    phi_drop_V=20e3,
    dse=1e4,
    energy_corrector=False,
    phase_deg=0,
)

if plot_trajectory:
    fig, ax = lf.drawing_planet(shift=False, projection="XZ", xmax=5)
    shoot1.plot_trajectory(ax, color="red", label="sodium, cusp")
    shoot2.plot_trajectory(ax, color="blue", label="proton, RC, E=0")
    shoot3.plot_trajectory(ax, color="green", label="sodium, cusp, E=0")

    fig, ax = lf.drawing_planet(shift=False, projection="XY", ymax=2, xmax=3)
    shoot1.plot_trajectory(ax, color="red", label="sodium, cusp", projection="XY")
    shoot2.plot_trajectory(ax, color="blue", label="proton, RC, E=0", projection="XY")
    shoot3.plot_trajectory(
        ax, color="green", label="sodium, cusp, E=0", projection="XY"
    )

if plot_physical_diagnostic:

    # kinetic energy vs time
    fig, ax = plt.subplots(figsize=(8, 6))
    shoot1.plot_kinetic_energy(ax=ax, color="red", label="sodium, cusp")
    shoot2.plot_kinetic_energy(ax=ax, color="blue", label="proton, RC, E=0")

    # magnetic moment vs time
    fig, ax = plt.subplots(figsize=(8, 8))
    shoot1.plot_mu(ax=ax, color="red", label="sodium, cusp")
    shoot2.plot_mu(ax=ax, color="blue", label="proton, RC")

    # magnetic field components vs time
    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(9, 5))
    shoot1.plot_B(axes=axes, color="red")
    shoot2.plot_B(axes=axes, color="blue")

    # electric field components vs time
    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(10, 8))
    shoot1.plot_E(axes=axes, color="red")
    shoot2.plot_E(axes=axes, color="blue")

    # gyroperiod vs time
    fig, ax = plt.subplots(figsize=(8, 8))
    shoot1.plot_cyclotron_period(ax=ax, color="red", label="sodium, cusp")
    shoot2.plot_cyclotron_period(ax=ax, color="blue", label="proton, RC")

    # total energy vs time
    fig, ax = plt.subplots(figsize=(8, 8))
    shoot1.plot_total_energy(
        ax=ax, title="sodium ion from the cusp with electric field"
    )

plt.show()
