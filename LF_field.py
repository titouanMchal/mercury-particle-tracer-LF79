import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import functions as f


RM = f.RM


def B_LF(
    coord: np.ndarray,
    shift: bool = True,
    M: float = 260e-9 * RM**3,
    BT: float = 70e-9,
    L: float = 0.5 * RM,
):
    """
    Luhmann-Friesen model (1979) - adapted to Mercury
    Warning: if you change the magnetic field settings and use the electric field,
    make sure you calculate the new reversal latitude with find_reversal_LF() and update the parameters in E_LF

    Parameters
    ----------
    coord : np.ndarray
        Position vector
    M : float
        Reduced dipole moment (T.m3)
    BT : float
        Asymptotic tail field (T)
    L : float
        Current sheet half-thickness (m)

    Outputs
    -------
    np.ndarray
        The 3-component [Bx, By, Bz] in simulation frame
    """

    x, y, z = coord[0], coord[1], coord[2]

    if shift:
        z = z - 0.196 * RM
    r, fi, lat1 = f.cartesian_to_spherical(x, y, z)
    Br = -2 * M / r**3 * np.sin(lat1)
    Bl = M * np.cos(lat1) / r**3
    Bx = (
        Br * np.cos(fi) * np.cos(lat1)
        - Bl * np.cos(fi) * np.sin(lat1)
        - BT * np.tanh(z / L)
    )
    By = Br * np.sin(fi) * np.cos(lat1) - Bl * np.sin(fi) * np.sin(lat1)
    Bz = Br * np.sin(lat1) + Bl * np.cos(lat1)
    return np.array([Bx, By, Bz])


def projection_LF(x0, rstop):
    """
    Trace a magnetic field line from the starting point x0 until it
    intersects the sphere of radius rstop.

    Parameters
    ----------
    x0 : np.ndarray
        Initial 3D position vector (meters).
    rstop : float, optional
        Radius of the target sphere (meters).

    Returns
    -------
    np.ndarray or None
        The 3-component position vector at the intersection point on the sphere,
        or None if the solver fails or the sphere is never reached.
    """
    x = x0.copy()
    x[2] = np.abs(x[2])

    if np.linalg.norm(x) < rstop:
        return x

    def fun(t, x):
        return B_LF(x, shift=False)

    def hit_sphere(t, x):
        return np.linalg.norm(x) - rstop

    hit_sphere.terminal = True
    hit_sphere.direction = 0

    events = hit_sphere

    try:
        sol = solve_ivp(
            fun,
            (0, 1e15),
            x,
            method="RK45",
            max_step=1e13,
            events=hit_sphere,
        )
    except Exception:
        return None

    if sol.success and sol.t_events[0].size > 0:
        return sol.y_events[0][0]

    return None


def read_potential_LF(coord: np.ndarray, lat_r: float, rstop: float, pdrop: float):
    """
    Computes electrostatic potential for a given position

    Inputs
    --------
    coord
        position vector
    lat_r
        reversal boundary latitude : make sure you compute it each time you modify B_LF parameters
    rstop
        projection surface radii in m
    pdrop
        half potential drop

    Returns
    ---------
    p
        local electrostatic potential value (V)
    lat
        projection's latitude (rad)
    fi
        projection's longitude (rad)

    """

    theta0 = np.deg2rad(1)

    x_proj = projection_LF(coord, rstop=rstop)

    if x_proj is None:
        return None

    x, y, z = projection_LF(coord, rstop=rstop)
    r, fi_proj, lat_proj = f.cartesian_to_spherical(x, y, z)

    theta_r = np.pi / 2 - lat_r
    theta1 = theta_r - theta0 / 2
    theta2 = theta_r + theta0 / 2
    theta = np.pi / 2 - lat_proj

    p1 = np.sin(theta) / np.sin(theta_r)
    p2 = 1 / p1**4
    p = p1

    if theta > theta2:
        p = p2

    if theta >= theta1 and theta <= theta2:
        delta = theta - theta1
        pf = (
            10 * (delta / theta0) ** 3
            - 15 * (delta / theta0) ** 4
            + 6 * (delta / theta0) ** 5
        )
        p = p1 + pf * (p2 - p1)

    return pdrop * p * np.sin(fi_proj)


def E_LF(coord: np.ndarray, shift: bool, pdrop: float):
    """
    Computes electrostatic potential, E ortho components, E vector at point coord

    Inputs
    ------
    coord
        position vector
    shift
        True if B_LF is shifted to the North
    pdrop
        half electrostatic potential drop (V)

    Outputs
    ------
    p0
        local potential
    e1, e2
        E orthogonal components (e1 is in the XZ plane)
    E
        electric field vector (V/m)
    """
    nB = np.linalg.norm(B_LF(coord, shift))
    dist_neighbors = 5e-2 / nB

    if shift:
        lat_r, rstop = 52.3986 * np.pi / 180, 0.804 * RM
        coord0 = coord.copy()
        coord0[2] -= 0.196 * RM

    else:
        lat_r, rstop = 47.94 * np.pi / 180, RM
        coord0 = coord

    u1, u2 = f.vec_ortho(B_LF(coord0, shift=shift))
    v1, v2 = f.neigbors(u1, u2, coord0, dist_vec=dist_neighbors)

    p0 = read_potential_LF(coord0, lat_r, rstop, pdrop)
    p1 = read_potential_LF(v1, lat_r, rstop, pdrop)
    p2 = read_potential_LF(v2, lat_r, rstop, pdrop)

    if None in (p0, p1, p2):
        return None

    e2 = (p0 - p2) / dist_neighbors
    e1 = (p0 - p1) / dist_neighbors
    E = e1 * u1 + e2 * u2
    E = np.array(E)

    return p0, e1, e2, E


def find_reversal_LF(
    niter: int = 20,
    crit: float = 1e-6,
    r_begin: float = RM * (1 - 0.196),
    M: float = 260e-9 * RM**3,
    BT: float = 70e-9,
    L_sheet: float = 0.5 * RM,
):
    """
    Finds reversal boundary latitude (dichotomous search) - use it when changing B_LF field parameters

    Inputs
    ------
    niter
        max number of iterations
    crit
        stopping criterion
    shift
        True if B_LF shifted to the North
    M : float
        Reduced dipole moment (T.m3)
    BT : float
        Asymptotic tail field (T)
    L : float
        Current sheet half-thickness (m)

    Outputs
    -------
    lat
        reversal latitude

    """
    from time import time

    print("\nSearching for reversal boundary")
    print("Starting dichotomous search : ")
    t0 = time()
    lat0 = 0
    lat1 = np.pi / 2
    lat = (lat0 + lat1) / 2
    L = np.zeros((niter, 3))
    list_lat = []
    list_lat.append(lat)
    print("init.  ;  lat = {}°  ".format(np.round(lat * 180 / np.pi, 4)))
    for i in range(niter):
        x = np.array([-r_begin * np.cos(lat), 0, r_begin * np.sin(lat)])
        L[i] = x

        def func(t, x):
            return -B_LF(x, shift=False, L=L_sheet, M=M, BT=BT)

        def event1(t, x):
            return x[2]

        def event2(t, x):
            return x[0]

        events = [event1, event2]

        event1.terminal = True
        event2.terminal = True

        sol = solve_ivp(
            func,
            y0=x,
            t_span=[0, 1e20],
            max_step=1e11,
            events=events,
            method="RK45",
        )
        x = sol.y[:, -1]

        if x[0] >= -0.01 * RM:
            lat1 = lat
            lat = (lat + lat0) / 2

        if x[2] <= 0.01 * RM:
            lat0 = lat
            lat = (lat + lat1) / 2

        list_lat.append(lat)
        diff = np.abs((list_lat[i + 1] - list_lat[i]) / list_lat[i + 1] * 100)
        print(
            "iter. {}  ;  lat = {}°  ;  relat. diff = {} %".format(
                i + 1, np.round(np.rad2deg(lat), 4), np.round(diff, 4)
            )
        )

        if diff < crit:
            print("tolerance reached")
            index = np.any(L != 0, axis=1)
            L = L[index]
            break

    print(
        "Dichotomous search completed in {} s -->  reversal latitude = {}° ".format(
            np.round(time() - t0), np.rad2deg(lat)
        )
    )

    return lat


def integrate_B_LF_line(
    x0: np.ndarray, B=B_LF, z_shift: float = 0.196, xmin: float = -5, xmax: float = 6
):

    dir = 1 if x0[2] < z_shift * RM else -1

    def fun(t, x):
        return B_LF(x, shift=z_shift) * dir

    def event(t, x):
        return np.linalg.norm(x) - 0.99999 * RM

    def event2(t, x):
        return np.abs(x[2]) - 3.5 * RM

    def eventbis(t, x):
        return x[2] - (z_shift + 0.001) * RM

    def event3(t, x):
        return np.abs(x[1]) - 3.2 * RM

    def event4(t, x):
        return x[0] - xmax * RM

    def event5(t, x):
        return x[0] - xmin * RM

    event.terminal = True
    event2.terminal = True
    event3.terminal = True
    event4.terminal = True
    event5.terminal = True
    eventbis.terminal = True
    events = [event, event, event2, event3, event4, event5, eventbis]

    max_step = 1e12
    sol = solve_ivp(
        fun, y0=x0, t_span=[0, 1e17], max_step=max_step, events=events, method="RK45"
    )

    return sol.y[:, -1], sol.y.T


def drawing_planet(
    dayside_color: str = "lightgray",
    shift: bool = True,
    projection: str = "XZ",
    xmin: float = -1.6,
    xmax: float = 6,
    ymax: float = 3,
    zmax: float = 3,
    plot_magnitude: bool = False,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Draws Mercury and (optionally) a background B-field magnitude or field lines
    in either the XZ or XY plane.

    Parameters
    ----------
    dayside_color
        Fill color for the planet’s dayside
    shift
        Whether to apply the LF_field z-shift when calling B_LF.
    projection
        "XZ" or "XY" projection.
    xmin, xmax
        X-axis limits in RM units.
    ymax
        Y-axis limit for the XY projection.
    zmax
        Z-axis limit for the XZ projection.
    plot_magnitude
        If True, draw a log-scaled background map of |B|.

    Returns
    -------
    fig, ax
        The Matplotlib figure and axis.
    """

    from matplotlib.patches import Circle
    import matplotlib.ticker as ticker
    from matplotlib.colors import LogNorm

    fig, ax = plt.subplots()
    res = 100

    if shift:
        z_shift = 0.196
    else:
        z_shift = 0

    if projection == "XZ":

        x = np.linspace(xmin * RM, xmax * RM, res)
        y = np.zeros(res)
        z = np.linspace(-zmax * RM, zmax * RM, res)
        X, Z = np.meshgrid(x, z)
        Bx, By, Bz = B_LF([X, y, Z], shift=shift)

        if plot_magnitude:
            magn = np.sqrt(Bx**2 + Bz**2 + By**2)
            im = ax.imshow(
                magn * 1e9,
                extent=[x.min() / RM, x.max() / RM, z.min() / RM, z.max() / RM],
                origin="lower",
                cmap="viridis",
                norm=LogNorm(vmax=1e3),
                interpolation="bilinear",
            )
            cbar = fig.colorbar(im, ax=ax)
            cbar.set_label(
                r"$\Vert \vec{B} \Vert$ (nT)", fontsize=15, fontname="DejaVu Serif"
            )
            cbar.ax.tick_params(labelsize=17)

        for lat in np.linspace(0.1, 0.98, 19) * np.pi / 2:
            for long in np.array([0, np.pi]):
                x0 = x0 = f.spherical_to_cartesian(RM, long, lat)
                x0 = np.array(x0)
                proj_x0, L0 = integrate_B_LF_line(
                    x0=x0, B=B_LF, z_shift=z_shift, xmax=xmax
                )
                L0 = np.array(L0)
                ax.plot(
                    L0[:, 0] / RM,
                    L0[:, 2] / RM,
                    color="black",
                    zorder=0.5,
                    linewidth=0.8,
                    alpha=0.3,
                )
                ax.plot(
                    L0[:, 0] / RM,
                    2 * z_shift - L0[:, 2] / RM,
                    color="black",
                    zorder=0.5,
                    linewidth=0.8,
                    alpha=0.3,
                )

        ax.set_xlabel("X ($R_M$)", fontsize=18, fontname="DejaVu Serif")
        ax.set_ylabel("Z ($R_M$)", fontsize=18, fontname="DejaVu Serif")
        ax.set_aspect("equal")
        ax.set_xlim(-1.6, xmax)
        ax.set_ylim(-zmax, zmax)

    if projection == "XY":

        if plot_magnitude:
            x = np.linspace(xmin * RM, xmax * RM, res)
            y = np.linspace(-ymax * RM, ymax * RM, res)
            z = np.zeros(res)
            X, Y = np.meshgrid(x, y)
            Bx, By, Bz = B_LF([X, Y, z], shift=shift)
            magn = np.sqrt(Bx**2 + Bz**2 + By**2)
            im = ax.imshow(
                magn * 1e9,
                extent=[x.min() / RM, x.max() / RM, y.min() / RM, y.max() / RM],
                origin="lower",
                cmap="viridis",
                norm=LogNorm(vmax=1e3),
                interpolation="bilinear",
            )
            cbar = fig.colorbar(im, ax=ax)
            cbar.set_label(
                r"$\Vert \vec{B} \Vert$ (nT)", fontsize=15, fontname="DejaVu Serif"
            )
            cbar.ax.tick_params(labelsize=17)

        ax.set_xlabel("X ($R_M$)", fontsize=18, fontname="DejaVu Serif")
        ax.set_ylabel("Y ($R_M$)", fontsize=18, fontname="DejaVu Serif")
        ax.set_xlim(-1.6, xmax)
        ax.set_ylim(-ymax, ymax)
        ax.set_aspect("equal")

    ax.add_patch(Circle((0, 0), 1, fill=True, color="black", zorder=1.5))
    angles = np.linspace(np.radians(90), np.radians(270), 100)
    x = np.cos(angles)
    y = np.sin(angles)
    ax.fill_betweenx(y, x, 0, color=dayside_color, zorder=2)
    fig.set_size_inches(8, 5)
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(2))
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(2))
    ax.tick_params(which="minor", length=4, color="gray")

    return fig, ax


if __name__ == "__main__":

    find_reversal_LF(r_begin=0.804 * RM, M=260e-9 * RM**3)
