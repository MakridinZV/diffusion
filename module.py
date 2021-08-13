from typing import Callable
from typing import Tuple
import numpy as np
import matplotlib.pyplot as plt

def step_function(x: float, step_coordinate: float = 0, value_1: float = 0, value_2: float = 1) -> float:
    """Simple step function.

    Parameters
    ----------
    x : float
        Coordinate.
    step_coordinate: float
        Step coordiante.
    value_1 : float
        Function returns value_1 if x < step_coordinate.
    value_2 : float
        Function returns value_2 if x >= step_coordinate.

    Returns
    -------
    float

    """
    return value_1 if x < step_coordinate else value_2


def sweep(
    F: float,
    Nx: int,
    b: np.ndarray,
) -> np.ndarray:
    A = np.zeros((Nx + 1, Nx + 1))  # matrix for the linear system solving (add EXACTLY TWO spaces before #)
    u = np.zeros(Nx + 1)

    for i in range(1, Nx):
        A[i, i - 1] = - F
        A[i, i + 1] = - F
        A[i, i] = 1 + 2 * F

    A[0, 0] = A[Nx, Nx] = 1 + 2 * F
    A[0, 1] = A[Nx, Nx - 1] = -2 * F

    u[:] = np.linalg.solve(A, b)
    return u


def solveSYS1(
    L: float,
    F: float,
    dt: float,
    D: float,
    coefficients: Tuple[float, ...],  # Simply pass coefficients to function as (k1, k2, k3, k4).
    u0: float,
    pn: np.ndarray,
    vn: np.ndarray,
    un: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Solver for equation ....

    Parameters
    ----------
    initial_data_func : Callable[[float], float]
        Function for generation initial data.
    L : float
        L
    F : float
        T
    dt : float
        dt
    D : float
        D
    coefficients : Tuple[float, ...]
        Coefficients of ...
    u0 : float
        u0
    p0 : float
        p0
    v0 : float
        v0
    pn : np.ndarray
        pn
    vn : np.ndarray
        vn
    un : np.ndarray
        un

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
    """
    dx = np.sqrt(D * dt / F)  # step size in space
    Nx = int(round(L / dx))  # amount of nodes in space

    # p, v, u are current time layer function values
    # bp, bv, bu are previous time layer function values
    p, v, u, bp, bv, bu = [np.zeros(Nx + 1) for _ in range(6)]

    k1, k2, k3, k4, k5, k6, k7, k8, k9 = coefficients  # Unpack tuple.
    for i in range(0, Nx + 1):
        bu[i] = un[i] - dt * k4 * un[i] + dt * (k1 + k2 * vn[i] + k3 * vn[i] ** 2) * (u0 - un[i])
        bv[i] = vn[i] - dt * k9 * vn[i] + dt * (k5 * un[i] + k6 * vn[i] + k7 * vn[i] ** 2 + k8 * vn[i] ** 3) * pn[i]
        bp[i] = pn[i] - dt * (k5 * un[i] + k6 * vn[i] + k7 * vn[i] ** 2 + k8 * vn[i] ** 3) * pn[i]

    p = sweep(F, Nx, bp)
    v = sweep(F, Nx, bv)
    u = sweep(F, Nx, bu)
    return p, u, v


def solveSYS2(
    initial_data: Tuple[np.ndarray, ...],
    L: float,
    dx: float,
    dt: float,
    coefficients: Tuple[float, ...],
) -> Tuple[np.ndarray, ...]:
    # TODO: Write docs.
    D, u0, k2, k21, h2, k5, h5, k8, h8, k9, h9, k10, k101, h10, k11, h11 = coefficients
    un, v5n, v8n, v9n, v10n, v11n = initial_data
    F = D * dt / dx ** 2
    Nx = int(round(L / dx))  # amount of nodes in space

    u, v5, v8, v9, v10, v11, bu, bv5, bv8, bv9, bv10, bv11 = [np.ndarray(Nx + 1) for _ in range(12)]

    for i in range(0, Nx + 1):
        bu[i] = un[i] + dt * ((k2 * v10n[i] + k21 * v5n[i] * v10n[i]) * (1.0 - un[i] / u0) - h2 * un[i])
        bv10[i] = v10n[i] + dt * (k10 * v9n[i] + k101 * v8n[i] * v9n[i] - h10 * v10n[i])
        
        bv5[i] = v5n[i] + dt * (k5 * un[i] - h5 * v5n[i])
        bv8[i] = v8n[i] + dt * (k8 * un[i] - h8 * v8n[i])
        bv9[i] = v9n[i] + dt * (k9 * v11n[i] - h9 * v9n[i])
        bv11[i] = v11n[i] + dt * (k11 * un[i] - h11 * v11n[i])

    u = sweep(F, Nx, bu)
    v5 = sweep(F, Nx, bv5)
    v8 = sweep(F, Nx, bv8)
    v9 = sweep(F, Nx, bv9)
    v10 = sweep(F, Nx, bv10)
    v11 = sweep(F, Nx, bv11)

    return u, v5, v8 ,v9, v10, v11

def graphicsSYS1(
    initial_data_func: Callable[[float], float],
    T: float,
    L: float,
    F: float,
    dt: float,
    D: float,
    coefficients: Tuple[float, ...],
    p0: float,
    u0: float,
):
    Nt = int(round(T / dt))
    dx = np.sqrt(D * dt / F)  # step size in space
    Nx = int(round(L / dx))
    x = np.linspace(0, L, Nx + 1)

    # p, v, u are current time layer function values
    # pn, vn, un are previous time layer function values
    p, v, u, pn, vn, un = [np.zeros(Nx + 1) for _ in range(6)]

    for i in range(0, Nx + 1):
        pn[i] = p0
        vn[i] = p0 * initial_data_func(0.1 - x[i])
        un[i] = 0

    for _ in range(0, Nt):
        p, u, v = solveSYS1(
            L=L,
            F=F,
            dt=dt,
            D=D,
            coefficients=coefficients,
            u0=u0,
            pn=pn,
            un=un,
            vn=vn,
        )
        un[:] = u
        vn[:] = v
        pn[:] = p

    
    plt.plot(v)
    print(dx)