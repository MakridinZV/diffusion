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
    L: float,
    F: float,
    dt: float,
    D: float,
    coefficients: Tuple[float, ...],  # Simply pass coefficients to function as (k1, k2, k3, k4).
    u0: float,
    v5n: np.ndarray,
    v8n: np.ndarray,
    v9n: np.ndarray,
    v10n: np.ndarray,
    v11n: np.ndarray,
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
    v5n : np.ndarray
        v5n
    v8n : np.ndarray
        v8n
    v9n : np.ndarray
        v9n
    v10n : np.ndarray
        v10n
    v11n : np.ndarray
        v11n
    un : np.ndarray
        un

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
    """
    dx = np.sqrt(D * dt / F)  # step size in space
    Nx = int(round(L / dx))  # amount of nodes in space

    # v5, v8, v9, v10, v11, u are current time layer function values
    # v5n, v8n, v9n, v10n, v11n, un are previous time layer function values
    v5, v8, v9, v10, v11, u, bv5, bv8, bv9, bv10, bv11, bu = [np.zeros(Nx + 1) for _ in range(12)]

    k2, k21, h2, k5, h5, k8, h8, k9, h9, k10, k101, h10, k11, h11 = coefficients  # Unpack tuple.
    for i in range(0, Nx + 1):
        bu[i] = un[i] - dt * h2 * un[i] + dt * (k2 * v10n[i] + v5n[i] * v10n[i] * k21) * (1 - un[i] / u0)
        bv5[i] = v5n[i] - dt * h5 * v5n[i] + dt * k5 * un[i]
        bv8[i] = v8n[i] - dt * h8 * v8n[i] + dt * k8 * un[i]
        bv9[i] = v9n[i] - dt * h9 * v9n[i] + dt * k9 * v11n[i]
        bv10[i] = v10n[i] - dt * h10 * v10n[i] + dt * (k10 * v9n[i] + v8n[i] * v9n[i] * k101)
        bv11[i] = v11n[i] - dt * h11 * v11n[i] + dt * k11 * un[i]
        

    v5 = sweep(F, Nx, bv5)
    v8 = sweep(F, Nx, bv8)
    v9 = sweep(F, Nx, bv9)
    v10 = sweep(F, Nx, bv10)
    v11 = sweep(F, Nx, bv11)
    u = sweep(F, Nx, bu)
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

    # plt.plot(p)
    # plt.plot(u)
    plt.plot(v)
    print(dx)

def graphicsSYS2(
    initial_data_func: Callable[[float], float],
    T: float,
    L: float,
    F: float,
    dt: float,
    D: float,
    coefficients: Tuple[float, ...],
    u0: float,
    v50: float,
    v80: float,
    v90: float,
    v100: float,
    v110: float,
):
    Nt = int(round(T / dt))
    dx = np.sqrt(D * dt / F)  # step size in space
    Nx = int(round(L / dx))
    x = np.linspace(0, L, Nx + 1)

    # v5, v8, v9, v10, v11, u are current time layer function values
    # v5n, v8n, v9n, v10n, v11n, un are previous time layer function values
    v5, v8, v9, v10, v11, u, v5n, v8n, v9n, v10n, v11n, un = [np.zeros(Nx + 1) for _ in range(12)]

    for i in range(0, Nx + 1):
        un[i] = u0 * initial_data_func(0.1 - x[i])
        v5n[i] = v50 * initial_data_func(0.1 - x[i])
        v8n[i] = v80 * initial_data_func(0.1 - x[i])
        v9n[i] = v90 * initial_data_func(0.1 - x[i])
        v10n[i] = v100 * initial_data_func(0.1 - x[i])
        v11n[i] = v110 * initial_data_func(0.1 - x[i])

    for _ in range(0, Nt):
        u, v5, v8, v9, v10, v11 = solveSYS2(
            L=L,
            F=F,
            dt=dt,
            D=D,
            coefficients=coefficients,
            u0=u0,
            v5n=v5n,
            v8n=v8n,
            v9n=v9n,
            v10n=v10n,
            v11n=v11n,
            un=un,
        )
        un[:] = u
        v5n[:] = v5
        v8n[:] = v8
        v9n[:] = v9
        v10n[:] = v10
        v11n[:] = v11
        

    
    plt.plot(u)
    print(dx)
