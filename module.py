from typing import Callable
from typing import Tuple
import numpy as np


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


def solve(
    initial_data_func: Callable[[float], float],  # This is signature of function that accept float value and returns float value.
    L: float,
    T: float,
    Nx: int,
    Nt: int,
    D: float,
    K: float,
    coefficients: Tuple[float, float, float],  # Simply pass coefficients to function as (k1, k2, k3).
    u0: float,
) -> np.ndarray:
    """Solver for equation ....

    Parameters
    ----------
    initial_data_func : Callable[[float], float]
        Function for generation initial data.
    L : float
        L
    T : float
        T
    Nx : int
        Nx
    Nt : int
        Nt
    D : float
        D
    K : float
        K
    coefficients : Tuple[float, float, float]
        Coefficients of ...
    u0 : float
        u0

    Returns
    -------
    np.ndarray
    """

    k3, k4, k5 = coefficients  # Unpack tuple.

    x = np.linspace(0, L, Nx + 1)    # mesh points in space interval [0, L]
    h = x[1] - x[0]                  # step size in space
    t = np.linspace(0, T, Nt + 1)    # mesh points in time interval [0, T]
    tau = t[1] - t[0]                # step size in time
    u = np.zeros(Nx + 1)             # current time layer function values
    un = np.zeros(Nx + 1)            # previous time layer function values

    A = np.zeros((Nx + 1, Nx + 1))   # matrix for the linear system solving
    b = np.zeros(Nx + 1)             # right hand side of the linear system

    # equation to be solved:    U_{t} = D U_{xx} + (k_3 U + k_4 U^2 + k_5 U^3)(U_0 - U) - K U, where K, D, k_i, U_0 are constants.
    # boundary conditions:      U_x = 0, (x = 0);  U_x = 0 (x = L).
    # initial condition:        U(x, 0) = HeavisideFunction(l - x).

    for i in range(0, Nx + 1):
        un[i] = u0 * initial_data_func(0.1 - x[i])

    # numerical scheme:         U^{n+1}_{j} (1 + 2 D tau / h^2) - U^{n+1}_{j-1} D tau / h^2 - U^{n+1}_{j+1} D tau / h^2 = 
    #                                                                           U^{n}_{j} - tau K U^{n}_{j} 
    #                                                                         + tau (k_3 U^{n}_{j} + k_4 (U^2)^{n}_{j} + k_5 (U^3)^{n}_{j})(U_0 - U^{n}_{j})
    #                           for j=1, 2, ..., Nx-1.
    #  
    # boundary conditions:      U^{n+1}_1 = U^{n+1}_{-1},   U^{n+1}_{Nx+1} = U^{n+1}_{Nx-1}, then
    #                           
    #                           for j = 0:   U^{n+1}_{0} (1 + 2 D tau / h^2) - U^{n+1}_{1} D tau / h^2 - U^{n+1}_{1} D tau / h^2 = 
    #                                                                           U^{n}_{0} - tau K U^{n}_{0} 
    #                                                                         + tau (k_3 U^{n}_{0} + k_4 (U^2)^{n}_{0} + k_5 (U^3)^{n}_{0})(U_0 - U^{n}_{0})
    #
    #                           for j = Nx:  U^{n+1}_{Nx} (1 + 2 D tau / h^2) - U^{n+1}_{Nx-1} D tau / h^2 - U^{n+1}_{Nx-1} D tau / h^2 = 
    #                                                                           U^{n}_{Nx} - tau K U^{n}_{Nx} 
    #                                                                         + tau (k_3 U^{n}_{Nx} + k_4 (U^2)^{n}_{Nx} + k_5 (U^3)^{n}_{Nx})(U_0 - U^{n}_{Nx})       
    for i in range(1, Nx):
        A[i, i - 1] = -D * tau / h ** 2
        A[i, i + 1] = -D * tau / h ** 2
        A[i, i] = 1 + 2 * (D * tau / h ** 2)

    A[0, 0] = A[Nx, Nx] = 1 + 2 * (D * tau / h ** 2)
    A[0, 1] = A[Nx, Nx - 1] = -2 * (D * tau / h ** 2)

    for _ in range(0, Nt):  # You don't use n, replace it by placeholder `_`.
        for i in range(0, Nx + 1):
            b[i] = un[i] - tau * K * un[i] + tau * (k3 * un[i] + k4 * un[i] ** 2 + k5 * un[i] ** 3) * (u0 - un[i])

        u[:] = np.linalg.solve(A, b)

        # Switch variables before next step
        un, u = u, un
    return u
