from typing import Callable
from typing import Tuple
import numpy as np
import matplotlib.pyplot as plt

# equation to be solved:    U_{t} = D U_{xx} + (k_3 U + k_4 U^2 + k_5 U^3)(U_0 - U) - K U, where K, D, k_i, U_0 are constants.
# boundary conditions:      U_x = 0, (x = 0);  U_x = 0 (x = L).
# initial condition:        U(x, 0) = HeavisideFunction(l - x).

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
    b: np.ndarray
) -> np.ndarray:
    A = np.zeros((Nx + 1, Nx + 1))   # matrix for the linear system solving
    u = np.zeros(Nx + 1)

    for i in range(1, Nx):
        A[i, i - 1] = - F
        A[i, i + 1] = - F
        A[i, i] = 1 + 2 * F

    A[0, 0] = A[Nx, Nx] = 1 + 2 * F
    A[0, 1] = A[Nx, Nx - 1] = -2 * F

    u[:] = np.linalg.solve(A, b)

    return u 
    

def solve(
    L: float,
    F: float,
    dt: float,
    D: float,
    coefficients: Tuple[float, float, float, float, float, float, float, float, float],  # Simply pass coefficients to function as (k1, k2, k3, k4).
    u0: float,
    pn: np.ndarray,
    vn: np.ndarray,
    un: np.ndarray,
) -> np.ndarray:
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
    coefficients : Tuple[float, float, float, float, float, float, float, float, float]
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
    np.ndarray
    """

    k1, k2, k3, k4, k5, k6, k7, k8, k9 = coefficients                     # Unpack tuple.


    dx = np.sqrt(D * dt / F)                         # step size in space
    Nx = int(round(L / dx))                          # amount of nodes in space 

    
    p = np.zeros(Nx + 1)
    v = np.zeros(Nx + 1)
    u = np.zeros(Nx + 1)                             # current time layer function values
    bu = np.zeros(Nx + 1)
    bp = np.zeros(Nx + 1)
    bv = np.zeros(Nx + 1)                             # right hand side of the linear system

    for i in range(0, Nx + 1):
        bu[i] = un[i] - dt * k4 * un[i] + dt * (k1 + k2 * vn[i] + k3 * vn[i] ** 2) * (u0 - un[i])
        bv[i] = vn[i] - dt * k9 * vn[i] + dt * (k5 * un[i] + k6 * vn[i] + k7 * vn[i] ** 2 + k8 * vn[i] ** 3) * pn[i]
        bp[i] = pn[i] - dt * (k5 * un[i] + k6 * vn[i] + k7 * vn[i] ** 2 + k8 * vn[i] ** 3) * pn[i]
    
    p = sweep(F, Nx, bp)
    v = sweep(F, Nx, bv)
    u = sweep(F, Nx, bu)

    return p, u, v



def graph(
    initial_data_func: Callable[[float], float],      # This is signature of function that accept float value and returns float value.
    T: float,
    L: float,
    F: float,
    dt: float,
    D: float,
    coefficients: Tuple[float, float, float, float, float, float, float, float, float],  # Simply pass coefficients to function as (k1, k2, k3, k4).
    p0: float,
    u0: float,
):     
    Nt = int(round(T / dt))   
    t = np.linspace(0, T, Nt + 1)
    dx = np.sqrt(D * dt / F)                          # step size in space
    Nx = int(round(L / dx)) 
    x = np.linspace(0, L, Nx + 1)           
    p = np.zeros(Nx + 1)
    v = np.zeros(Nx + 1)
    u = np.zeros(Nx + 1)                             # current time layer function values
    pn = np.zeros(Nx + 1)
    vn = np.zeros(Nx + 1)
    un = np.zeros(Nx + 1)

    for i in range(0, Nx + 1):
        pn[i] = p0
        vn[i] = p0 * initial_data_func(0.1 - x[i])
        un[i] = 0

    for n in range(0, Nt):                           # You don't use n, replace it by placeholder `_`.
        p, u, v = solve(L, F, dt, D, coefficients, u0, pn, un, vn)
        un[:] = u
        vn[:] = v
        pn[:] = p

        
    # plt.plot(p)
    # plt.plot(u)
    plt.plot(v)
    print(dx)

