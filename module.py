import numpy as np  

def I(x):
    if x < 0:
        return 0
    else:
        return 1

def solver_T (I, L, T, Nx, Nt, D, K, k3, k4, k5, u0):
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

    for i in range(0, Nx+1):
        un[i] = u0 * I(0.1 - x[i])

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
        A[i,i-1] = - D * tau / h ** 2
        A[i,i+1] = - D * tau / h ** 2
        A[i,i] = 1 + 2 * (D * tau / h ** 2)     

    A[0,0] = A[Nx,Nx] = 1 + 2 * (D * tau / h ** 2)
    A[0,1] = A[Nx,Nx-1] = -2 * (D * tau / h ** 2)
    
    
    
    for n in range(0, Nt):  

        for i in range(0, Nx+1):
            b[i] = un[i] - tau * K * un[i] + tau * (k3 * un[i] + k4 * un[i] ** 2 + k5 * un[i] ** 3) * (u0 - un[i]) 
             
        u[:] = np.linalg.solve(A, b)

        # Switch variables before next step

        un, u = u, un

    return u



