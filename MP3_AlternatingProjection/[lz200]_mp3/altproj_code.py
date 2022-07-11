import numpy as np
from scipy.linalg import inv, svd
from tqdm import tqdm_notebook as tqdm

### Helper functions

# Compute null space
def null_space(A, rcond=None):
    """
    Compute null spavce of matrix XProjection on half space defined by {v| <v,w> = c}
    Arguments:
        A {numpy.ndarray} -- matrix whose null space is desired
        rcond {float} -- intercept
    Returns:
        Q {numpy.ndarray} -- matrix whose (rows?) span null space of A
    """
    u, s, vh = svd(A, full_matrices=True)
    M, N = u.shape[0], vh.shape[1]
    if rcond is None:
        rcond = np.finfo(s.dtype).eps * max(M, N)
    tol = np.amax(s) * rcond
    num = np.sum(s > tol, dtype=int)
    Q = vh[num:,:].T.conj()
    return Q

### End Helper Functions

# Exercise 1: Alternating projection for subspaces
def altproj(A, B, v0, n):
    """
    Arguments:
        A {numpy.ndarray} -- matrix whose columns form basis for subspace U
        B {numpy.ndarray} -- matrix whose columns form baiss for subspace W
        v0 {numpy.ndarray} -- initialization vector
        n {int} -- number of sweeps for alternating projection
    Returns:
        v {numpy.ndarray} -- the output after 2n steps of alternating projection
        err {numpy.ndarray} -- the error after each full pass
    """
    
    ### Add code here

    #########################################################################################################
    PU = np.dot(A, np.dot((inv((A.T).dot(A))), A.T))
    PW = np.dot(B, np.dot((inv((B.T).dot(B))), B.T))

    basis_UnitW = np.hstack([A, B]) @ null_space(np.hstack([A, -B]))
    null_space_flag = 0 

    if basis_UnitW.shape[1] != 0 :
        P_UnitW = np.dot(basis_UnitW, np.dot((inv((basis_UnitW.T).dot(basis_UnitW))), basis_UnitW.T))
        v_star = np.matmul(P_UnitW, v0)
        null_space_flag = 0
    else :
        null_space_flag = 1 

    # print(null_space_flag)
    # print(null_space(np.hstack([A, -B])).shape)
    # print(basis_UnitW.shape)
    # print(P_UnitW.shape)
    # print(v0.shape)

    v = v0
    err = np.zeros(n)

    tqdm_total = 2 * n 
    for i in tqdm(range(0, tqdm_total), total = tqdm_total, leave = False) :
        if i % 2 == 0 :
            v = np.dot(PU, v)
        else :
            v = np.dot(PW, v)
        if (i + 1) % 2 == 0 :
            if null_space_flag == 0 :
                err[i // 2] = np.amax(abs(v - v_star))
            if null_space_flag == 1 :
                err[i // 2] = np.amax(abs(v))
    #########################################################################################################

    return v, err 

# Exercise 2: Kaczmarz algorithm for solving linear systems
def kaczmarz(A, b, I):
    """
    Arguments:
        A {numpy.ndarray} -- matrix defines the LHS of linear equation
        b {numpy.ndarray} -- vector defines the RHS of linear equation
        I {int} -- number of full passes through the Kaczmarz algorithm
    Returns:
        X {numpy.ndarray} -- the output of all I full passes
        err {numpy.ndarray} -- the error after each full pass
    """
    
    ### Add code here
    #########################################################################################################
    m = A.shape[0]
    n = A.shape[1]

    X = np.zeros((n, I))
    err = np.zeros(I)

    v = np.zeros(n)
    
    tqdm_total = I * m 
    for i in tqdm(range(0, tqdm_total), total = tqdm_total, leave = False) :
        A_i = A[i % m]
        b_i = b[i % m]
        v = v - np.dot(A_i, (np.dot(v, A_i) - b_i) / (np.dot(A_i, A_i)))
        if (i + 1) % m == 0 :
            X[:, i // m] = v.T 
            err[i // m] = np.amax(abs(np.dot(A, v) - b))
    #########################################################################################################

    return X, err

# Exercise 4: Alternating projection to satisfy linear inequalities
def lp_altproj(A, b, I, s=1):
    """
    Find a feasible solution for A v >= b using alternating projection
    starting from v0 = 0
    Arguments:
        A {numpy.ndarray} -- matrix defines the LHS of linear equation
        b {numpy.ndarray} -- vector defines the RHS of linear equation
        I {int} -- number of full passes through the alternating projection
        s {numpy.float} -- step size of projection (defaults to 1)
    Returns:
        v {numpy.ndarray} -- the output after I full passes
        err {numpy.ndarray} -- the error after each full pass
    """
    
    # Add code here
    #########################################################################################################
    m = A.shape[0]
    n = A.shape[1]

    v = np.zeros(n)
    err = np.zeros(I)

    tqdm_total = I * m 
    for i in tqdm(range(0, tqdm_total), total = tqdm_total, leave = False) :
        A_i = A[i % m]
        b_i = b[i % m]
        project_compare = np.dot(A_i, v)
        if project_compare >= b_i :
            v = v 
        else :
            update_v = (np.dot(A_i, v) - b_i) / (np.dot(A_i, A_i))
            update_v = update_v * A_i 
            # v = v - update_v 
            update_v = v - update_v 
            v = (1 - s) * v + s * update_v 
        if (i + 1) % m == 0 :
            update_err = np.max(b - A @ v)
            if update_err > 0 :
                err[i // m] = update_err 
    #########################################################################################################
    
    return v, err