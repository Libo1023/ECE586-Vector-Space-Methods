import numpy as np
from scipy.linalg import null_space



# General function to compute expected hitting time for Exercise 1
def compute_Phi_ET(P, ns=100):
    '''
    Arguments:
        P {numpy.array} -- n x n, transition matrix of the Markov chain
        ns {int} -- largest step to consider

    Returns:
        Phi_list {numpy.array} -- (ns + 1) x n x n, the Phi matrix for time 0, 1, ...,ns
        ET {numpy.array} -- n x n, expected hitting time approximated by ns steps ns
    '''
    # Add code here to compute following quantities:
    # Phi_list[m, i, j] = phi_{i,j}^{(m)} = Pr( T_{i, j} <= m )
    # ET[i, j] = E[ T_{i, j} ] ~ \sum_{m=1}^ns m Pr( T_{i, j} = m )
    # Notice in python the index starts from 0
    
    ###############################################################################

    n = P.shape[0]
    Phi_list = np.zeros(((ns + 1), n, n))
    ET       = np.zeros((n, n))

    for i in range(0, ns + 1) :
        Phi_list[i, :, :] = np.identity(n)

    for j in range(1, ns + 1) :
        Phi_list[j, :, :] = np.dot(P, Phi_list[j-1, :, :])
        for k in range(0, n) :
            diagonal = Phi_list[j, :, :]
            diagonal[k, k] = 1  
        Phi_list[j, :, :] = diagonal

    for i in range(1, ns+1) :
        ET = ET + np.dot((i), (Phi_list[i, :, :] - Phi_list[i-1, :, :]))

    ###############################################################################

    return Phi_list, ET  



# General function to simulate hitting time for Exercise 1
def simulate_hitting_time(P, states, nr):
    '''
    Arguments:
        P {numpy.array} -- n x n, transition matrix of the Markov chain
        states {list[int]} -- the list [start state, end state], index starts from 0
        nr {int} -- largest step to consider

    Returns:
        T {list[int]} -- a size nr list contains the hitting time of all realizations
    '''
    # Add code here to simulate following quantities:
    # T[i] = hitting time of the i-th run (i.e., realization) of process
    # Notice in python the index starts from 0
    
    ##############################################################################
    start_state = states[0]
    end_state   = states[1]
    if start_state == end_state :
        T = np.zeros(nr)
    else :
        T = np.zeros(nr)
        for i in range(0, nr) :
            current_state = start_state
            ith_hitting_time = 0
            while current_state != end_state :
                current_row = P[current_state,:]
                # Let the random variable U be 
                # uniformly distributed on the interval [0, 1).
                U = np.random.random_sample()
                # Let FX be the cumulative distribution function of X.
                FX = 0
                for j in range(0, P.shape[1]) :
                    FX = FX + current_row[j]
                    if U < FX :
                        current_state = j
                        break
                
                ith_hitting_time = ith_hitting_time + 1
                
            T[i] = ith_hitting_time
    ##############################################################################
    
    return T



# General function to compute the stationary distribution (if unique) of a Markov chain for Exercise 3
def stationary_distribution(P):
    '''
    Arguments:
        P {numpy.array} -- n x n, transition matrix of the Markov chain

    Returns:
        pi {numpy.array} -- length n, stationary distribution of the Markov chain
    '''

    # Add code here: Think of pi as column vector, solve linear equations:
    #     P^T pi = pi
    #     sum(pi) = 1

    ########################################################################################
    n = P.shape[0]
    nn_identity_matrix = np.identity(n)
    # Can find stationary distribution using row reduction after rewriting (pi)P = (pi) as
    # transpose(I - P) dot_product transpose(pi) = 0
    pi = null_space((nn_identity_matrix - P).T)
    pi = pi / float(sum(pi))
    ########################################################################################

    return pi
