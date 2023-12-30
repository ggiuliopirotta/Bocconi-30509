import numpy as np


def accept(
        C: np.ndarray,
        ro: np.ndarray
    ):
    '''
    compute the acceptance matrix for the Metropolis-Hastings algorithm

    parameters
    ----------
        mat: proposal arbitrary matrix for the transition
        ro: final desired distribution
    
    returns
    ----------
        A: acceptance matrix
    '''
    
    n = C.shape[0] # get the shape of the proposal matrix

    A = np.zeros((n, n), dtype=float) # initialize the acceptance matrix
    # fill in the acceptance matrix
    for i in range(n):
        for j in range(n):

            # check if the spot belongs to the main diagonal and skip it for now eventually
            if i == j:
                continue

            # check if the proposal transition probability is 0 and skip it eventually
            if C[i, j] == 0:
                continue

            A[i, j] = min(1,(C[j, i]*ro[i])/(C[i, j]*ro[j])) # compute the Metropolis-Hastings' rule

    return A


def compute_trans_matrix(
        C: np.ndarray,
        A: np.ndarray
    ):
    '''
    compute the transition matrix for the Metropolis-Hastings' algorithm given the proposal and the acceptance matrix

    parameters
    ----------
        C: proposal arbitrary matrix for the transition
        A: acceptance matrix
    
    returns
    ----------
        Q: final transition matrix
    '''

    n = C.shape[0] # get the shape of the transition matrix

    Q = C*A # compute the transition matrix
    # fill the diagonals so that the columns are normalized
    nn = np.arange(n)
    Q[nn, nn] = 1-np.sum(Q, axis=0)

    return Q


ro = np.array([0.5, 0.25, 0.25]) # define the final desired probability distribution

C = np.array([
    [0, 0.5, 0],
    [1, 0, 1],
    [0, 0.5, 0]
]) # create an arbitrary proposal matrix

A = accept(
    C,
    ro
) # compute the aceptance matrix

Q = compute_trans_matrix(
    C,
    A
) # compute the final transition matrix

Q = np.linalg.matrix_power(Q, n=1000) # wait for a reasonably high number of steps

start = np.random.rand(3) # start with a random distribution
end = Q@start/np.sum(Q@start) # perform the transition

print(f'initial distribution: {start}')
print(f'final distribution: {end}')
