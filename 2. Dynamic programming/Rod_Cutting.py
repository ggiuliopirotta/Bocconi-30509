import numpy as np


# design the test for debugging
rod_test = np.array([1, 5, 8, 9, 10, 17, 17, 20, 24, 30])
r_best = 30
cut_best = [10]


def test_fun(
        fun,
        rod: np.ndarray,
        r_best: int,
        cut_best: list
):
    '''
    test that the given function performs the **rod cutting algorithm** in the correct manner

    parameters
    ----------
        fun: rod cutting function to test
        rod: rod to cut
        r: best possible revenue achievable with the given rod
        cut: optimal cut of the rod that guarantees the best revenue
    '''

    # check that the function to test returns the optimal cut and raise an exception otherwise
    try:
        r, cut = fun(rod) # perform the algorithm
    except:
        r = fun(rod)
        cut = None

    print(f'r: {r}, cut: {cut}')

    s = '' # initialize the message
    # check that the algorithm has been performed correctly and print a message according to the case
    if r == r_best and (cut == cut_best or cut is None):
        s = 'well done'
    elif (cut == cut_best or cut is None):
        s = 'not the optimal cut'
    else:
        s = 'not the best revenue'
    
    print(s)


def rodcut_rec(rod: np.ndarray):
    '''
    recursive function to perform the **rod cutting algorithm**

    the algorithm computes the highest revenue achievable on rod through optimal cuts of the rod itself

    
    parameters
    ----------
        rod: rod to cut
    
    returns
    ----------
        r_best: best possible revenue achievable with the given rod
    '''

    n = len(rod)

    # check that the rod is null and return 0 eventually
    if n == 0:
        return 0.0

    r_best = 0 # initialize the best revenue
    # use a recursive approach to break down the remainder in smaller portions
    for i in range (n):
        r_i = rod[i]+rodcut_rec(rod[:n-i-1])

        # check that the current cut is the best and set it eventually
        if r_i >= r_best:
            r_best = r_i

    return r_best


def rodcut_memo(
        rod: np.ndarray,
        memo: dict=None
    ):
    '''
    recursive function with an helping dictionary to perform the **rod cutting algorithm**

    the algorithm computes the highest revenue achievable on rod through optimal cuts of the rod itself

    parameters
    ----------
        rod: rod to cut
        memo: dictionary storing the best revenues and optimal cuts for smaller portions of the rod
    
    returns
    ----------
        r_best: best possible revenue achievable with the given rod
        cut_best: optimal cut of the rod that guarantees the best revenue
    '''

    n = len(rod)

    # check that the rod is null and return 0 eventually
    if n == 0:
        return 0.0, []

    # initialize the structures for the algorithm
    r_best = 0
    cut_best = []
    if memo is None:
        memo = {}

    # loop over all the possible remainders of the rod after the cut
    for i in range(1, n+1):
        k = n-i

        # check that the remainder is in the dictionary and recover the results eventually
        if k in memo:
            r, r_cut = memo[k]

        # otherwise compute the best revenue and optimal cut recursively
        else:
            r, r_cut = rodcut_memo(rod[:k], memo)
            memo[k] = r, r_cut # store the results in the dictionary

        r_i = rod[i-1]+r # get the best revenue

        # check that the current cut is the best and set it eventually
        if r_i >= r_best:
            r_best = r_i
            cut_best = r_cut+[i]

    return r_best, cut_best


def rodcut_dyn(rod: np.ndarray):
    '''
    dynamic function to perform the **rod cutting algorithm**

    the algorithm computes the highest revenue achievable on rod through optimal cuts of the rod itself

    parameters
    ----------
        rod: rod to cut
    
    returns
    ----------
        r_best: best possible revenue achievable with the given rod
        cut_best: optimal cut of the rod that guarantees the best revenu
    '''

    n = len(rod)
    
    # initialize the structures for the algorithm
    r_best = np.zeros(n+1)
    whence = np.zeros(n+1, dtype=int)

    # compute the best best revenue and optimal cut for progressively bigger portions of the rod based on previous results
    for i in range(1, n+1):
        # initialize the results
        r_i = 0
        r_cut = 0

        # loop over the possible remainders of the rod after the cut
        for j in range(1, i+1):
            r_j = rod[j-1]+r_best[i-j] # get the best revenue

            # check that the current combination is the best and set it eventually
            if r_j >= r_i:
                r_i = r_j
                r_cut = j

        # store the results for the current portion
        r_best[i] = r_i
        whence[i] = r_cut

    best_cut = [] # initialize the optimal cut list

    k = n
    # loop for the backward pass
    while k > 0:
        # recover the cut made and append it
        c = whence[k]
        best_cut.append(c)
        k -= c # shift to the previous smaller portion

    # adjust the results
    r_best = r_best[-1] # return only the best revenue for the original problem
    best_cut = sorted(best_cut)

    return r_best, best_cut


test_fun(
    fun=rodcut_dyn,
    rod=rod_test,
    r_best=r_best,
    cut_best=cut_best
) # test a rod cutting function
