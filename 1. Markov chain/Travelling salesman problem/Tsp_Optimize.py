import matplotlib.pyplot as plt
import numpy as np


def greedy(
        prob,
        iters: int,
        restarts: int,
        plot: bool=True,
        seed: int=None
    ):
    '''
    perform a greedy optimization on the given problem

    parameters
    ----------
        prob: implementation for the given problem 
        iters: number of iterations
        restarts: number of different starting points
        plot: boolean for visualization
        seed: boolean for reproducibility

    returns
    ----------
        best: problem configuration associated to the best cost
    '''

    # set the seed if necessary
    if seed is not None:
        np.random.seed(seed)

    # get all possible pairs of nodes
    e_pairs = prob.couple_edges()

    best = None # initialize the best problem
    c_min = np.inf # initialize the best cost

    # optimization loop
    for _ in range(restarts): # loop over different starting points
        # set the initial configuration
        prob.init_config()
        c = prob.compute_cost()
        print(f'initial cost: {c}')

        for _ in range(iters):
            move_best = None # initialize the best move
            c_delta_min = np.inf # initialize the best delta cost

            # compute the delta cost associated to each move
            for move in e_pairs:
                c_delta = prob.compute_delta_cost(move)
    
                # check if the delta cost is the best and set it eventually
                if c_delta < c_delta_min:
                    move_best = move
                    c_delta_min = c_delta

            # accept the best move and update the cost
            prob.accept_move(move_best)
            c += c_delta_min

            # plot the new configuration if necessary
            if plot:
                prob.display()

        # check if the new configuration is better and set it eventually
        if c < c_min:
            c_min = c
            best = prob.copy()
    
    print(f'min cost: {c_min}')
    # plot the best configuration 
    best.display()
    plt.show()

    return best
