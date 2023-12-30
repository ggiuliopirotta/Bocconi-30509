import numpy as np
from numpy import random as rnd
import time


def accept(
        c_delta: float,
        b: float
    ):
    '''
    accept the proposed move according to Boltzmann's rule

    parameters
    ----------
        c_delta: cost difference associated to the alternative move
        b: inverse of the entropy of the state

    returns
    ----------
        acc: boolean for acceptance

    '''

    acc = None # initialize the acceptance verdict
    
    # check if the cost difference is non-negative and accept the move eventually
    if c_delta <= 0:
        acc = True
    
    # check if the entropy is too low and reject the move eventually
    if b == np.inf:
        acc = False
    
    # compute the probability of acceptance using Boltzmann's rule
    if acc == None:
        prob = np.exp(-b*c_delta)
        acc = rnd.random() < prob

    return acc


def sim_ann(
        prob,
        iters: int,
        b0: float,
        b1: float,
        ann_steps: int,
        seed: int=None
    ):
    '''
    perform a simulated annealing optimization on the given problem

    parameters
    ----------
        prob: implementation for the given problem 
        iters: number of iterations
        b0: starting inverse of the entropy
        b1: ending inverse of the entropy
        ann_steps: annealing steps from start to end
        seed: boolean for reproducibility

    returns
    ----------
        best: problem configuration associated to the best cost
    '''
    
    # set the seed if necessary
    if seed is not None:
        rnd.seed(seed)

    best = None # initialize the best problem
    c_min = np.inf # initialize the best cost

    # set the initial configuration
    prob.init_config()
    prob.display()
    c = prob.compute_cost()
    print (f'initial cost: {round(c, 2)}')

    # get a log scale list of the entropy levels
    b_list = np.zeros(ann_steps)
    b_list[:-1] = np.logspace(np.log10(b0), np.log10(b1), ann_steps-1)
    b_list[-1] = np.inf
    
    # loop over the entropy levels
    for b in b_list:
        acc = 0 # keep track of the acceptance rate

        # compute the delta cost associated to each move
        for _ in range(iters):
            move = prob.propose_move()
            c_delta = prob.compute_delta_cost(move)

            # check if Boltzmann's rule is met and accept the move eventually
            if accept(c_delta, b):
                prob.accept_move (move)
                c += c_delta # update the cost
                acc += 1 # update the acceptance rate
    
                # check if the new configuration is better and set it eventually
                if c < c_min:
                    c_min = c
                    best = prob.copy()
            
            # check that the cost is null and break the loop eventually
            if c == 0:
                break
        
        # check that the cost is null and break the loop eventually
        if c == 0:
            break

        print(f'beta: {round(b, 2)}, cost: {round(c, 2)}, acc. rate: {acc/iters}') # print info

    print(f'min cost: {round(c_min, 2)}\n')
    best.display() # print the best configuration

    return best


def greedy(
        prob,
        iters: int,
        restarts: int,
        seed: int=None
    ):
    '''
    perform a greedy optimization on the given problem

    parameters
    ----------
        prob: implementation for the given problem 
        iters: number of iterations
        restarts: number of different starting points
        seed: boolean for reproducibility

    returns
    ----------
        best: problem configuration associated to the best cost
    '''

    # set the seed if necessary
    if seed is not None:
        np.random.seed(seed)

    best = None # initialize the best problem
    c_min = np.inf # initialize the best cost

    # keep track of the time
    start = time.time()
    end = np.inf

    # optimization loop
    for _ in range(restarts): # loop over different starting points
        # set the initial configuration
        prob.init_config()
        c = prob.compute_cost()
        print(f'initial cost: {c}')

        # compute the delta cost associated to each move
        for _ in range(iters):
            move = prob.propose_move()
            c_delta = prob.compute_delta_cost(move)

            # check if the move is an improvement and accept it eventually
            if c_delta <= 0:
                prob.accept_move(move)
                c += c_delta # update the cost
            
            # check that the cost is null and break the loop eventually
            if c == 0:
                break
        end = min(end, time.time())

        # check if the new configuration is better and set it eventually
        if c < c_min:
            c_min = c
            best = prob.copy()
        
        # check that the cost is null and break the loop eventually
        if c == 0:
            break
    
    # check that the optimal configuration hasn't been reached and get the time eventually
    if end == np.inf:
        end = time.time()

    print(f'min cost: {c_min}')
    print(f'time to get it: {round(end-start, 2)}\n')
    best.display() # print the best configuration

    return best
