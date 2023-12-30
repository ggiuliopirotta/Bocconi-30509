from copy import deepcopy
import matplotlib.pyplot as plt
import numpy as np
from numpy import random as rnd


class Wmc:
    '''
    represent the **weighted max cut problem**

    the problem asks to separe a given number of interconnected points in two groups,
    such that the total cut between points from different groups is the longest possible

    parameters
    ----------
        n: number of nodes
        x, y: coordinates of the nodes
        m: connection matrix storing the distance between all connected nodes
        p: partition list separating the nodes in two groups
        seed: integer seed for reproducibility
    
    methods
    ----------
        set_connections: create the edges between some random pairs of nodes
        init_config: set the initiaĺ random partition for the problem
        propose_move: propose a change to the partition
        accept_move: accept the proposed change to the partition
        compute_delta_cost: compute the cost difference associated to a change to the partition
        compute_cost: compute the total cost of the partition
        display: visualize the nodes and the connections
        copy: copy the problem
    '''

    def __init__(
            self,
            n: int,
            p: float,
            seed: int=None
        ):

        # set the seed if necessary
        if seed is not None:
            rnd.seed(seed)

        # store the problem variables
        self.n = n
        self.x, self.y = rnd.random(n), rnd.random(n)

        self.m = self.set_connections(p) # configure the connection matrix
        self.p = [[], []] # initialize the partition

    def set_connections(
            self,
            p: float
        ):
        '''
        create the edges between some random pairs of nodes respecting the given probability

        parameters
        ----------
            p: probability of two nodes to be connected to each other
        
        returns
        ----------
            m: connection matrix storing the distance between all connected nodes
        '''
        
        n, x, y = self.n, self.x, self.y

        m = np.zeros((n, n)) # initialize the connection matrix
        # fill in the connection matrix
        for i in range(n):
            for j in range(i+1, n): # exploit the simmetry of the matrix

                # check if the nodes are connected and store the distance eventually
                if rnd.random() < p:
                    m[i, j] = np.sqrt((x[i]-x[j])**2 + (y[i]-y[j])**2) # compute the distance between two nodes
                    m[j, i] = m[i, j]

        return m
    
    def init_config(self):
        '''
        set the initiaĺ random partition for the problem
        '''

        n = self.n

        part1 = np.sort(rnd.choice(np.arange(n), rnd.randint(1, n-1), replace=False)) # select some random nodes for the first group
        part2 = np.delete(np.arange(n), part1) # put the remaining nodes into the second group

        self.p[0], self.p[1] = part1, part2 # store the partition
    
    def propose_move(self):
        '''
        propose a change to the partition moving a node from its group to the other

        returns
        ----------
            p_choice: group to move the node from
            v_ind: index of the node to move
            v: node to move

        '''

        p = self.p

        p_choice = rnd.randint(2) # choose the group to pick the node from
        # check if the group has enough nodes and switch the group otherwise
        if len(p[p_choice]) <= 1:
            p_choice = 1-p_choice

        p_from = p[p_choice]

        # propose a node to move from the chosen group to the other
        v_ind = rnd.choice(len(p_from))
        v = p_from[v_ind]

        return p_choice, v_ind, v

    def accept_move(
            self,
            move: tuple
        ):
        '''
        accept the proposed change to the partition moving a node from its group to the other

        parameters
        ----------
            move: tuple containing the group and the node to move
        '''
        p = self.p
        p_choice, v_ind, v = move

        p_from, p_to = p[p_choice], p[1-p_choice] # distinguish the two groups
        p_from, p_to = np.delete(p_from, v_ind), np.sort(np.append(p_to, v)) # move the proposed node from the chosen group to the other

        self.p[p_choice], self.p[1-p_choice] = p_from, p_to # set the new partition

    def compute_delta_cost(
            self,
            move: tuple
        ):
        '''
        compute the cost difference associated to a change to the partition without virtually accepting the move

        parameters
        ----------
            move: tuple containing the group and the node to move
        
        returns
        ----------
            w_delta: cost difference between the old partition and the alternative one
        '''

        m, p = self.m, self.p
        p_choice, v = move[0], move[2]

        w_old = np.sum(m[v, p[1-p_choice]]) # compute the cost associated to the old partition
        w_new = np.sum(m[v, p[p_choice]]) # compute the cost associated to the alternative partition

        w_delta = w_new-w_old # compute the cost difference
        return w_delta

    def compute_cost(self):
        '''
        compute the total cost of the partition using the nodes' coordinates

        returns
        ----------
            w: total cost of the partition
        '''
        n, m, p = self.n, self.m, self.p

        w = 0.0 # initialize the cost
        # add the single costs cumulatively
        for i in range(n):
            for j in range(i+1, n):
                # check that the nodes belongs to different groups and add the cost eventually
                if (i in p[0] and j in p[1]) or (i in p[1] and j in p[0]):
                    w += m[i, j]

        return w

    def display(self):
        '''
        visualize the nodes and the connections in matplotlib
        '''
        
        n, x, y, m, p = self.n, self.x, self.y, self.m, self.p

        # pause and clear the figure
        plt.pause(0.01)
        plt.figure(1)
        plt.clf()

        for i in range(n):
            for j in range(i+1, n):

                # check that the two nodes are connected and get their coordinates eventually
                if m[i, j] != 0:
                    xx = [x[i], x[j]]
                    yy = [y[i], y[j]]

                    # check that the two nodes belong to different groups
                    if (i in p[0] and j in p[1]) or (i in p[1] and j in p[0]):
                        plt.plot(xx, yy, color='green', alpha=0.5) # plot the connections in green eventually

                    else:
                        plt.plot(xx, yy, color='grey', alpha=0.3) # plot the connections in grey otherwise

        plt.plot(x[p[0]], y[p[0]], 'ro') # plot the nodes belonging to the first group in red
        plt.plot(x[p[1]], y[p[1]], 'bo') # plot the nodes belonging to the second group in blue
    
    def copy(self):
        '''
        copy the problem with all its attributes and methods
        '''

        return deepcopy(self)
