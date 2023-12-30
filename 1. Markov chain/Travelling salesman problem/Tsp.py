from copy import deepcopy
import matplotlib.pyplot as plt
import numpy as np
from numpy import random as rnd


class Tsp:
    '''
    represent the **travelling salesman problem**

    the problem asks to arrange a given number of points in the optimal order,
    such that the route to go through all of them is the shortest possible

    attributes
    ----------
        n: number of nodes
        x, y: coordinates of the nodes
        dist: matrix storing the distance within the nodes
        route: optimal path to follow
        seed: integer seed for reproducibility
    
    methods
    ----------
        init_config: set the initiaĺ random route for the problem
        couple_edges: create all possible edges connecting any combination of nodes
        propose_move: propose a change to the route
        accept_move: accept the proposed change to the route
        compute_dist: compute the distance between two nodes
        compute_delta_cost: compute the cost difference associated to a change to the route
        compute_cost: compute the total cost of the route
        display: visualize the nodes and the route
        copy: copy the problem
    '''

    def __init__(
            self,
            n: int,
            seed: int=None
        ):

        # set the seed if necessary
        if seed is not None:
            rnd.seed(seed)

        # store the problem variables
        self.n = n
        self.x, self.y = rnd.random(n), rnd.random(n)

        self.dist = np.zeros((n, n)) # initialize the distance matrix
        # fill in the distance matrix
        for e1 in range(n):
            for e2 in range(e1+1, n): # exploit the symmetry of the matrix
                self.dist[e1, e2] = self.compute_dist(e1, e2) # compute the distance between two nodes
                self.dist[e2, e1] = self.dist[e1, e2]

        self.route = np.arange(n) # initialize the path

    def init_config(self):
        '''
        set the initial random route for the problem
        '''

        n = self.n

        self.route[:] = rnd.permutation(n) # generate a random route
    
    def couple_edges(self):
        '''
        create all possible edges connecting any combination of nodes and store their distance in a matrix

        returns
        ----------
            e_pairs: matrix containing the distance between all possible pairs of nodes
        '''

        n = self.n

        e_pairs = [] # initialize the edge matrix
        # fill in the distance matrix
        for e1 in range(n):
            for e2 in range(n):
                # check that the two nodes are not adjacent and append the couple eventually
                if e1 < e2 and (e1-1) %n != e2 and (e1+1) %n != e2:
                    e_pairs.append((e1, e2))

        return e_pairs

    def propose_move(self):
        '''
        propose a change to the route swapping two nodes

        returns
        ----------
            e1: first node to swap
            e2: second node to swap
        '''

        n = self.n

        # loop until the conditions are satiisfied
        while True:
            # propose two nodes to swap
            e1 = rnd.randint(n)
            e2 = rnd.randint(n)

            # check that the second node comes actually after and swap them otherwise
            if e2 < e1:
                e1, e2 = e2, e1
                
            # check that the two nodes are not adjacent and break the loop eventually
            if e1 != e2 and (e1-1) %n != e2 and (e1+1) %n != e2:
                break

        return e1, e2
        
    def accept_move(
            self,
            move: tuple
        ):
        '''
        accept the proposed change to the route swapping the two nodes

        parameters
        ----------
            move: tuple containing the two nodes to swap 
        '''
        
        e1, e2 = move

        self.route[e1+1:e2+1] = self.route[e2:e1:-1] # swap the proposed nodes and all the enclosed route accordingly
    
    def compute_dist(
            self,
            e1: int,
            e2: int
        ):
        '''
        
        compute the distance between two nodes using their coordinates

        parameters
        ----------
            e1: first node
            e2: second node
        
        returns
        ----------
            dist: distance between the two nodes
        '''
        
        x, y = self.x, self.y

        dist = np.sqrt((x[e1]-x[e2])**2 + (y[e1]-y[e2])**2) # compute the euclidean distance
        return dist

    def compute_delta_cost(
            self,
            move: tuple
        ):
        '''
        compute the cost difference associated to a change to the route without virtually accepting the move

        parameters
        ----------
            move: tuple containing the two nodes to swap
        
        returns
        ----------
            c_delta: cost difference between the old route and the alternative one
        '''
        
        n, dist, route = self.n, self.dist, self.route
        e1, e2 = move

        # detect the critical nodes
        e1_prev, e1_next = route[e1], route[(e1+1) %n] # identify the old and alternative first nodes
        e2_prev, e2_next = route[e2], route[(e2+1) %n] # identify the old and alternative second nodes

        # compute the costs
        c_old = dist[e1_prev, e1_next] + dist[e2_prev, e2_next] # compute the cost associated to the old route
        c_new = dist[e1_prev, e2_prev] + dist[e1_next, e2_next] # compute the cost associated to the alternative route

        c_delta = c_new-c_old # compute the cost difference
        return c_delta

    def compute_cost(self):
        '''
        compute the total cost of the route using the nodes' coordinates

        returns
        ----------
            c: total cost of the route
        '''
        
        n, dist, route = self.n, self.dist, self.route

        c = 0.0 # initialize the cost
        # add the single costs cumulatively
        for e in range(n):
            c += dist[route[e], route[(e+1) %n]]

        return c

    def display(self):
        '''
        visualize the nodes and the route in matplotlib
        '''
      
        x, y, route = self.x, self.y, self.route

        # pause and clear the figure
        plt.pause(0.01)
        plt.figure(1)
        plt.clf()

        plt.plot(x[route], y[route], color='orange') # plot the route in orange

        # plot the comeback in orange
        e_last = [route[-1], route[0]]
        plt.plot(x[e_last], y[e_last], color='orange')

        plt.plot(x, y, 'o', color='blue') # plot the nodes in blue

    def copy(self):
        '''
        copy the problem with all its attributes and methods
        '''
        
        return deepcopy(self)
