import matplotlib.pyplot as plt
import numpy as np


class Random_Walk:
    '''
    represent the trajectoies of a number of random walkers on a plane

    parameters
    ----------
        n_walkers: number of random walkers on the board
        size: size of the board
        p_marg: marginal probability for the walkers' location
        seed: integer for reproducibility
    
    methods
    ----------
        walk: compute the marginal probability for the walkers' location for a given number of steps
        display_walk: plot the walkers' location
        display_prob: plot the marginal probability distribution
    '''
    
    def __init__(
            self,
            n_walkers: int,
            size: int,
            seed: int=None
        ):

        # set the seed if necessary
        if seed is not None:
            np.seed(seed)

        # store the variables
        self.n_walkers = n_walkers
        self.size = size
        self.p_marg = np.zeros(size**2) # initialize the probability matrix

    def walk(
            self,
            iters: int,
            show_prob: bool=False
        ):
        '''
        perform a given number of random walk steps for all the walkers

        parameters
        ----------
            iters: number of steps to walk
            show_prob: boolean for marginal probability display
        '''

        n_walkers, size = self.n_walkers, self.size

        # define the possible moves and the probabilities associated
        moves = np.array([[1,0], [-1,0], [0,1], [0,-1]])
        p = np.array([1/4, 1/4, 1/4, 1/4])

        # place all walkers in the center of the board
        coord = np.zeros((n_walkers, 2))
        coord[:,1] = size//2
        coord[:,0] = size//2
        self.p_marg[size//2+size*(size//2)] = 1 # set the probability of being in the center to 1

        Q = np.zeros((size**2, size**2)) # initialize the transition probability matrix

        # loop over all spots in the board and get each coordinates identifiers
        for i in range(size**2):
            j = i//size
            k = i %size

            # loop over all possible moves and compute the transition probability
            for m_i, p_i in zip(moves, p):
                # get the new coordinates identifiers after the move
                j_new = (j+m_i[1]) %size
                k_new = (k+m_i[0]) %size
                Q[size*j_new+k_new, i] += p_i # sum the transition probability

        # loop over all iterations
        for _ in range(iters):
            coord_old = coord.copy()
            # get the new location after the move
            coord += moves[np.random.choice(len(moves), n_walkers, p=p)]
            coord %= size
            self.p_marg = Q@self.p_marg # compute the new marginal probability
            self.display_walk(coord, coord_old) # plot the walkers

            # plot the marginal probability distribution if necessary
            if show_prob:
                self.display_prob()

        print(self.p_marg)
        plt.show()

    def display_walk(
            self,
            coord: np.ndarray,
            coord_old: np.ndarray
        ):
        '''
        plot the the walkers' location at the current and previous step on the board

        parameters
        ----------
            coord: walkers' coordinates at the current step
            coord_old: walkers' coordinates at the previous step
        '''

        n_walkers, size = self.n_walkers, self.size

        plt.pause(0.1)
        plt.figure(1)
        plt.clf()

        # plot the grid of the board
        for i in range(size):
            plt.plot([0, size-1], [i,i], color='grey', alpha=0.7)
            plt.plot([i,i], [0, size-1], color='grey', alpha=0.7)

        x, y = coord[:,0], coord[:,1] # get the walkers' location at the current step
        x_old, y_old = coord_old[:,0], coord_old[:,1] # get the walkers' location at the previous step

        # loop over all the walkers
        for j in range(n_walkers):
            # check that walkers aren't on the boundaries and plot their tail eventually 
            if abs(x_old[j]-x[j]) == size-1 or abs(y_old[j]-y[j]) == size-1:
                continue

            plt.plot([x_old[j], x[j]], [y_old[j], y[j]], color='orange', linewidth=3, alpha=0.7) # plot the walkers' tail

        plt.plot(x, y, 'o', color='red', markersize=10, alpha=1/np.sqrt(n_walkers)) # plot the walkers' head

    def display_prob(self):
        '''
        plot the marginal probability distribution for the walkers' location on the board
        '''

        p_marg, size = self.p_marg, self.size

        plt.pause(0.1)
        plt.figure(2)
        plt.clf()

        plt.pcolormesh(p_marg.reshape((size, size)), cmap='Greys') # plot the matrix


rnd_walk = Random_Walk(
    n_walkers=20,
    size=20
) # create an instance

rnd_walk.walk(
    iters=50,
    show_prob=False
) # perform some random walk steps
