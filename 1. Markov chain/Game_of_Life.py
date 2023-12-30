import matplotlib.pyplot as plt
import numpy as np


class Game_of_Life:
    '''
    represent the classic **Conway's game of life**

    parameters
    ----------
        size: size of the board
        board: table of cells
        seed: integer for reproducibility
    
    methods
    ----------
        set_alive: set the initial alive cells for the game
        play: make the cells interact with each other for a given number of steps
        display: show how the cells interact on the board
    '''
    
    def __init__(
            self,
            size: int,
            seed: int=None
        ):

        # set the seed if necessary
        if seed is not None:
            np.seed(seed)

        # store the board variables
        self.size = size
        self.board = np.zeros((size, size)) # initialize the board

    def set_alive(
            self,
            r: float
        ):
        '''
        set the initial alive cells for the game respecting the given rate

        parameters
        ----------
            r: rate of alive cells
        '''

        size = self.size

        self.board = np.array(np.random.rand(size, size) < r, dtype=int) # set the alive cells

    def play(
            self,
            iters: int
        ):
        '''
        compute the cells' status on the board after a given number of interactions

        parameters
        ----------
            iters: number of interactions
        
        returns
        ----------
            board: table of cells after the interactions 
        '''
        
        size = self.size

        # recompute the cells' status after each step
        for _ in range(iters):
            board = self.board
            nearest = np.zeros((size, size))

            # loop over each cell on the board and identify the neighbors
            for i in range(size):
                row_i = np.zeros(size) # initialize the neighbors count

                for j in range(size):
                    row_i[j] = np.sum(board[max(i-1, 0):min(i+2, size), max(j-1, 0):min(j+2, size)]) - board[i, j] # count the numbers of neighbors alive

                nearest[i] = row_i # set the count

            alive = board*nearest # get the new status for previously alive cells
            # adjust the new status according to the rules
            alive[alive >= 4] = 0
            alive[alive <= 2] = 0
            alive[alive != 0] = 1

            dead = (board == 0)*nearest # get the new status for previously alive cells
            # adjust the new status according to the rules
            dead[dead < 3] = 0
            dead[dead != 0] = 1

            # get the final new board and plot it
            self.board = alive+dead
            self.display()
        
        return self.board

    def display(self):
        '''
        show how the cells interact on the board step after step
        '''

        board, size = self.board, self.size

        plt.pause(0.1)
        plt.figure(1, figsize=(7, 7))
        plt.clf()

        # plot the grid for a better visualization
        for i in range(size):
            plt.plot([-1, size], [i-0.5, i-0.5], color='grey', alpha=0.7)
            plt.plot([i-0.5, i-0.5], [-1, size], color='grey', alpha=0.7)

        plt.imshow(board) # plot the board


game = Game_of_Life(size=20) # create the game of life

game.set_alive(r=0.22) # configure the cells' status for the game
board = game.play(iters=100) # let the cells interact for some steps
plt.show()
