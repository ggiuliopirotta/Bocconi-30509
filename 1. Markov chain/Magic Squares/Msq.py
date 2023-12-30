from copy import deepcopy
from numpy import random as rnd
import numpy as np


def get_diagonals(mat: np.ndarray):
    '''
    slice the two main diagonals out of the given square matrix

    parameters
    ----------
        n: size of the matrix
        mat: matrix to slice
    
    returns
    ----------
        diag1: major diagonal
        diag2: minor diagonal
    '''

    arr = np.arange(len(mat)) # use an array to slice the matrix

    diag1 = mat[arr, arr] # slice the major diagonal
    diag2 = mat[arr, arr[::-1]] # slice the minor diagonal

    return diag1, diag2


def check_error(
        arr: np.ndarray,
        s: int
    ):
    '''
    check if the numbers the given array sum up to the desired total
    
    parameters
    ----------
        arr: array to check
        s: desired total to respect
    
    returns
    ----------
        err: absolute error between the sum of the numbers in the array and the desired total
    '''

    err = abs(s-np.sum(arr)) # compute the absolute error
    return err


class Magic_Squares:
    '''
    represent the **magic squares problem**

    the problem asks to arrange some positive numbers in a matrix of given dimensions,
    such that rows, columns and main diagonals sum up to the same desired total

    parameters
    ----------
        n: size of the matrix
        s: desired total to respect
        table: matrix containing positive numbers only
        seed: integer seed for reproducibility
    
    methods
    ----------
        init_config: set the initiaĺ random combination for the problem
        propose_move: propose a change to the combination
        accept_move: accept the proposed change to the combination
        compute_delta_cost: compute the cost difference associated to a change to the combination
        compute_cost: compute the total cost of the combination
        display: print the magic squares matrix
        copy: copy the problem
    '''

    def __init__(
            self,
            n: int,
            s: int,
            seed: int=None
        ):

        # set the seed if necessary
        if seed is not None:
            np.seed(seed)

        # store the problem variables
        self.n = n
        self.s = s
        self.table = rnd.randint(1, s-n+2, size=(n, n))

    # overload the print operator
    def __str__(self):
        '''
        print the problem
        '''
        
        s = f'n: {self.n}, s: {self.s}\n{self.table}'
        return s

    def init_config(self):
        '''
        set the initial random combination for the problem
        '''
        
        n, s = self.n, self.s

        # loop over the columns
        for i in range(n):
            self.table.transpose()[i] = rnd.randint(1, s-n+2, size=n) # fill each column with random positive integers

    def propose_move(self):
        '''
        propose a change to the combination increasing or decreasing a number by 1

        returns
        ----------
            i: row of the matrix
            j: column of the matrix
            num: new number to replace the old one
        '''

        n, table = self.n, self.table
        
        i, j = rnd.randint(n, size=2) # choose a spot
        num = rnd.choice([max(table[i, j]-1, 1), table[i, j]+1]) # increase or decrease the selected number by 1

        return i, j, num

    def accept_move(
            self,
            move: tuple
        ):
        '''
        accept the proposed change to the combination increasing or decreasing a number by 1

        parameters
        ----------
            move: tuple containing the selected the row, the column and whether to increase or decrease the number
        '''

        i, j, num = move
        
        self.table[i, j] = num # replace the old number with the new one

    def compute_delta_cost(
            self,
            move: tuple
        ):
        '''
        compute the cost difference associated to a change to the combination virtually accepting the move

        parameters
        ----------
            move: tuple containing the selected the row, the column and whether to increase or decrease the number
        
        returns
        ----------
        c_delta: cost difference between the old combination and the alternative one
        '''
        n, s, table = self.n, self.s, self.table
        i, j, num = move

        # slice the row and the column of the number
        row_old, col_old = table[i], table.transpose()[j]
        row_new, col_new = row_old.copy(), col_old.copy()

        row_new[j], col_new[i] = num, num # replace the number in the copy

        c_delta = 0 # initialize the delta cost
        c_delta = check_error(row_new, s) + check_error(col_new, s) # add the error in the new combination
        c_delta -= check_error(row_old, s) + check_error(col_old, s) # subtract the error in the old combination

        # check that the number belongs to the major diagonal and account for the error in the major diagonal eventually
        if i == j:
            # slice the major diagonal
            diag1_old = get_diagonals(table)[0]
            diag1_new = diag1_old.copy()

            diag1_new[j] = num # replace the number in the copy
            c_delta += check_error(diag1_new, s) - check_error(diag1_old, s) # add the error in the new combination and subtract the error in the old one

        # check that the number belongs to the minor diagonal and account for the error in the minor diagonal eventually 
        if j == n-1-i:
            # slice the minor diagonal
            diag2_old = get_diagonals(table)[1]
            diag2_new = diag2_old.copy()

            diag2_new[i] = num # replace the number in the copy
            c_delta += check_error(diag2_new, s) - check_error(diag2_old, s) # add the error in the new combination and subtract the error in the old one

        return c_delta

    def compute_cost(self):
        '''
        compute the total cost of the combination counting the errors according to the rules

        returns
        ----------
            c: total cost of the combination
        '''

        n, s, table = self.n, self.s, self.table

        c = 0 # initialize the cost

        # add all the errors to the cost
        for i in range(n):
            c += check_error(table[i], s) # count the errors in the rows
            c += check_error(table.transpose()[i], s) # count the errors in the columns
        
        # count the errors in the diagonals
        diag1, diag2 = get_diagonals(table)
        c += check_error(diag1, s) + check_error(diag2, s)

        return c

    def copy(self):
        '''
        copy the problem with all its attributes and methods
        '''

        return deepcopy(self)
