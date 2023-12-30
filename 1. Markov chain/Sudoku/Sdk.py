from copy import deepcopy
import numpy as np
from numpy import random as rnd


def check_error(array):
    '''
    count the violations to the sudoku rules
    
    parameters
    ----------
        array: list to inspect
    
    returns
    ----------
        err: number of violations found in the array
    '''

    # check that the array is not empty and return 0 violations otherwise
    if array.size == 0:
        err = 0
    # count the number of repetitions in the array eventually
    else:
        err = array.size-np.unique(array).size

    return err


class Generate:
    '''
    create a sudoku as a numpy array

    parameters
    ----------
        n: size of the schema
        sn: square root of the size
        r: rate of spots that are set and fixed
        mask: slice of spots that are left empty to be filled in
        table: schema of the sudoku
        seed: integer seed for reproducibility
    
    methods
    ----------
        set_mask: create the sudoku mask
        set_table: create the sudoku table
        compute_cost: compute the total cost of the sudoku
        display: print the sudoku schema
        copy: copy the problem
    '''

    def __init__(
            self,
            n: int,
            r: float,
            seed: int=None
        ):

        # set the seed if necessary
        if seed is not None:
            np.seed(seed)

        # store the schema variables
        self.n = n
        self.sn = int(np.sqrt(n))
        self.r = r

        self.mask = self.set_mask() # create the sudoku mask
        # create the sudoku table
        self.table = np.ones((n, n), dtype=int)*(-1)
        self.set_table()
    
    # overload the print operator
    def __repr__(self):
        '''
        represent the problem
        '''

        return str(self.table) # print the schema
    
    # overload the sum operator
    def __add__(self, num):
        '''
        sum a number to the problem

        parameters
        ----------
            self: the problem
            num: number to add to the problem
        
        returns
        ----------
            add: sum of the problem and the number
        '''

        add = self.table+num # sum
        return add

    def set_mask(self):
        '''
        create the sudoku mask respecting the given rate

        returns
        ----------
            mask: slice of spots that are left empty to be filled in
        '''
        
        n, r = self.n, self.r

        mask = np.zeros((n, n), dtype=bool) # initialize the mask
        # fill in the sudoku mask
        for i in range(n):
            for j in range(n):
                mask[i, j] = rnd.choice((0, 1), p=(1-r, r))

        return mask

    def set_table(self):
        '''
        create the sudoku table on the basis of the sudoku mask
        '''
        
        n, mask, table = self.n, self.mask, self.table

        c = np.inf # initialize the cost
        # loop until all the errors in the schema are fixed
        while c != 0:
            table[mask] = rnd.choice(np.arange(n), table[mask].size, replace=True) # generate new random numbers
            c = self.compute_cost()

    def compute_cost(self):
        '''
        compute the total cost of the sudoku counting the errors according to the rules

        returns
        ----------
            c: total cost of the sudoku
        '''

        n, sn, mask, table = self.n, self.sn, self.mask, self.table

        c = 0 # initialize the cost

        # count the errors in the whole schema
        for i in range(n):
            c += check_error(table[i][mask[i]]) # count the errors horizontally
            c += check_error(table.T[i][mask.T[i]]) # count the errors vertically

        # count the errors in the subsquares
        for si in range(sn):
            for sj in range(sn):
                table_ij = table[si*sn:(si+1)*sn, sj*sn:(sj+1)*sn] # slice the table
                mask_ij = mask[si*sn:(si+1)*sn, sj*sn:(sj+1)*sn] # slice the mask
                c += check_error (table_ij[mask_ij])

        return c

    def display(self):
        '''
        print the sudoku schema in the terminal
        '''

        n, sn, table = self.n, self.sn, self.table
        grid = table.copy()+1 # convert numbers to 0-9

        sdk = str() # initialize the string
        # loop over the rows
        for i in range(n):
            sdk += '| '

            # loop over the number of small squares
            for j in range(sn):
                sub = grid[i, j*sn:(j+1)*sn] # slice the subsquare

                # loop over the subsquare
                for k in sub:
                    # check if the spot is empty and leave a space eventually
                    if k == 0:
                        sdk += '  '

                    # fill it with the number otherwise
                    else:
                        sdk += f'{k} '

                sdk += '| '

            # check that the whole subsquare has been printed and close it eventually
            if i in [h*sn-1 for h in range (1, sn)]:
                sdk += '\n-------------------------\n'

            # don't close it otherwise
            else:
                sdk += '\n'

        print(sdk)

    def copy(self):
        '''
        copy the problem with all its attributes and methods
        '''

        return deepcopy(self)


class Import:
    '''
    import a sudoku from a numpy array

    parameters
    ----------
        table: numpy array to produce the sudoku schema
        n: size of the schema
        sn: square root of the size
        mask: slice of spots that are left empty to be filled in

    methods
    ----------
        display: print the sudoku schema
    '''

    def __init__(
            self,
            table: np.ndarray
        ):

        # store the schema variables
        self.n = len(table)
        self.sn = int(np.sqrt(self.n))

        self.table = table-1 # use -1 as sentinel value

        # generate a mask to hide the gaps to fill in
        self.mask = np.full(self.n**2, 0, dtype=bool).reshape((self.n, self.n))
        self.mask[(table != 0)] = 1

    # overload the print operator
    def __repr__(self):
        '''
        represent the problem
        '''

        return str(self.table) # print the schema
    
    # overload the sum operator
    def __add__(self, num):
        '''
        sum a number to the problem

        parameters
        ----------
            self: the problem
            num: number to add to the problem
        
        returns
        ----------
            add: sum of the problem and the number
        '''

        add = self.table+num # sum
        return add

    def display(self):
        '''
        print the sudoku schema in the terminal
        '''

        n, sn, table = self.n, self.sn, self.table
        grid = table.copy()+1 # convert numbers to 0-9 

        sdk = str() # initialize the string
        # loop over the rows
        for i in range(n):
            sdk += '| '

            # loop over the number of small squares
            for j in range(sn):
                sub = grid[i, j*sn:(j+1)*sn] # slice the subsquare

                # loop over the subsquare
                for k in sub:
                    # check if the spot is empty and leave a space eventually
                    if k == 0:
                        sdk += '  '

                    # fill it with the number otherwise
                    else:
                        sdk += f'{k} '

                sdk += '| '

            # check that the whole subsquare has been printed and close it eventually
            if i in [h*sn-1 for h in range (1, sn)]:
                sdk += '\n-------------------------\n'

            # don't close it otherwise
            else:
                sdk += '\n'

        print(sdk)

    def copy(self):
        '''
        copy the problem with all its attributes and methods
        '''

        return deepcopy(self)


class Solve:
    '''
    solve a sudoku

    parameters
    ----------
        n: size of the schema
        sn: square root of the size
        mask: slice of spots that are left empty to be filled in
        table: schema of the sudoku
        seed: integer seed for reproducibility
    
    methods
    ----------
        init_config: set the initiaĺ random combination for the problem
        propose_move: propose a change to the combination
        accept_move: accept the proposed change to the combination
        compute_delta_cost: compute the cost difference associated to a change to the combination
        compute_cost: compute the total cost of the combination
        display: print the sudoku schema
        copy: copy the problem
    '''
    
    def __init__(
            self,
            obj,
            seed: int=None
        ):

        # set the seed if necessary
        if seed is not None:
            np.seed(seed)

        # store the schema variables
        self.n = obj.n
        self.sn = int(np.sqrt(obj.n))
        self.table = obj.table
        self.mask = obj.mask
    
    # overload the print operator
    def __repr__(self):
        '''
        represent the problem
        '''

        return str(self.table) # print the schema
    
    # overload the sum operator
    def __add__(self, num):
        '''
        sum a number to the problem

        parameters
        ----------
            self: the problem
            num: number to add to the problem
        
        returns
        ----------
            add: sum of the problem and the number
        '''

        add = self.table+num # sum
        return add
    
    def init_config(self):
        '''
        set the initial random combination for the problem
        '''

        n, table, mask = self.n, self.table, self.mask

        for i in range(n):
            col = table.T[i] # get the column
            mask_fix = mask.T[i] # get the fixed spots in the column

            # get the numbers for the empty spots discarding the fixed ones
            l = np.arange(n)
            l = np.delete(l, col[mask_fix])

            col[(mask_fix == False)] = rnd.permutation(l) # fill in the empty spots
    
    def propose_move(self):
        '''
        propose a change to the combination swapping two spots within the same column

        returns
        ----------
            i: column of the matrix
            j: first spot to swap
            k: second spot to swap
        '''
        
        n, mask = self.n, self.mask

        i = rnd.randint(n) # select a column
        # distinguish the spots that are fixed from the ones that are left empty
        fixed = mask.T[i]
        free = np.where(fixed == False)[0]

        # choose two spots to swap within the same column
        moves = np.arange(n)
        j, k = rnd.choice(moves[free], 2, replace=False) # sample without replacement

        return i, j, k
    
    def accept_move(
            self,
            move: tuple
        ):
        '''
        accept the proposed change to the combination swapping two spots within the same column

        parameters
        ----------
            move: tuple containing the column and the spots to swap
        '''
        
        i, j, k = move

        self.table[j, i], self.table[k, i] = self.table[k, i], self.table[j, i] # swap the spots within the same column
 
    def compute_delta_cost(
            self,
            move: tuple
        ):
        '''
        compute the cost difference associated to a change to the combination virtually accepting the move

        parameters
        ----------
            move: tuple containing the column and the spots to swap

        returns
        ----------
            c_delta: cost difference between the old combination and the alternative one
        '''

        # virtually accept the move on a copy of the problem
        c_old = self.compute_cost()
        table_new = self.copy()
        table_new.accept_move(move)
        c_new = table_new.compute_cost() # compute the cost of the modification on the copy

        c_delta = c_new-c_old # compute the cost difference
        return c_delta

    def compute_cost(self):
        '''
        compute the total cost of the combination counting the errors according to the rules

        returns
        ----------
            c: total cost of the combination
        '''
        
        n, sn, table = self.n, self.sn, self.table

        c = 0 # initialize the cost
    
        # count the errors in the whole schema
        for i in range (n):
            c += check_error(table[i]) # count the errors horizontally
            c += check_error(table.T[i]) # count the errors vertically

        # count the errors in the subsquares
        for si in range(sn):
            for sj in range(sn):
                table_ij = table[si*sn:(si+1)*sn, sj*sn:(sj+1)*sn] # slice the table
                c += check_error(table_ij)

        return c

    def display(self):
        '''
        print the sudoku schema in the terminal
        '''

        n, sn, table = self.n, self.sn, self.table
        grid = table.copy()+1 # convert numbers to 0-9 

        sdk = str() # initialize the string
        # loop over the rows
        for i in range(n):
            sdk += '| '

            # loop over the number of small squares
            for j in range(sn):
                sub = grid[i, j*sn:(j+1)*sn] # slice the subsquare

                # loop over the subsquare
                for k in sub:
                    # check if the spot is empty and leave a space eventually
                    if k == 0:
                        sdk += '  '

                    # fill it with the number otherwise
                    else:
                        sdk += f'{k} '

                sdk += '| '

            # check that the whole subsquare has been printed and close it eventually
            if i in [h*sn-1 for h in range (1, sn)]:
                sdk += '\n-------------------------\n'

            # don't close it otherwise
            else:
                sdk += '\n'

        print(sdk)

    def copy(self):
        '''
        copy the problem with all its attributes and methods
        '''

        return deepcopy(self)
