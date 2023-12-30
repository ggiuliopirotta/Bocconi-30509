from Msq import *
from Msq_Optimize import *


msq = Magic_Squares(
    n=6,
    s=1520
) # create the magic squares problem

solution = sim_ann(
    msq,
    iters=5000,
    b0=2,
    b1=100,
    ann_steps=40,
    seed=1
) # optimize the problem with a simulated annealing algorithm
