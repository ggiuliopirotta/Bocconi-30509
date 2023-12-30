from Wmc import *
from Wmc_Optimize import *


wmc = Wmc(
    n=20,
    p=0.94
) # create the weighted max cut problem

greedy(
    wmc,
    iters=10000,
    restarts=1,
    plot=True
) # optimize the problem with a greedy algorithm

# sim_ann(
#     wmc,
#     iters=5000,
#     b0=0.5,
#     b1=20,
#     ann_steps=10
# ) # optimize the problem with a simulated annealing algorithm
