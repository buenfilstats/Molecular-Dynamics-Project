What I sent:
1. TSLC jupyter notebook file, which uses the TSLCSim to define a mockSimulation object.
2. TSLCSim file
3. picture of how the methods call each other and what steps of our paper(Algorithm 1 on page 3)
4. When I was using the REAP code I noticed that in the REAP-ReinforcementLearningBasedAdaptiveSampling/toyPotential/LPotential/RL/RLSim.py
file (but this was a while ago so I might be forgetting) that the weights would sometimes go negative because a constraint wasn't being
enforced in the updateW method. I also noticed that it wasn't using SLSQP to solve the optimziation problem(which is what the REAP paper
says it uses), which made it a little slower than when you do use SLSQP. I don't know if this was in the other files as well but I 
thought I should mention it; I included the way I was using it.

I used python 3.5.6. I think the only packages you need are matplotlib, scipy, pytorch, and numpy.