JV

CS23M036, Malladi Tejasvi

This is the repo for the CS6700, Reinforcement Learning, IIT Madras, Programming assignment 3.

The primary goal is to implement one step SMDP Q learning and One step Intra Option Q learning as per the original papers, for the Taxi-v3 environment in Gymnasium.

This implementation uses Gymnasium v0.29.1 and Numpy v1.22.3

"Standard Options" Refer to 4 options, each terminating upon reaching a designated location (R/G/Y/B).
"Custom Options" Refer to a set of 2 options (mutually exclusive with the "Standard Options"), corresponding to picking up and dropping the passenger respectively.

The primitive actions are in any case there along with each of the above two variants.

JV_OneStep_IntraOption_QLearning.py is the code that implements the one step SMDP Q learning.

JV_OneStep_SMDP_QLearning.py is the code that implements the one step Intra Option Q learning.

The local variable "non_primitive_options" in both of the above codes helps in the instantiation of an agent.

Setting:

      non_primitive_options = 4, uses the "Standard Options"
      non_primitive_options = 2, uses the "Custom Options"

Both the codes automatically generate the reward curve and the Q-Value heatmap visualizations and create a directory named after the algorithm and save the plots in that directory.
The plots are not displayed during the execution of the code.

The hyperparamters, Viz. alpha, number of episodes, epsilon starting value, etc could be tuned as required.
