### Assignment 2 Part 2
**Team 66** - Ahana Datta (2019111007), Tanvi Narsapur (2019111005)
__________
### Task 1
> Iterations = 125
> Gamma = 0.999
> Step cost = -20

The value iteration algorithm converges in 125 iterations. We observe the following from the trace file - 
- If MM is in D (dormant) state, then IJ is usually risk taking, i.e, if for example IJ was at S then he will try to go to C so that he can defeat MM instead of staying at S and hiding.
- If MM is in R (ready) state, then IJ is usually observed to be risk aversing. IJ tries to stay in S if MM is in R state.
- IJ can't carry more than 2 materails and 3 arrows at any time during the game
- IJ can shoot only if he has at least 1 arrow and if he is located in 'C', 'E' or 'W' positions.
- IJ can gather material when he is in the S state. IJ usually STAYS if MM is in ready state and usually goes UP if MM is Dormant. Exceptions occur when IJ has material and few or no arrows. In that case he tries to go UP to eventually reach N and craft arrows and defeat MM!
- At E, IJ almost always either SHOOTS or HITS so as to eventually kill the monster and win the game. When the health of MM becomes 0, IJ gets 50 reward and wins the game. Also, the probability for arrow hitting MM and blade making contact with MM is more at E than at any other position. So he tries to defeat MM when at E.
- When IJ is at W, he usually tries to go to C if MM is dormant and otherwise SHOOTS or STAYS. There is an exception when IJ has no arrows and MM is Ready. In this case he tries to move RIGHT to C so that he can HIT MM.
- For the case when IJ is at N position, if IJ has no material, he prefers to stay at the same position when MM is in ready state and when MM is in dormant state, IJ prefers to go down. The same is observed when IJ has 3 arrows and any number of material. When IJ has material, he usually chooses the CRAFT action. IJ usually crafts arrows if he has at least 1 material and has less than 3 arrows.
- At C, IJ mostly performs RIGHT action when MM is dormant. Deviations are seen when IJ has material and few or no arrows, that's when IJ decides to go UP to N so that he can CRAFT arrows using the material. When MM is ready, IJ goes UP mostly if he has material and few or no arrows, SHOOTS if he has arrows, and goes RIGHT if he has no arrows and/or no material.


Simulations based on the policy obtained from the value iteration algorithm - 

For starting position (W, 0, 0, D, 100):
| Current state | Best Action |Success| Next state |
|----------------|----------|---------|-------|
|(W, 0, 0, D, 100)| RIGHT |Yes| (C, 0, 0, D, 100)|
|(C, 0, 0, D, 100)| RIGHT |Yes|(E, 0, 0, D, 100)|
| (E, 0, 0, D, 100)| HIT | No|(E, 0, 0, R, 100)|
| (E, 0, 0, R, 100)| HIT | Yes|(E, 0, 0, R, 50)|
| (E, 0, 0, R, 50)| HIT | No (MM attacked)|(E, 0, 0, D, 75)|
| (E, 0, 0, D, 75)| HIT |Yes| (E, 0, 0, D, 25)|
| (E, 0, 0, D, 25)| HIT |No| (E, 0, 0, R, 25)|
| (E, 0, 0, R, 25)| HIT |No (MM attacked)| (E, 0, 0, D, 50)|
| (E, 0, 0, D, 50)| HIT |Yes| (E, 0, 0, D, 0)|
| (E, 0, 0, D, 0)|NONE|-| GAME OVER

For the second starting position (C, 2, 0, R, 100):
| Current state | Best Action | Success | Next state |
|----------------|----------|------|----------|
| (C, 2, 0, R, 100) | UP | Yes | (N, 2, 0, R, 100) |
| (N, 2, 0, R, 100) | CRAFT | Yes (2 arrows) | (N, 1, 2, R, 100) |
| (N, 1, 2, R, 100) | CRAFT | Yes (1 arrow) |(N, 0, 3, R, 100) |
| (N, 0, 3, R, 100) | STAY | No | (E, 0, 3, R, 100) |
| (E, 0, 3, R, 100) | HIT | Yes | (E, 0, 3, R, 50) |
| (E, 0, 3, R, 50) | SHOOT | No | (E, 0, 2, R, 50) |
| (E, 0, 2, R, 50) | SHOOT | Yes | (E, 0, 1, R, 25) |
| (E, 0, 1, R, 25) | SHOOT | No (MM attacked) | (E, 0, 0, D, 50) |
| (E, 0, 0, D, 50) | HIT | No | (E, 0, 0, R, 50) |
| (E, 0, 0, R, 50) | HIT | No (MM attacked) | (E, 0, 0, D, 75)|
| (E, 0, 0, D, 75)| HIT | Yes | (E, 0, 0, D, 25)|
| (E, 0, 0, D, 25)| HIT | Yes | (E, 0, 0, D, 0)|
| (E, 0, 0, D, 0)| NONE | - | GAME OVER |

### Task 2.1
> Iterations = 126
> Gamma = 0.999
> Step cost = -20
> IJ on LEFT action at East Square will go to the West Square. 

As compared to task 1, IJ chooses the left action in state E more often. This is observed because after choosing the left action in the E state IJ goes to W state from where he can shoot MM without being affected by MM attack. 
126 iterations are required for the algorithm to converge.

### Task 2.2
> Iterations = 63
> Gamma = 0.999
> Step cost = -20, Step cost for STAY action = 0

We observed from the trace file that, when IJ is in C or E and MM is in D state and its health is 25, then IJ always SHOOTS if he has an arrow. Also, if state of MM is R and his health is 25, then Indiana chooses the actions SHOOT, when Indiana is located at C or E. For all the other cases IJ prefers the STAY action over all the other actions. In general, IJ decides to STAY more than he did in task 1 since the step cost of staying is now 0. For W, IJ always performs STAY action. This shows that IJ is risk averse as he stays in W to prevent getting attacked by MM and cost for STAY is 0. As a result of this, utility of W positions is always 0 because STAY at W leads to no step cost and no reward in any iteration.
The algorithm converges in 63 iterations.

### Task 2.3
> Iterations = 9
> Gamma = 0.25
> Step cost = -20

The discount factor gamma is the factor by which contraction takes place in the value iteration algorithm. It is representing the preference for short term solutions over long term solutions. Value of gamma closer to 1, gives more importance to future state utilities in determining current utility value. On the other hand, gamma closer to 0, gives lesser importance to future state utilities. So for smaller values of gamma, the algorithm converges at a faster rate. But this may result in missing out on the long term effects of the agent’s actions.
For gamma=0.25, the algorithm converges in 9 iterations.
