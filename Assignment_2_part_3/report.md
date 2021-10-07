### Assignment 2 Part 3
**Team 66** - Ahana Datta (2019111007), Tanvi Narsapur (2019111005)
__________

#### Construction of A matrix:
- A State object with 5 attributes is used to uniquely identify each state. 
- The object has 5 attributes as follows:
        - IJ's position: {'C', 'E', 'W', 'N', 'S'}
        - Material: {0, 1, 2}
        - Arrows: {0, 1, 2, 3}
        - MM's state: {'D', 'R'}
        - MM's health: {0, 1, 2, 3, 4} correspoding to health values {0, 25, 50, 75, 100}
- There are total 600 states and 1936 possible state-action pairs. Thus the dimension of the A matrix is 600x1936.
- Each state corresponds to a unique row in the A matrix. The row number corresponding to each state is calculated using the `state_to_hash()` function.
- For each state, all the possible actions are computed based on the given constraints. The possible actions are returned by the `get_actions()` function.
- Corresponding to every possible state, we iterate over all the valid action-state pairs. Every state-action pair corresponds to a column of the A matrix which will contain zero at all the places except at the index for the current state and the indices for the next states where our actions take us to.
- Based on the constraints, we are calling either `get_d_next_states()`  or `get_r_next_states()` to obtain the next possible states corresponding to the state-action pair.
- For the index corresponding to the current state, the value placed will be equal to the summation of probabilities of all the valid actions, which is equal to 1.
- Positive value corresponds to the outflowing action and negative value corresponds to the incoming action. 
- For example, if action 'a' takes us from state S to S' with probability p, then p is placed at the row for state S and column corresponding to S-a pair and -p is placed at the row for state S' and column corresponding to S'-a pair, since the action 'a' is an outflowing action for state S and an inflowing action for state S'.
- We are Ignoring self-looping transitions of a valid action to avoid unbounded LP. For this purpose, in case of self looping actions, we are adding as well as subtracting the probability value corresponding to the self looping action which is equivalent to ignoring the self loop.

#### Computing the policy:
- Policy defines the sequence of actions followed and it is represented using State and action pair.
- The A matrix, reward vector R and the alpha vector are used for generation of X vector.
- The A matrix is a two dimensional matrix which contains the flow of probabilities for the possible actions for each state.
- The reward vector R is an one dimensional vector that contains the reward values corresponding to the possible actions for each state. In case of multiple possible rewards, the expectation of all the possible values is considered based on the probability values.
- The alpha vector is an one dimensional vector that contains initial probabilities for all the states. As the start state is (C,2,3,R,100), the value 1 is placed at the index corresponding to the chosen start state and zeroes are placed at all the other places.
- The X vector contains the utility values for each state and action pair. This vector is computed by passing the alpha vector, A matrix and reward vector R to the LPP solver function.
- In this LPP we aim to maximize the value of RX subject to the following constraints 
        - AX = alpha
        - X >= 0
- After obtaining the X vector, we iterate over all the X values corresponding to a particular state. We find out the maximum from these X values for the state using `np.argmax()` and the action corresponding to this X value is included in the policy. The policy is computed using the `get_policy()` function.
#### Analysis:
- If MM is in D (dormant) state, then IJ is usually risk taking, i.e, if for example IJ was at S then he will try to go to C so that he can defeat MM instead of staying at S and hiding.
- If MM is in R (ready) state, then IJ is usually observed to be risk aversing. IJ tries to stay in S if MM is in R state.
- IJ can't carry more than 2 materails and 3 arrows at any time during the game
- IJ can shoot only if he has at least 1 arrow and if he is located in 'C', 'E' or 'W' positions.
- IJ can gather material when he is in the S state. IJ usually STAYS if MM is in ready state and usually goes UP if MM is Dormant. Exceptions occur when IJ has material and few or no arrows. In that case he tries to go UP to eventually reach N and craft arrows and defeat MM!
- At E, IJ almost always either SHOOTS or HITS so as to eventually kill the monster and win the game. Also, the probability for arrow hitting MM and blade making contact with MM is more at E than at any other position. So he tries to defeat MM when at E.
- When IJ is at W, he usually tries to go to C if MM is dormant and otherwise SHOOTS or STAYS. There is an exception when IJ has no arrows and MM is Ready. In this case he tries to move RIGHT to C so that he can HIT MM.

#### Can Multiple policies exist?

Yes, there can be multiple policies. As there can be different actions with the same X value for a state, we can have multiple policies.

1. We can obtain a different policy by changing the start state. This will change the alpha vector and we will obtain a different policy. We can also change the step-cost value associated with state-action pairs. This will alter the reward vector and we will get a different policy. Similarly, if we change the values of probabilities for executing the actions, the policy will be different as the state diagram itself is changed. But changing the start state, the step-cost or the probability values changes the question posed. We can obtain a different policy for the same question using the following -
2. We can change the way we are computing the maximum X value for a particular state. Using `np.argmax()` we get the index corresponding to the first occurence of the highest value in case many $x_i$'s have the same highest value. Instead of this, we can iterate as x[i]>=x[i+1] which will give us the index corresponding to the last occurence of the highest value. Thus we obtain a different policy.
3. Another way can be changing the order in which the actions are considered. The current order is UP, DOWN, RIGHT, LEFT, STAY, SHOOT, HIT, CRAFT, GATHER, NONE. 
- Changing the order in which the actions are considered does not alter the A matrix since it is formed before solving the LPP. 
- Alpha vector is unchanged as the initial probabilities are not changed. 
- The reward vector R remains unchanged as it is calculated before solving the LPP, using the reward values associated with each state-action pair.
- The X matrix remains unchanged as X is computed by passing the A matrix, R and alpha vector to the LPP solver and all three are unaltered. But the actions that take place to form X vary for a different policy.