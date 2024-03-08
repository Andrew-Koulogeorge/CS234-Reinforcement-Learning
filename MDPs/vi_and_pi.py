### MDP Value Iteration and Policy Iteration
# Andrew Koulogeorge | Winter 2024 
import argparse
import numpy as np
import gymnasium as gym
import time
from lake_envs import *

np.set_printoptions(precision=3)

parser = argparse.ArgumentParser(
    description="A program to run assignment 1 implementations.",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)

parser.add_argument(
    "--env",
    type=str,
    help="The name of the environment to run your algorithm on.",
    choices=["Deterministic-4x4-FrozenLake-v0", "Stochastic-4x4-FrozenLake-v0"],
    default="Deterministic-4x4-FrozenLake-v0",
)

parser.add_argument(
    "--render-mode",
    "-r",
    type=str,
    help="The render mode for the environment. 'human' opens a window to render. 'ansi' does not render anything.",
    choices=["human", "ansi"],
    default="human",
)

"""
For policy_evaluation, policy_improvement, policy_iteration and value_iteration,
the parameters P, nS, nA, gamma are defined as follows:

	P: nested dictionary of a nested lists
		From gym.core.Environment
		For each pair of states in [1, nS] and actions in [1, nA], P[state][action] is a
		tuple of the form (probability, nextstate, reward, terminal) where
			- probability: float
				the probability of transitioning from "state" to "nextstate" with "action"
			- nextstate: int
				denotes the state we transition to (in range [0, nS - 1])
			- reward: int
				either 0 or 1, the reward for transitioning from "state" to
				"nextstate" with "action"
			- terminal: bool
			  True when "nextstate" is a terminal state (hole or goal), False otherwise
	nS: int
		number of states in the environment
	nA: int
		number of actions in the environment
	gamma: float
		Discount factor. Number in range [0, 1)
"""


def policy_evaluation(P, nS, nA, policy, gamma=0.9, tol=1e-3):
    """Evaluate the value function from a given policy.

	Parameters
	----------
	P, nS, nA, gamma:
		defined at beginning of file
	policy: np.array[nS]
		The policy to evaluate. Maps states to actions.
	tol: float
		Terminate policy evaluation when
			max |value_function(s) - prev_value_function(s)| < tol
	Returns
	-------
	value_function: np.ndarray[nS]
		The value function of the given policy, where value_function[s] is
		the value of state s
	"""

    # random init of the value function
    value_function = np.zeros(nS) 

    ############################
    # YOUR IMPLEMENTATION HERE #
    
    while True: # loop until convergence := none of the states in a sweep change by more than tol

        delta = 0
        for state in range(nS): # loop over all states 

            curr_val_state = value_function[state] # get the old value function at state i
            new_val_state = 0

            # dynamics could be stochastic; a (s,a) pair could have different next states and rewards to consider
            # updating value of state from (1) reward obtained from transitioning to next state (2) bootstrapped discounted value of next state
            for pr, next_state, transition_reward, terminal in P[state][policy[state]]: # considering all transitions from this (s,a) pair
                
                if terminal: next_state_val = 0 # val of terminal state = 0 since we cannot act afterwards
                else: next_state_val = value_function[next_state] # bootstrapping the value of the next state with current estimate

                new_val_state += pr * (transition_reward + gamma * next_state_val)
            
            delta = max(delta, abs(curr_val_state - new_val_state)) # keep track of how much we updated our estimate of the value of this state
            value_function[state] = new_val_state # update value function

        if delta < tol: 
            print(f"Evaluation of this policy converged!")
            break

    ############################
    return value_function


def policy_improvement(P, nS, nA, value_from_policy, policy, gamma=0.9):
    """
    Given the value function from policy improve the policy.

	Parameters
	----------
	P, nS, nA, gamma:
		defined at beginning of file
	value_from_policy: np.ndarray
		The value calculated from the policy
	policy: np.array
		The previous policy.

	Returns
	-------
	new_policy: np.ndarray[nS]
		An array of integers. Each integer is the optimal action to take
		in that state according to the environment dynamics and the
		given value function.
	"""

    new_policy = policy.copy() # pi: states --> actions

    ############################
    # YOUR IMPLEMENTATION HERE #

    while True: # for each state, want best action based on the value function of the current policy
        
        stable_policy = True # flag to keep track if we changed our policy 
    
        for state in range(nS): # loop over all of the states

            best_val, best_action = 0,0

            for action in range(nA): # for each state, consider all actions we could take from this state

                q_sa = 0 
                for pr, next_state, transition_reward, terminal in P[state][action]: # dynamics induce distribution over next states given (s,a) pair
                    
                    if terminal: next_state_val = 0
                    else: next_state_val = value_from_policy[next_state]
                    
                    q_sa += pr * (transition_reward + gamma*next_state_val) # compute expected return of taking action a in state s 
                
                if q_sa > best_val: # if this action is better than any other action so far, update values
                    best_val = q_sa
                    best_action = action
            
            # comparing the init action to the new action. If the policy has changed, not stable.
            if new_policy[state] != best_action:
                new_policy[state] = best_action
                stable_policy = False
        
        if stable_policy:
            print(f"Policy improvment for this value function converged!")
            break

    ############################
    return new_policy


def policy_iteration(P, nS, nA, gamma=0.9, tol=1e-3):
    """Runs policy iteration.

	You should call the policy_evaluation() and policy_improvement() methods to
	implement this method.

	Parameters
	----------
	P, nS, nA, gamma:
		defined at beginning of file
	tol: float
		tol parameter used in policy_evaluation()
	Returns:
	----------
	value_function: np.ndarray[nS]
	policy: np.ndarray[nS]
	"""

    value_function = np.zeros(nS)
    policy = np.zeros(nS, dtype=int)

    ############################
    # YOUR IMPLEMENTATION HERE #

    # iterativley evaluate the policy to get the value function, then imporve your policy by acting greedily wrt the value function. 
    # do this until policy converges (actions taken in alls states stops changing)
    steps_to_converge = 0
    while True:
        
        steps_to_converge +=1
        value_function = policy_evaluation(P, nS, nA, policy, gamma, tol) # evaluate policy 
        old_policy = policy.copy() # hold onto old policy to see if anything changed
        policy = policy_improvement(P, nS, nA, value_function, old_policy, gamma) # improve the policy

        if np.array_equal(policy, old_policy): # if the policy didnt change after improvment
            print(f"It took {steps_to_converge} many steps for policy iteration to converge")
            print(f"Policy: {policy}\n")
            print(f"Value Function: {value_function}\n")
            break

    ############################
    return value_function, policy


def value_iteration(P, nS, nA, gamma=0.9, tol=1e-3):
    """
	Learn value function and policy by using value iteration method for a given
	gamma and environment.

	Parameters:
	----------
	P, nS, nA, gamma:
		defined at beginning of file
	tol: float
		Terminate value iteration when
			max |value_function(s) - prev_value_function(s)| < tol
	Returns:
	----------
	value_function: np.ndarray[nS]
	policy: np.ndarray[nS]
	"""

    # init value function and policy
    value_function = np.zeros(nS)
    policy = np.zeros(nS, dtype=int)

    ############################
    # YOUR IMPLEMENTATION HERE #

    # implement a value function which considers the best action (taking the max over each actions)
    while True: # looping until value function convergence
        
        delta = 0
        for state in range(nS):# loop over all states 

            val_state = value_function[state] # get the old value function at state i
            
            for action in range(nA): # considering max over all actions, not considering what the current policy even is
                q_sa = 0

                for pr, next_state, transition_reward, terminal in P[state][action]:
                    
                    if terminal:next_state_val = 0
                    else: next_state_val = value_function[next_state]
                    q_sa += pr * (transition_reward + gamma * next_state_val)
                
                value_function[state] = max(value_function[state], q_sa)

            delta = max(delta, value_function[state] - val_state )

        if delta < tol: break
    
    old_policy = policy.copy() # hold onto old policy to see if anything changed
    policy = policy_improvement(P, nS, nA, value_function, old_policy, gamma) # improve the policy

    print(f"Policy: {policy}\n")
    print(f"Value Function: {value_function}\n")
    

    ############################
    return value_function, policy


def render_single(env, policy, max_steps=100):
    """
    This function does not need to be modified
    Renders policy once on environment. Watch your agent play!

    Parameters
    ----------
    env: gym.core.Environment
      Environment to play on. Must have nS, nA, and P as
      attributes.
    Policy: np.array of shape [env.nS]
      The action to take at a given state
  """

    episode_reward = 0
    ob, _ = env.reset()
    for t in range(max_steps):
        env.render()
        time.sleep(0.25)
        a = policy[ob]
        ob, rew, done, _, _ = env.step(a)
        episode_reward += rew
        if done:
            break
    env.render()
    if not done:
        print(
            "The agent didn't reach a terminal state in {} steps.".format(
                max_steps
            )
        )
    else:
        print("Episode reward: %f" % episode_reward)


# Edit below to run policy and value iteration on different environments and
# visualize the resulting policies in action!
# You may change the parameters in the functions below
if __name__ == "__main__":
    # read in script argument
    args = parser.parse_args()

    # Make gym environment
    env = gym.make(args.env, render_mode=args.render_mode)

    env.nS = env.nrow * env.ncol
    env.nA = 4

    print("\n" + "-" * 25 + "\nBeginning Policy Iteration\n" + "-" * 25)


    V_pi, p_pi = policy_iteration(env.P, env.nS, env.nA, gamma=0.9, tol=1e-3)

    render_single(env, p_pi, 100)

    print("\n" + "-" * 25 + "\nBeginning Value Iteration\n" + "-" * 25)

    V_vi, p_vi = value_iteration(env.P, env.nS, env.nA, gamma=0.9, tol=1e-3)

    render_single(env, p_vi, 100)

