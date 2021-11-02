# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).
import math
from collections import defaultdict
from typing import Dict, Any, Set, Tuple, Sequence
from util import PriorityQueue
from mdp import MarkovDecisionProcess
from learningAgents import ValueEstimationAgent

def find_q_value(mdp_obj, value, discount, state, transition):
    q_value = 0
    for next_state, probability in mdp_obj.getTransitionStatesAndProbs(state, transition):
        q_value +=  probability * (mdp_obj.getReward(state, transition, next_state) + discount * value[next_state])
    return q_value

def calculate_q_values(mdp_obj, value, discount, state):
    q_values = []
    for transition in mdp_obj.getPossibleActions(state):
        q_values.append((find_q_value(mdp_obj, value, discount, state, transition), transition))
    return q_values


def find_transition(mdp_obj, value, discount, state) -> Tuple[float, Tuple]:
    q_values = calculate_q_values(mdp_obj, value, discount, state)
    if not q_values:
        return 0, ()
    max_q_value, transition = max(q_values)
    max_action = tuple(action for qValue, action in q_values if qValue == max_q_value)
    return max_q_value, max_action


class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.policy = None
        self.values = defaultdict(float)
        self.runValueIteration()

    def runValueIteration(self):
        # Write value iteration code here
        "*** YOUR CODE HERE ***"
        discount = self.discount
        mdp_obj = self.mdp
        old_values = defaultdict(int)
        new_values = self.values
        transition_states = mdp_obj.getStates()
        for i in range(self.iterations):
            old_values, new_values = new_values, old_values
            for tstate in transition_states:
                q_values = calculate_q_values(mdp_obj, old_values, discount, tstate)
                if q_values:
                    max_q_value = max(q_values)[0]
                    new_values[tstate] = max_q_value
                else:
                    new_values[tstate] = 0
        self.values = new_values

    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        "*** YOUR CODE HERE ***"
        return find_q_value(self.mdp, self.values, self.discount, state, action)

    def computeActionFromValues(self, tstate):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        if self.policy is None:
            policy = self.policy = dict()
            for tstate in self.mdp.getStates():
                q_values = calculate_q_values(self.mdp, self.values, self.discount, tstate)
                policy[tstate] = max(q_values)[1] if q_values else None
        return self.policy[tstate]

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)

class AsynchronousValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        An AsynchronousValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs cyclic value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 1000):
        """
          Your cyclic value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy. Each iteration
          updates the value of only one state, which cycles through
          the states list. If the chosen state is terminal, nothing
          happens in that iteration.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state)
              mdp.isTerminal(state)
        """
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        mdp = self.mdp
        transition_states = mdp.getStates()
        for k in range(self.iterations):
            tstate = transition_states[k % len(transition_states)]
            actions = mdp.getPossibleActions(tstate)
            if not actions:
                continue
            else:
                q_values = calculate_q_values(mdp, self.values, self.discount, tstate)
                max_q_value = max(q_values)[0]
                self.values[tstate] = max_q_value

class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100, theta = 1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def get_previous_states(self, mdp) -> Dict[Any, Set]:
        transition_states = mdp.getStates()
        previous_states = {tstate: set() for tstate in transition_states}
        for tstate in transition_states:
            for transition in mdp.getPossibleActions(tstate):
                for successor, probability in mdp.getTransitionStatesAndProbs(tstate, transition):
                    if probability > 0:
                        previous_states[successor].add(tstate)
        return previous_states

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        transition_states = self.mdp.getStates()
        previous_states = self.get_previous_states(self.mdp)
        queue = PriorityQueue()
        for tstate in transition_states:
            q_values = calculate_q_values(self.mdp, self.values, self.discount, tstate)
            if not q_values:
                continue
            new_value = max(q_values)[0]
            error = abs(self.values[tstate] - new_value)
            queue.push(tstate, -error)
    
        for i in range(self.iterations):
            if queue.isEmpty():
                break
            tstate = queue.pop()
            self.values[tstate] = find_transition(self.mdp, self.values, self.discount, tstate)[0]
            for pre_state in previous_states[tstate]:
                q_values = calculate_q_values(self.mdp, self.values, self.discount, pre_state)
                if not q_values:
                    continue
                new_value = max(q_values)[0]
                error = abs(self.values[pre_state] - new_value)
                if error > self.theta:
                    queue.update(pre_state, -error)