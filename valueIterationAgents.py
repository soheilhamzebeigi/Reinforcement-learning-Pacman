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


import mdp, util

from learningAgents import ValueEstimationAgent
import collections


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
        self.values = util.Counter()
        self.runValueIteration()


    def runValueIteration(self):
        """
        terminal states have no reward, set val to 0
        self.values is a dictionary of {s : q_val, ....}
        I want to update the values associated with s
         """
        "*** YOUR CODE HERE ***"
        for i in range(self.iterations):
            update_value = self.values.copy()
            for s in self.mdp.getStates():
                final_state = self.mdp.isTerminal(s)
                QValues = [float('-inf')]
                if final_state:
                    update_value[s] = 0
                else:
                    legals = self.mdp.getPossibleActions(s)
                    for a in legals:
                        QValues.append(self.getQValue(s, a))
                    update_value[s] = max(QValues)
            self.values = update_value


    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
          Q-Values: expected future utility from a q-state (chance node)
          Q*(s, a) = sum_s' of T(s, a, s') [R(s, a, s') + gamma V*(s')]
        """
        "*** YOUR CODE HERE ***"
        Utility_Of_Value = []
        Next_states = self.mdp.getTransitionStatesAndProbs(state, action)
        for i, j in Next_states:
            gamma_V = self.discount * self.values[i]
            Rew = self.mdp.getReward(state, action, i)
            Utility_Of_Value.append(j * (Rew + gamma_V))
        return sum(Utility_Of_Value)


    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.
          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
          This is policy Extraction
          V*(s) = max_a Q*(s, a)
          return the action with the highest Q-value
        """
        "*** YOUR CODE HERE ***"
        legals = self.mdp.getPossibleActions(state)
        if len(legals) == 0:
            return None
        Qvalue_action = []
        for action in legals:
            qvalue = self.getQValue(state, action)
            Qvalue_action.append((action, qvalue))
        best_Qvalue = max(Qvalue_action, key=lambda x: x[1])[0]
        return best_Qvalue

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
        self.runValueIteration()


    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        states = self.mdp.getStates()
        for s in states:
            self.values[s] = 0
        States_numbers = len(states)
        for i in range(self.iterations):
            index_Of_state = i % States_numbers
            current_state = states[index_Of_state]
            Final = self.mdp.isTerminal(current_state)
            if not Final:
                action = self.getAction(current_state)
                qvalue = self.getQValue(current_state, action)
                self.values[current_state] = qvalue


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
        self.runValueIteration()


    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        fathers = {}
        priorities = util.PriorityQueue()
        states = self.mdp.getStates()
        for s in states:
            self.values[s] = 0
            fathers[s] = self.get_predecessors(s)
        for s in states:
            Final = self.mdp.isTerminal(s)
            if not Final:
                state_value = self.values[s]
                difference = abs(state_value - self.max_Qvalue(s))
                priorities.push(s, -difference)
        for h in range(self.iterations):
            if priorities.isEmpty():
                return
            s = priorities.pop()
            self.values[s] = self.max_Qvalue(s)
            for p in fathers[s]:
                diff = abs(self.values[p] - self.max_Qvalue(p))
                if diff > self.theta:
                    priorities.update(p, -diff)


    def max_Qvalue(self, state):
        return max([self.getQValue(state, f) for f in self.mdp.getPossibleActions(state)])

    def get_predecessors(self, state):
        harakat = ['north', 'south', 'east', 'west']
        states = self.mdp.getStates()
        set_of_fathers = set()
        if not self.mdp.isTerminal(state):
            for i in states:
                Final = self.mdp.isTerminal(i)
                legals = self.mdp.getPossibleActions(i)
                if not Final:
                    for j in harakat:
                        if j in legals:
                            act = self.mdp.getTransitionStatesAndProbs(i, j)
                            for k, l in act:
                                if (k == state) and (l > 0):
                                    set_of_fathers.add(i)
        return set_of_fathers