import numpy as np
import sys
from contextlib import closing
from io import StringIO
from gym import Env, spaces, utils
from gym.utils import seeding

UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3

def categorical_sample(prob_n, np_random):
    """
    Sample from categorical distribution
    Each row specifies class probabilities
    """
    prob_n = np.asarray(prob_n)
    csprob_n = np.cumsum(prob_n)
    return (csprob_n > np_random.rand()).argmax()


class DiscreteWithinBoxEnv(Env):

    """
    Inpired from 
    https://github.com/openai/gym/blob/ee5ee3a4a5b9d09219ff4c932a45c4a661778cd7/gym/envs/toy_text/discrete.py 
    
    Has the following members
    - nS: number of states
    - nA: number of actions
    - P: transitions (*)
    - isd: initial state distribution (**)
    (*) dictionary of lists, where
      P[s][a] == [(probability, nextstate, reward, done), ...]
    (**) list or array of length nS
    """
    def __init__(self, nS, nA, P, isd):
        self.P = P
        self.isd = isd
        self.lastaction = None  # for rendering
        self.nS = nS
        self.nA = nA

        self.action_space = spaces.Discrete(self.nA)

        self.observation_space = spaces.Box(low=np.array([0]), high=np.array([self.nS]),  shape=(1, ), dtype=np.int32)

        self.seed()
        self.s = np.array([int(categorical_sample(self.isd, self.np_random))])

        self.max_episode_length = 200
        self.episode_length = 0

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        self.episode_length = 0
        self.s = np.array([int(categorical_sample(self.isd, self.np_random))])
        self.lastaction = None
        return self.s

    def step(self, a):
        transitions = self.P[self.s[0]][a]
        i = categorical_sample([t[0] for t in transitions], self.np_random)
        p, s, r, d = transitions[i]
        print("input state, action, output state ",(self.s,a,s))
        self.s = np.array([int(s)])
        self.lastaction = a
        self.episode_length += 1
        done = (self.episode_length >= self.max_episode_length) or d
        return (s, r, done, {"prob": p})

class BoxSpaceCliffWalker(DiscreteWithinBoxEnv, utils.EzPickle):
    """
    This is a simple implementation of the Gridworld Cliff
    reinforcement learning task with Box state space.
    With inspiration from:
    https://github.com/dennybritz/reinforcement-learning/blob/master/lib/envs/cliff_walking.py
    """
    metadata = {'render.modes': ['human', 'ansi']}

    def __init__(self):
        utils.EzPickle.__init__(self)
        self.shape = (4, 12)
        self.start_state_index = np.ravel_multi_index((3, 0), self.shape)

        nS = np.prod(self.shape)
        nA = 4

        # Cliff Location
        self._cliff = np.zeros(self.shape, dtype=np.bool)
        self._cliff[3, 1:-1] = True

        # Calculate transition probabilities and rewards
        P = {}
        for s in range(nS):
            position = np.unravel_index(s, self.shape)
            P[s] = {a: [] for a in range(nA)}
            P[s][UP] = self._calculate_transition_prob(position, [-1, 0])
            P[s][RIGHT] = self._calculate_transition_prob(position, [0, 1])
            P[s][DOWN] = self._calculate_transition_prob(position, [1, 0])
            P[s][LEFT] = self._calculate_transition_prob(position, [0, -1])

        # Calculate initial state distribution
        # We always start in state (3, 0)
        isd = np.zeros(nS)
        isd[self.start_state_index] = 1.0

        super(BoxSpaceCliffWalker, self).__init__(nS, nA, P, isd)

    def _limit_coordinates(self, coord):
        """
        Prevent the agent from falling out of the grid world
        :param coord:
        :return:
        """
        coord[0] = min(coord[0], self.shape[0] - 1)
        coord[0] = max(coord[0], 0)
        coord[1] = min(coord[1], self.shape[1] - 1)
        coord[1] = max(coord[1], 0)
        return coord

    def _calculate_transition_prob(self, current, delta):
        """
        Determine the outcome for an action. Transition Prob is always 1.0.
        :param current: Current position on the grid as (row, col)
        :param delta: Change in position for transition
        :return: (1.0, new_state, reward, done)
        """
        new_position = np.array(current) + np.array(delta)
        new_position = self._limit_coordinates(new_position).astype(int)
        new_state = np.ravel_multi_index(tuple(new_position), self.shape)
        if self._cliff[tuple(new_position)]:
            return [(1.0, self.start_state_index, -100, False)]

        terminal_state = (self.shape[0] - 1, self.shape[1] - 1)
        is_done = tuple(new_position) == terminal_state
        return [(1.0, new_state, -1, is_done)]

    def render(self, mode='human'):
        outfile = StringIO() if mode == 'ansi' else sys.stdout

        for s in range(self.nS):
            position = np.unravel_index(s, self.shape)
            if self.s == s:
                output = " x "
            # Print terminal state
            elif position == (3, 11):
                output = " T "
            elif self._cliff[position]:
                output = " C "
            else:
                output = " o "

            if position[1] == 0:
                output = output.lstrip()
            if position[1] == self.shape[1] - 1:
                output = output.rstrip()
                output += '\n'

            outfile.write(output)
        outfile.write('\n')

        # No need to return anything for human
        if mode != 'human':
            with closing(outfile):
                return outfile.getvalue()