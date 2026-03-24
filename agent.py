"""
agent.py — Q-learning AI agent

Q-learning is a reinforcement learning algorithm. The agent explores a
maze, collecting rewards and punishments, and gradually learns which
action (up/down/left/right) is best from each cell.

The "Q-table" is the agent's memory: a dictionary mapping each cell the
agent has visited to a list of 4 values — one per action — representing
the expected total reward for taking that action from that cell.

In Java terms: think of q_table as a HashMap<int[], double[]>,
where the key is the cell position and the value is an array of
4 expected rewards (one per direction).
"""

import random


# Directions as (row_delta, col_delta)
ACTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]   # up, down, left, right
ACTION_NAMES = ["up", "down", "left", "right"]


class QLearningAgent:
    def __init__(
        self,
        rows,
        cols,
        alpha=0.1,          # Learning rate: how fast we update Q-values (0=never, 1=immediately)
        gamma=0.9,          # Discount factor: how much we care about future rewards vs immediate
        epsilon=1.0,        # Starting exploration rate: 1.0 = fully random at start
        epsilon_decay=0.995,# How fast exploration shrinks each step
        epsilon_min=0.05,   # Minimum exploration — always keep a little randomness
    ):
        self.rows = rows
        self.cols = cols
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        # The Q-table: maps (row, col) → [q_up, q_down, q_left, q_right]
        # It starts empty; entries are created on first visit to a cell.
        self.q_table = {}

        self.actions = ACTIONS
        self.pos = (0, 0)      # Current position in maze coords
        self.steps = 0         # Steps taken this episode
        self.episode = 0       # How many episodes have completed
        self.total_reward = 0  # Accumulated reward this episode

    # ------------------------------------------------------------------
    # Q-table access
    # ------------------------------------------------------------------

    def get_q(self, state):
        """
        Return the Q-value list for a given state.
        If we've never visited this cell, initialize all 4 action values to 0.

        In Java: like map.getOrDefault(state, new double[]{0,0,0,0})
        """
        if state not in self.q_table:
            self.q_table[state] = [0.0, 0.0, 0.0, 0.0]
        return self.q_table[state]

    def get_max_q(self, state):
        """Highest Q-value the agent currently assigns to any action at this cell."""
        return max(self.get_q(state))

    def get_best_action(self, state):
        """Index (0–3) of the action the agent currently believes is best at this cell."""
        return max(range(4), key=lambda a: self.get_q(state)[a])

    # ------------------------------------------------------------------
    # Decision-making
    # ------------------------------------------------------------------

    def choose_action(self, state):
        """
        Epsilon-greedy policy:
          - With probability epsilon → pick a RANDOM action (explore)
          - Otherwise → pick the action with the highest Q-value (exploit)

        Early on epsilon is high so the agent mostly explores.
        Over time epsilon decays so the agent exploits what it's learned.
        This is the classic explore-vs-exploit tradeoff.
        """
        if random.random() < self.epsilon:
            return random.randint(0, 3)         # random action
        return self.get_best_action(state)      # best known action

    # ------------------------------------------------------------------
    # Learning
    # ------------------------------------------------------------------

    def update(self, state, action, reward, next_state):
        """
        The Q-learning update rule:

            Q(s, a) ← Q(s, a) + α * [reward + γ * max Q(s') − Q(s, a)]

        - reward: what we actually got (e.g., -1 for a step, +100 for exit)
        - max Q(s'): the best we expect to do from the next state
        - The bracketed term is the "TD error" — how surprised we were

        In plain English: nudge our estimate toward the actual outcome.
        alpha controls how big that nudge is.
        """
        q_vals = self.get_q(state)
        next_q_vals = self.get_q(next_state)

        td_target = reward + self.gamma * max(next_q_vals)
        td_error = td_target - q_vals[action]
        q_vals[action] += self.alpha * td_error

    def decay_epsilon(self):
        """
        Gradually reduce the exploration rate so the agent shifts from
        random wandering to confident exploitation of learned knowledge.
        """
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    # ------------------------------------------------------------------
    # Reset helpers
    # ------------------------------------------------------------------

    def reset_episode(self):
        """Reset position and counters for a new episode (maze still intact)."""
        self.pos = (0, 0)
        self.steps = 0
        self.total_reward = 0

    def reset_memory(self):
        """
        Wipe the Q-table completely and restart from scratch.
        Bound to the R key — lets you watch the agent re-learn a changed maze.
        """
        self.q_table = {}
        self.epsilon = 1.0
        self.pos = (0, 0)
        self.steps = 0
        self.episode = 0
        self.total_reward = 0
