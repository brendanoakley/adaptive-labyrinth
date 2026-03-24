"""
main.py — Game loop and entry point

This file ties everything together:
  1. Creates the maze, agent, adversary, and renderer
  2. Runs the main game loop: handle input → step agent → (maybe) step adversary → draw
  3. Manages episodes: when the agent reaches the exit (or times out), reset for next round
  4. Handles all keyboard controls

Run with:
    python main.py
"""

import sys
import pygame

from maze import Maze
from agent import QLearningAgent
from adversary import AdversaryBrain
from renderer import Renderer

# -----------------------------------------------------------------------
# Configuration — tweak these to change the feel of the simulation
# -----------------------------------------------------------------------

ROWS = 15           # Maze height in rooms
COLS = 20           # Maze width in rooms

# How often the adversary gets a turn (every N agent steps)
ADVERSARY_INTERVAL = 50

# Max steps before an episode is considered a timeout and resets
MAX_STEPS_PER_EPISODE = ROWS * COLS * 25

# Available simulation speeds (frames per second)
# Lower FPS = slower, easier to watch. Higher = faster training.
SPEEDS = [4, 8, 15, 30, 60, 120]
DEFAULT_SPEED_IDX = 3   # starts at 30 FPS

# Starting malice level
DEFAULT_MALICE = 0.5


# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------

def make_world(rows, cols):
    """
    Create a fresh maze, agent, and adversary.
    Called at startup and when the player presses R.
    Returns all three so main() can reassign them.
    """
    maze = Maze(rows, cols)
    agent = QLearningAgent(rows, cols)
    adversary = AdversaryBrain(maze, agent)
    return maze, agent, adversary


def step_agent(maze, agent):
    """
    Run one decision-action-learn cycle for the agent.

    The agent:
      1. Looks at its current position (the state)
      2. Chooses an action (epsilon-greedy)
      3. Tries to move; gets blocked by walls
      4. Receives a reward
      5. Updates its Q-table

    Returns (reward, done) where done=True means the agent reached the exit.
    """
    state = agent.pos
    action_idx = agent.choose_action(state)
    dr, dc = agent.actions[action_idx]
    nr, nc = state[0] + dr, state[1] + dc

    # Check if the move is valid (in bounds and no wall)
    in_bounds = 0 <= nr < maze.rows and 0 <= nc < maze.cols
    if in_bounds:
        wr, wc = 2 * state[0] + 1 + dr, 2 * state[1] + 1 + dc
        wall_hit = maze.grid[wr][wc] == 1
    else:
        wall_hit = True  # treat out-of-bounds as a wall

    if wall_hit:
        next_state = state    # stay put
        reward = -5           # penalty for bumping a wall
    else:
        next_state = (nr, nc)
        reward = -1           # small penalty per step (encourages efficiency)

    done = next_state == maze.exit
    if done:
        reward = 100          # big reward for reaching the exit

    # Update Q-table with what we just learned
    agent.update(state, action_idx, reward, next_state)
    agent.pos = next_state
    agent.steps += 1
    agent.total_reward += reward

    return reward, done


# -----------------------------------------------------------------------
# Main loop
# -----------------------------------------------------------------------

def main():
    pygame.init()

    maze, agent, adversary = make_world(ROWS, COLS)
    renderer = Renderer(maze)

    clock = pygame.time.Clock()
    speed_idx = DEFAULT_SPEED_IDX
    malice = DEFAULT_MALICE
    paused = False
    episode = 0
    steps = 0

    while True:
        # --- Event handling -------------------------------------------
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            elif event.type == pygame.KEYDOWN:

                if event.key == pygame.K_SPACE:
                    paused = not paused

                elif event.key == pygame.K_UP:
                    # Speed up simulation
                    speed_idx = min(len(SPEEDS) - 1, speed_idx + 1)

                elif event.key == pygame.K_DOWN:
                    # Slow down simulation
                    speed_idx = max(0, speed_idx - 1)

                elif event.key == pygame.K_LEFT:
                    # Decrease malice (maze fights back less)
                    malice = max(0.0, round(malice - 0.1, 1))

                elif event.key == pygame.K_RIGHT:
                    # Increase malice (maze fights back more)
                    malice = min(1.0, round(malice + 0.1, 1))

                elif event.key == pygame.K_r:
                    # Full reset: new maze AND wipe the agent's memory
                    maze, agent, adversary = make_world(ROWS, COLS)
                    renderer.maze = maze        # renderer needs new maze ref
                    renderer.flash_timers = {}  # clear any leftover flashes
                    episode = 0
                    steps = 0

        # --- Simulation step ------------------------------------------
        if not paused:
            _, done = step_agent(maze, agent)
            steps += 1

            # Decay exploration rate every 10 steps
            if steps % 10 == 0:
                agent.decay_epsilon()

            # Adversary acts every ADVERSARY_INTERVAL steps
            if steps % ADVERSARY_INTERVAL == 0:
                adversary.act(malice)
                renderer.add_flashes(adversary.flash_cells)

            # Episode ends when agent reaches exit or times out
            if done or steps >= MAX_STEPS_PER_EPISODE:
                episode += 1
                steps = 0
                agent.reset_episode()
                agent.episode = episode

                # Every 15 episodes, regenerate the maze entirely
                # (keeps the simulation from getting stale)
                if episode % 15 == 0:
                    maze, agent_new, adversary = make_world(ROWS, COLS)
                    # Keep the agent's learned Q-table across maze regenerations
                    # so it has to unlearn old knowledge in a new layout
                    agent_new.q_table = agent.q_table
                    agent_new.epsilon = agent.epsilon
                    agent_new.episode = episode
                    agent = agent_new
                    adversary.agent = agent
                    renderer.maze = maze
                    renderer.flash_timers = {}

        # --- Render ---------------------------------------------------
        renderer.draw(agent, episode, steps, malice, paused)
        clock.tick(SPEEDS[speed_idx])


if __name__ == "__main__":
    main()
