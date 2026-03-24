# Adaptive Labyrinth

A two-player adversarial maze simulator. A Q-learning AI tries to navigate a maze — but the maze is alive. It reads the AI's memory and deliberately reshapes itself to undo everything the AI has learned.

---

## What is this?

Two competing intelligences:

- **The AI agent** learns the maze using Q-learning, building a table of which moves are best from each cell.
- **The Maze** reads that table every 50 steps and uses it against the agent — blocking its favorite paths, moving the exit when it gets close, and more.

It's a cat-and-mouse game between a learner and its environment.

---

## How to run

**1. Install Python 3.8+** — [python.org](https://python.org)

**2. Install pygame:**
```bash
pip install pygame
```

**3. Run:**
```bash
python main.py
```

---

## Controls

| Key | Action |
|-----|--------|
| `Space` | Pause / unpause |
| `↑` / `↓` | Speed up / slow down |
| `←` / `→` | Decrease / increase malice level |
| `R` | Wipe AI memory and regenerate maze |

---

## What you're watching

| Visual | Meaning |
|--------|---------|
| Cyan glowing dot | The AI agent |
| Green cell | The exit — reach it for +100 reward |
| Yellow flash | Walls the maze just mutated |
| ε (epsilon) in HUD | Exploration rate — how often the AI acts randomly |

---

## Malice level

Controls how aggressively the maze fights the AI (adjust with `←` / `→`):

| Level | Behavior |
|-------|----------|
| `0.0` | Static maze — AI trains normally |
| `0.3` | Mild interference, one strategy per turn |
| `0.7` | Aggressive — multiple strategies per turn |
| `1.0` | Full adversarial mode, all strategies every turn |

---

## Maze strategies

The adversary picks from four strategies each turn. Higher malice means more strategies fire simultaneously.

### Confidence Destroyer
Finds the cells where the AI has the highest Q-values (its "safe zones" where it's most certain) and flips walls around those cells. The agent's most trusted knowledge becomes wrong.

### False Progress
Simulates the AI's believed best path to the exit by following its Q-table greedily. Then blocks the middle of that path — turning "the route I know works" into a dead end.

### Déjà Vu
Copies wall patterns from cells the AI has already explored and pastes them into unexplored areas. When the AI enters new territory, it looks familiar — causing the AI to make wrong assumptions about what routes exist.

### Exit Runner
When the AI gets close to the exit, teleports the exit somewhere far away and walls off the old approach corridor. All the Q-values pointing toward the old exit location suddenly lead nowhere.

---

## How Q-learning works (quick version)

The AI keeps a Q-table: for every cell it's visited, it stores 4 numbers — the expected total reward for going up, down, left, or right from there.

At each step:
1. It picks the action with the highest Q-value (or a random action if still exploring)
2. It moves, hits a wall, or reaches the exit
3. It updates the Q-value using the actual reward it received

Over many episodes, the Q-values converge on the optimal path — *unless the maze keeps changing them*.

**Rewards:** `+100` for reaching the exit · `-1` per step · `-5` for hitting a wall

---

## Project structure

```
adaptive-labyrinth/
├── main.py        # Game loop, controls, episode management
├── maze.py        # Maze generation (recursive backtracking) and wall mutation
├── agent.py       # Q-learning agent
├── adversary.py   # Adversarial brain — all four strategies
├── renderer.py    # Pygame visualization and HUD
└── requirements.txt
```
