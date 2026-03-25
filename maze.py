"""
maze.py — Maze generation and mutation engine

The maze is stored as a 2D grid where:
  - Room cells sit at odd (row, col) positions
  - Wall cells sit at even row or even col positions

For a maze with R rows and C cols, the grid is (2R+1) x (2C+1).
Room at maze coords (r, c) → grid coords (2r+1, 2c+1)
Wall between (r,c) and (r,c+1) → grid coords (2r+1, 2c+2)
Wall between (r,c) and (r+1,c) → grid coords (2r+2, 2c+1)

grid[gr][gc] == 1 means WALL (blocked)
grid[gr][gc] == 0 means OPEN (passable)
"""

import random
from collections import deque


class Maze:
    def __init__(self, rows, cols):
        self.rows = rows
        self.cols = cols

        # Start at top-left, exit at bottom-right
        self.start = (0, 0)
        self.exit = (rows - 1, cols - 1)

        # Initialize everything as walls (all 1s)
        self.grid = [[1] * (2 * cols + 1) for _ in range(2 * rows + 1)]

        # Carve a perfect maze using recursive backtracking
        self._generate()

    # ------------------------------------------------------------------
    # Maze generation
    # ------------------------------------------------------------------

    def _generate(self):
        """
        Iterative recursive backtracking (also called "randomized DFS").

        Think of it like a depth-first search: start at (0,0), carve a
        passage to a random unvisited neighbor, keep going until stuck,
        then backtrack. The result is a "perfect maze" — exactly one path
        between any two cells, which guarantees solvability.

        We use an explicit stack instead of Python recursion so we don't
        hit Python's recursion limit on large mazes.
        """
        visited = set()

        # Open all room cells (the cells the agent can actually stand on)
        for r in range(self.rows):
            for c in range(self.cols):
                self.grid[2 * r + 1][2 * c + 1] = 0

        start = (0, 0)
        visited.add(start)
        stack = [start]

        while stack:
            r, c = stack[-1]

            # Find unvisited neighbors in maze coordinates
            neighbors = []
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if (0 <= nr < self.rows and 0 <= nc < self.cols
                        and (nr, nc) not in visited):
                    neighbors.append((dr, dc, nr, nc))

            if neighbors:
                # Pick a random unvisited neighbor and carve through
                dr, dc, nr, nc = random.choice(neighbors)
                # Remove the wall between current cell and chosen neighbor
                self.grid[2 * r + 1 + dr][2 * c + 1 + dc] = 0
                visited.add((nr, nc))
                stack.append((nr, nc))
            else:
                # Dead end — backtrack
                stack.pop()

    # ------------------------------------------------------------------
    # Helper utilities
    # ------------------------------------------------------------------

    def room_to_grid(self, r, c):
        """Convert maze room coords (r, c) → grid coords (gr, gc)."""
        return (2 * r + 1, 2 * c + 1)

    def get_neighbors(self, r, c):
        """
        Return all rooms reachable from room (r, c) — i.e., adjacent rooms
        where the wall between them is currently open.
        Used for pathfinding (BFS/DFS).
        """
        neighbors = []
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < self.rows and 0 <= nc < self.cols:
                # The wall cell between (r,c) and (nr,nc)
                wr, wc = 2 * r + 1 + dr, 2 * c + 1 + dc
                if self.grid[wr][wc] == 0:  # wall is open
                    neighbors.append((nr, nc))
        return neighbors

    def is_solvable(self, agent_pos):
        """
        BFS from the agent's current position to the exit.
        Returns True if a path exists, False if the exit is cut off.
        Called after every mutation to guarantee the maze stays beatable.
        """
        if agent_pos == self.exit:
            return True

        visited = {agent_pos}
        queue = deque([agent_pos])

        while queue:
            r, c = queue.popleft()
            for nr, nc in self.get_neighbors(r, c):
                if (nr, nc) == self.exit:
                    return True
                if (nr, nc) not in visited:
                    visited.add((nr, nc))
                    queue.append((nr, nc))
        return False

    # ------------------------------------------------------------------
    # Wall mutation helpers
    # ------------------------------------------------------------------

    def try_add_wall(self, r, c, dr, dc, agent_pos):
        """
        Attempt to add a wall between room (r,c) and its neighbor in
        direction (dr, dc). Only commits if:
          - The neighbor is in bounds
          - Neither cell is the agent or the exit
          - The maze remains solvable after the change

        Returns True on success, False if the wall was rejected.
        """
        nr, nc = r + dr, c + dc
        if not (0 <= nr < self.rows and 0 <= nc < self.cols):
            return False

        # Never trap the agent or seal off the exit cell itself
        if (r, c) in (agent_pos, self.exit) or (nr, nc) in (agent_pos, self.exit):
            return False

        wr, wc = 2 * r + 1 + dr, 2 * c + 1 + dc
        if self.grid[wr][wc] == 1:   # already a wall, nothing to do
            return False

        # Tentatively add the wall, then verify solvability
        self.grid[wr][wc] = 1
        if not self.is_solvable(agent_pos):
            self.grid[wr][wc] = 0   # revert — this wall would trap the agent
            return False

        return True

    def try_remove_wall(self, r, c, dr, dc):
        """
        Remove the wall between room (r,c) and its neighbor.
        Removing a wall always keeps the maze solvable (it can only add
        connections, never cut them), so no solvability check needed.

        Returns True on success, False if there was no wall to remove.
        """
        nr, nc = r + dr, c + dc
        if not (0 <= nr < self.rows and 0 <= nc < self.cols):
            return False

        wr, wc = 2 * r + 1 + dr, 2 * c + 1 + dc
        if self.grid[wr][wc] == 0:   # already open
            return False

        self.grid[wr][wc] = 0
        return True

    # ------------------------------------------------------------------
    # Pathfinding & fingerprinting
    # ------------------------------------------------------------------

    def shortest_path(self, start, goal):
        """
        BFS to find the fewest-steps path from start to goal.
        Returns a list of (r, c) cells start→goal (inclusive),
        or None if no path exists.

        Used to cache the provably optimal route once the agent
        finds the exit for the first time on a given maze layout.
        """
        if start == goal:
            return [start]

        # came_from maps each cell to the cell we arrived from.
        # Following backpointers from goal → start lets us reconstruct
        # the full path at the end (like a parent pointer in Java BFS).
        came_from = {start: None}
        queue = deque([start])

        while queue:
            r, c = queue.popleft()
            for nr, nc in self.get_neighbors(r, c):
                if (nr, nc) not in came_from:
                    came_from[(nr, nc)] = (r, c)
                    if (nr, nc) == goal:
                        # Reconstruct by walking backpointers
                        path = []
                        node = goal
                        while node is not None:
                            path.append(node)
                            node = came_from[node]
                        return list(reversed(path))
                    queue.append((nr, nc))
        return None   # goal is unreachable

    def signature(self):
        """
        A hash that uniquely identifies the current maze state.
        Changes whenever any wall is added/removed OR the exit moves.

        Used to check whether a cached optimal path is still valid —
        if the signature matches, the path is safe to replay.

        In Java terms: like overriding hashCode() so you can use the
        maze as a key in a HashMap.
        """
        grid_hash = hash(tuple(tuple(row) for row in self.grid))
        return (grid_hash, self.exit)
