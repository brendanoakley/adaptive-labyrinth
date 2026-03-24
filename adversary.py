"""
adversary.py — The adversarial maze brain

Every N steps the maze gets a turn. It reads the agent's Q-table —
the agent's entire memory — and uses one or more strategies to mutate
walls so that what the agent has learned becomes wrong.

The malice level (0.0–1.0) controls how aggressively strategies are
applied. At 0 nothing happens; at 1 all strategies fire every turn.

Each strategy is a separate method so they're easy to mix, tune, or
extend. They all go through try_add_wall / try_remove_wall on the Maze
so they never break solvability.
"""

import random
from collections import deque


class AdversaryBrain:
    def __init__(self, maze, agent):
        self.maze = maze
        self.agent = agent
        # List of grid coords (gr, gc) changed this turn — renderer flashes these
        self.flash_cells = []

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def act(self, malice):
        """
        Called every ADVERSARY_INTERVAL steps.
        Picks strategies proportional to malice level and runs them.
        """
        if malice <= 0:
            return

        self.flash_cells = []

        strategies = [
            self.confidence_destroyer,
            self.false_progress,
            self.deja_vu,
            self.exit_runner,
        ]

        # Higher malice → more strategies used per turn
        # At malice=0.25: 1 strategy, 0.5: 2, 0.75: 3, 1.0: all 4
        num_strategies = max(1, round(malice * len(strategies)))
        chosen = random.sample(strategies, num_strategies)

        for strategy in chosen:
            strategy(malice)

    # ------------------------------------------------------------------
    # Strategy 1: Confidence Destroyer
    # ------------------------------------------------------------------

    def confidence_destroyer(self, malice):
        """
        Find the cells the agent is MOST confident about (highest max
        Q-value) and flip walls around those cells. The agent's carefully
        learned path values suddenly point the wrong way.

        Why this hurts the agent: Q-values are only accurate if the wall
        layout stays the same. Changing walls around high-confidence cells
        forces the agent to re-learn from scratch in its "safe zones".
        """
        if not self.agent.q_table:
            return

        # Rank every visited cell by how confident the agent is there
        # (confidence = highest Q-value in that cell)
        q_scores = sorted(
            self.agent.q_table.keys(),
            key=lambda s: self.agent.get_max_q(s),
            reverse=True,
        )

        # Target the top confident cells; more malice → more targets
        n_targets = max(1, int(len(q_scores) * malice * 0.25))
        targets = q_scores[:n_targets]

        max_changes = max(2, int(malice * 6))
        changes = 0

        for r, c in targets:
            if changes >= max_changes:
                break
            dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
            random.shuffle(dirs)
            for dr, dc in dirs:
                if changes >= max_changes:
                    break
                nr, nc = r + dr, c + dc
                if not (0 <= nr < self.maze.rows and 0 <= nc < self.maze.cols):
                    continue
                wr, wc = 2 * r + 1 + dr, 2 * c + 1 + dc
                if self.maze.grid[wr][wc] == 0:
                    # Open passage → try adding a wall
                    if self.maze.try_add_wall(r, c, dr, dc, self.agent.pos):
                        self.flash_cells.append((wr, wc))
                        changes += 1
                else:
                    # Existing wall → open it up (confuses the agent too)
                    if self.maze.try_remove_wall(r, c, dr, dc):
                        self.flash_cells.append((wr, wc))
                        changes += 1

    # ------------------------------------------------------------------
    # Strategy 2: False Progress
    # ------------------------------------------------------------------

    def false_progress(self, malice):
        """
        Simulate the agent's BELIEVED best path to the exit by greedily
        following Q-values. Then block the middle of that path.

        Why this hurts the agent: the agent thinks it knows how to get to
        the exit. This strategy finds that exact route and puts a wall
        across it, turning "the way I know works" into a dead end.
        """
        agent_pos = self.agent.pos
        max_steps = self.maze.rows * self.maze.cols

        # Greedily follow Q-table to simulate where the agent THINKS it's going
        believed_path = []
        current = agent_pos
        visited_sim = {current}

        for _ in range(max_steps):
            if current == self.maze.exit:
                break
            if current not in self.agent.q_table:
                break

            best_action = self.agent.get_best_action(current)
            dr, dc = self.agent.actions[best_action]
            nr, nc = current[0] + dr, current[1] + dc

            if not (0 <= nr < self.maze.rows and 0 <= nc < self.maze.cols):
                break
            if (nr, nc) in visited_sim:
                break   # loop detected — bail out

            believed_path.append(current)
            visited_sim.add((nr, nc))
            current = (nr, nc)

        if len(believed_path) < 3:
            return  # path too short to meaningfully disrupt

        # Block the middle of the believed path
        mid = len(believed_path) // 2
        r, c = believed_path[mid]

        max_changes = max(1, int(malice * 3))
        changes = 0
        dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        random.shuffle(dirs)

        for dr, dc in dirs:
            if changes >= max_changes:
                break
            if self.maze.try_add_wall(r, c, dr, dc, self.agent.pos):
                wr, wc = 2 * r + 1 + dr, 2 * c + 1 + dc
                self.flash_cells.append((wr, wc))
                changes += 1

    # ------------------------------------------------------------------
    # Strategy 3: Déjà Vu
    # ------------------------------------------------------------------

    def deja_vu(self, malice):
        """
        Copy the wall pattern around an EXPLORED cell and paste it into
        an UNEXPLORED cell. When the agent arrives at the new area it will
        look exactly like somewhere it's already been, making its position
        estimates unreliable.

        Why this hurts the agent: Q-learning depends on the environment
        being consistent. If an unexplored corner suddenly looks like a
        known dead-end, the agent may skip it entirely — missing a path
        it hasn't actually tried yet.
        """
        explored = set(self.agent.q_table.keys())
        unexplored = [
            (r, c)
            for r in range(self.maze.rows)
            for c in range(self.maze.cols)
            if (r, c) not in explored
        ]

        if not explored or not unexplored:
            return

        n_copies = max(1, int(malice * 3))

        for _ in range(n_copies):
            src = random.choice(list(explored))
            dst = random.choice(unexplored)
            src_r, src_c = src
            dst_r, dst_c = dst

            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                src_nr, src_nc = src_r + dr, src_c + dc
                dst_nr, dst_nc = dst_r + dr, dst_c + dc

                if not (0 <= src_nr < self.maze.rows and 0 <= src_nc < self.maze.cols):
                    continue
                if not (0 <= dst_nr < self.maze.rows and 0 <= dst_nc < self.maze.cols):
                    continue

                src_wr = 2 * src_r + 1 + dr
                src_wc = 2 * src_c + 1 + dc
                src_wall = self.maze.grid[src_wr][src_wc]

                dst_wr = 2 * dst_r + 1 + dr
                dst_wc = 2 * dst_c + 1 + dc

                if src_wall == 1:
                    if self.maze.try_add_wall(dst_r, dst_c, dr, dc, self.agent.pos):
                        self.flash_cells.append((dst_wr, dst_wc))
                else:
                    if self.maze.try_remove_wall(dst_r, dst_c, dr, dc):
                        self.flash_cells.append((dst_wr, dst_wc))

    # ------------------------------------------------------------------
    # Strategy 4: Exit Runner
    # ------------------------------------------------------------------

    def exit_runner(self, malice):
        """
        When the agent is close to the exit, MOVE the exit to somewhere far
        away and wall off the old exit's approach corridor.

        Why this hurts the agent: the agent has built up high Q-values
        guiding it toward the old exit location. Suddenly that whole
        region is a dead end, and the actual exit is somewhere the agent
        hasn't mapped yet. All that exit-approach knowledge is worthless.
        """
        agent_pos = self.agent.pos
        er, ec = self.maze.exit

        # Manhattan distance from agent to exit
        dist = abs(agent_pos[0] - er) + abs(agent_pos[1] - ec)

        # Trigger threshold shrinks as malice rises (fires from farther away)
        threshold = max(2, int((1.0 - malice) * 8) + 2)
        if dist > threshold:
            return

        # Find a new exit location far from the agent
        candidates = [
            (r, c)
            for r in range(self.maze.rows)
            for c in range(self.maze.cols)
            if (abs(agent_pos[0] - r) + abs(agent_pos[1] - c)) > self.maze.rows // 2
            and (r, c) != agent_pos
        ]

        if not candidates:
            return

        new_exit = random.choice(candidates)

        # Flash both the old and new exit positions
        self.flash_cells.append(self.maze.room_to_grid(*self.maze.exit))
        self.flash_cells.append(self.maze.room_to_grid(*new_exit))

        old_exit = self.maze.exit
        self.maze.exit = new_exit

        # Wall off the old exit to make the approach a dead end
        old_r, old_c = old_exit
        dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        random.shuffle(dirs)
        max_changes = max(1, int(malice * 3))
        changes = 0

        for dr, dc in dirs:
            if changes >= max_changes:
                break
            if self.maze.try_add_wall(old_r, old_c, dr, dc, self.agent.pos):
                wr, wc = 2 * old_r + 1 + dr, 2 * old_c + 1 + dc
                self.flash_cells.append((wr, wc))
                changes += 1
