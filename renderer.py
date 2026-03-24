"""
renderer.py — Pygame visualization

Draws the maze grid, the agent, the exit, and a HUD showing live stats.
Also handles the "flash" effect: when the maze mutates, changed wall
cells briefly glow yellow before fading back to normal.

Coordinate systems:
  - maze room coords: (r, c) where r is row, c is col
  - grid coords: (2r+1, 2c+1) for rooms; wall cells in between
  - screen pixels: gc * CELL_SIZE, gr * CELL_SIZE (top-left origin)
"""

import pygame

# -----------------------------------------------------------------------
# Visual constants
# -----------------------------------------------------------------------

CELL_SIZE = 22          # Pixels per grid unit (room = CELL_SIZE×CELL_SIZE)
HUD_HEIGHT = 55         # Pixels reserved at the bottom for stats

# Color palette — dark fantasy labyrinth theme
DARK_BG       = (12, 12, 20)
WALL_COLOR    = (55, 55, 75)
PATH_COLOR    = (25, 25, 38)
AGENT_CORE    = (0, 210, 255)     # bright cyan dot
AGENT_GLOW    = (0, 80, 160)      # darker blue halo
EXIT_COLOR    = (0, 230, 90)      # green exit cell
FLASH_COLOR   = (255, 210, 50)    # yellow flash for mutated cells
HUD_BG        = (8, 8, 16)
HUD_TEXT      = (200, 200, 220)
HUD_SUBTEXT   = (110, 110, 135)

FLASH_DURATION = 45     # frames a mutated cell stays highlighted


class Renderer:
    def __init__(self, maze):
        # maze is stored by reference so mutations are visible immediately
        self.maze = maze

        grid_w = (2 * maze.cols + 1) * CELL_SIZE
        grid_h = (2 * maze.rows + 1) * CELL_SIZE
        self.width = grid_w
        self.height = grid_h + HUD_HEIGHT

        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Adaptive Labyrinth")

        self.font_main = pygame.font.SysFont("monospace", 15)
        self.font_sub  = pygame.font.SysFont("monospace", 13)

        # Maps grid coord (gr, gc) → remaining flash frames
        # (like a Java HashMap<int[], Integer>)
        self.flash_timers = {}

    # ------------------------------------------------------------------
    # Flash management
    # ------------------------------------------------------------------

    def add_flashes(self, grid_coords):
        """Register a list of grid cells to flash (called after adversary acts)."""
        for coord in grid_coords:
            self.flash_timers[coord] = FLASH_DURATION

    def _tick_flashes(self):
        """Decrement flash timers and remove expired ones each frame."""
        expired = [k for k, v in self.flash_timers.items() if v <= 0]
        for k in expired:
            del self.flash_timers[k]
        for k in self.flash_timers:
            self.flash_timers[k] -= 1

    # ------------------------------------------------------------------
    # Main draw call (called once per frame from main.py)
    # ------------------------------------------------------------------

    def draw(self, agent, episode, steps, malice, paused):
        self.screen.fill(DARK_BG)

        self._draw_maze()
        self._draw_exit()
        self._draw_agent(agent)
        self._draw_hud(agent, episode, steps, malice, paused)

        self._tick_flashes()
        pygame.display.flip()

    # ------------------------------------------------------------------
    # Internal drawing helpers
    # ------------------------------------------------------------------

    def _draw_maze(self):
        """Draw every cell in the grid — walls dark, passages darker."""
        for gr in range(2 * self.maze.rows + 1):
            for gc in range(2 * self.maze.cols + 1):
                x = gc * CELL_SIZE
                y = gr * CELL_SIZE
                rect = pygame.Rect(x, y, CELL_SIZE, CELL_SIZE)

                if (gr, gc) in self.flash_timers:
                    # Blend flash color toward normal based on time remaining
                    t = self.flash_timers[(gr, gc)] / FLASH_DURATION  # 1.0→0.0
                    base = WALL_COLOR if self.maze.grid[gr][gc] == 1 else PATH_COLOR
                    color = tuple(
                        int(FLASH_COLOR[i] * t + base[i] * (1 - t))
                        for i in range(3)
                    )
                elif self.maze.grid[gr][gc] == 1:
                    color = WALL_COLOR
                else:
                    color = PATH_COLOR

                pygame.draw.rect(self.screen, color, rect)

    def _draw_exit(self):
        """Highlight the exit cell in green."""
        er, ec = self.maze.exit
        x = (2 * ec + 1) * CELL_SIZE
        y = (2 * er + 1) * CELL_SIZE
        pygame.draw.rect(self.screen, EXIT_COLOR,
                         pygame.Rect(x, y, CELL_SIZE, CELL_SIZE))

    def _draw_agent(self, agent):
        """Draw the agent as a glowing cyan dot at its current room cell."""
        ar, ac = agent.pos
        # Center of the room cell in screen pixels
        cx = (2 * ac + 1) * CELL_SIZE + CELL_SIZE // 2
        cy = (2 * ar + 1) * CELL_SIZE + CELL_SIZE // 2

        # Outer glow (larger, darker circle)
        pygame.draw.circle(self.screen, AGENT_GLOW, (cx, cy), CELL_SIZE - 2)
        # Inner core (smaller, bright circle)
        pygame.draw.circle(self.screen, AGENT_CORE, (cx, cy), CELL_SIZE // 2)

    def _draw_hud(self, agent, episode, steps, malice, paused):
        """Draw the stats bar at the bottom of the screen."""
        hud_top = (2 * self.maze.rows + 1) * CELL_SIZE
        pygame.draw.rect(self.screen, HUD_BG,
                         pygame.Rect(0, hud_top, self.width, HUD_HEIGHT))

        # Draw a thin separator line
        pygame.draw.line(self.screen, WALL_COLOR,
                         (0, hud_top), (self.width, hud_top), 1)

        pause_tag = "  [PAUSED]" if paused else ""
        main_line = (
            f"Episode: {episode}   Steps: {steps}   "
            f"Malice: {malice:.1f}   ε: {agent.epsilon:.3f}"
            f"{pause_tag}"
        )
        ctrl_line = "SPACE: pause    ↑↓: speed    ←→: malice    R: reset AI"

        self.screen.blit(
            self.font_main.render(main_line, True, HUD_TEXT),
            (10, hud_top + 8),
        )
        self.screen.blit(
            self.font_sub.render(ctrl_line, True, HUD_SUBTEXT),
            (10, hud_top + 30),
        )
