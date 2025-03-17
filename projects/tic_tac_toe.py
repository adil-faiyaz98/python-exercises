import pygame as pg
import sys
import time
from pygame.locals import *

# Initialize game variables
current_player = 'x'
current_winner = None
is_draw = False

WIDTH = 400
HEIGHT = 400  # Game area (100px added for status bar)
BACKGROUND = (255, 255, 255)
LINE_COLOR = (0, 0, 0)
RED = (250, 0, 0)

grid = [[None] * 3 for _ in range(3)]

pg.init()
FPS = 30
clock = pg.time.Clock()

screen = pg.display.set_mode((WIDTH, HEIGHT + 100), 0, 32)  # 400 x 500 display
pg.display.set_caption("Tic Tac Toe")

size = 80  # Size of X / O marks


def game_initiating_window():
    """Initializes the game window and draws the board."""
    screen.fill(BACKGROUND)
    for i in range(1, 3):
        pg.draw.line(screen, LINE_COLOR, (WIDTH / 3 * i, 0), (WIDTH / 3 * i, HEIGHT), 7)
        pg.draw.line(screen, LINE_COLOR, (0, HEIGHT / 3 * i), (WIDTH, HEIGHT / 3 * i), 7)
    draw_status()


def draw_status():
    """Displays the current game state at the bottom."""
    global is_draw
    message = f"{current_player.upper()}'s Turn"

    if current_winner:
        message = f"{current_winner.upper()} Wins!"
    elif is_draw:
        message = "Game Draw!"

    font = pg.font.Font(None, 40)
    text = font.render(message, True, (255, 255, 255))

    screen.fill((0, 0, 0), (0, HEIGHT, WIDTH, 100))  # Status bar background
    text_rect = text.get_rect(center=(WIDTH / 2, HEIGHT + 50))  # Proper positioning
    screen.blit(text, text_rect)
    pg.display.update()


def check_win():
    """Checks for a win or draw and draws a red line for winning moves."""
    global current_winner, is_draw

    # Check rows and columns
    for i in range(3):
        if grid[i][0] == grid[i][1] == grid[i][2] and grid[i][0]:
            current_winner = grid[i][0]
            pg.draw.line(screen, RED, (10, HEIGHT / 3 * (i + 0.5)), (WIDTH - 10, HEIGHT / 3 * (i + 0.5)), 7)
            break
        if grid[0][i] == grid[1][i] == grid[2][i] and grid[0][i]:
            current_winner = grid[0][i]
            pg.draw.line(screen, RED, (WIDTH / 3 * (i + 0.5), 10), (WIDTH / 3 * (i + 0.5), HEIGHT - 10), 7)
            break

    # Check diagonals
    if grid[0][0] == grid[1][1] == grid[2][2] and grid[0][0]:
        current_winner = grid[0][0]
        pg.draw.line(screen, RED, (20, 20), (WIDTH - 20, HEIGHT - 20), 7)

    if grid[0][2] == grid[1][1] == grid[2][0] and grid[0][2]:
        current_winner = grid[0][2]
        pg.draw.line(screen, RED, (WIDTH - 20, 20), (20, HEIGHT - 20), 7)

    # Check for draw
    if all(all(row) for row in grid) and not current_winner:
        is_draw = True

    draw_status()


def drawXO(row, col):
    """Draws X or O at the clicked position."""
    global current_player
    pos_x, pos_y = (WIDTH / 3) * (col - 1) + 50, (HEIGHT / 3) * (row - 1) + 50
    grid[row - 1][col - 1] = current_player

    if current_player == 'x':
        pg.draw.line(screen, LINE_COLOR, (pos_x - 25, pos_y - 25), (pos_x + 25, pos_y + 25), 5)
        pg.draw.line(screen, LINE_COLOR, (pos_x - 25, pos_y + 25), (pos_x + 25, pos_y - 25), 5)
        current_player = 'o'
    else:
        pg.draw.circle(screen, LINE_COLOR, (int(pos_x), int(pos_y)), 30, 5)
        current_player = 'x'

    pg.display.update()


def user_click():
    """Handles user clicks and places X/O in the correct spot."""
    x, y = pg.mouse.get_pos()
    col = 1 if x < WIDTH / 3 else 2 if x < WIDTH / 3 * 2 else 3
    row = 1 if y < HEIGHT / 3 else 2 if y < HEIGHT / 3 * 2 else 3

    if grid[row - 1][col - 1] is None:
        drawXO(row, col)
        check_win()


def reset_game():
    """Resets the game and logs the wait time."""
    global grid, current_winner, current_player, is_draw

    # Start the timer
    start_time = time.time()

    # Wait 2 seconds before restarting
    pg.time.delay(2000)

    # Calculate elapsed time
    elapsed_time = time.time() - start_time
    print(f"Game waited for {elapsed_time:.2f} seconds before restarting.")

    # Reset game state
    current_player = 'x'
    current_winner = None
    is_draw = False
    grid = [[None] * 3 for _ in range(3)]

    # Clear screen and redraw game board
    screen.fill(BACKGROUND)
    game_initiating_window()


# Initialize the game
game_initiating_window()

# Main game loop
while True:
    for event in pg.event.get():
        if event.type == QUIT:
            pg.quit()
            sys.exit()
        elif event.type == MOUSEBUTTONDOWN:
            user_click()
            if current_winner or is_draw:
                reset_game()
    pg.display.update()
    clock.tick(FPS)
