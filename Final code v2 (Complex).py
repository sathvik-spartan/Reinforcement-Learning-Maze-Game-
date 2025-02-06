import numpy as np
import random
import pygame

# Maze setup
MAZE_SIZE = 10  # Larger grid size (10x10)
REWARD_GOAL = 10
REWARD_STEP = -1
REWARD_WALL = -5

# Define maze (0 = empty, 1 = wall, 2 = goal)
maze = np.zeros((MAZE_SIZE, MAZE_SIZE))
maze[MAZE_SIZE - 1, MAZE_SIZE - 1] = 2  # Goal position at bottom-right corner

# Add random obstacles (walls)
random.seed(42)
wall_positions = set()

# Make sure that start and end positions are not blocked
wall_positions.add((0, 0))
wall_positions.add((MAZE_SIZE - 1, MAZE_SIZE - 1))

# Randomly place walls (about 30% of the grid)
for _ in range(MAZE_SIZE * MAZE_SIZE // 3):
    x, y = random.randint(0, MAZE_SIZE - 1), random.randint(0, MAZE_SIZE - 1)
    if (x, y) != (0, 0) and (x, y) != (MAZE_SIZE - 1, MAZE_SIZE - 1):
        wall_positions.add((x, y))

# Mark wall positions in the maze
for pos in wall_positions:
    maze[pos] = 1  # Set the maze grid to walls where needed

# Q-learning setup
alpha = 0.1  # Learning rate
gamma = 0.9  # Discount factor
epsilon = 1.0  # Exploration rate
epsilon_decay = 0.995  # Epsilon decay rate
min_epsilon = 0.1  # Minimum epsilon value
ACTIONS = [(0, -1), (0, 1), (-1, 0), (1, 0)]  # Up, Down, Left, Right

# Initialize Q-table
Q_table = np.zeros((MAZE_SIZE, MAZE_SIZE, 4))

# Function to take a step in the environment
def take_step(state, action):
    # Move agent
    new_x = max(0, min(state[0] + ACTIONS[action][0], MAZE_SIZE - 1))
    new_y = max(0, min(state[1] + ACTIONS[action][1], MAZE_SIZE - 1))

    # If new position is a wall, stay in the same place
    if maze[new_x, new_y] == 1:
        return state, REWARD_WALL

    new_state = (new_x, new_y)

    reward = REWARD_GOAL if maze[new_x, new_y] == 2 else REWARD_STEP
    return new_state, reward

# Train the agent using Q-learning
num_episodes = 3000  # Increase episodes for a better trained agent
for episode in range(num_episodes):
    state = (0, 0)  # Start position
    done = False

    while not done:
        action = np.random.randint(4) if np.random.rand() < epsilon else np.argmax(Q_table[state[0], state[1]])
        new_state, reward = take_step(state, action)

        # Q-learning update rule
        best_future_q = np.max(Q_table[new_state[0], new_state[1]])
        Q_table[state[0], state[1], action] += alpha * (reward + gamma * best_future_q - Q_table[state[0], state[1], action])

        state = new_state
        if maze[state[0], state[1]] == 2:  # Goal reached
            done = True

    # Decay epsilon
    epsilon = max(min_epsilon, epsilon * epsilon_decay)

# Pygame setup
pygame.init()
WIDTH, HEIGHT = 500, 500
CELL_SIZE = WIDTH // MAZE_SIZE
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Q-learning Maze Game")

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREEN = (0, 255, 0)  # Goal
BLUE = (0, 0, 255)  # Agent
GRAY = (100, 100, 100)  # Obstacles

# Draw the maze
def draw_maze():
    screen.fill(WHITE)
    
    for x in range(MAZE_SIZE):
        for y in range(MAZE_SIZE):
            rect = pygame.Rect(x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE)
            if maze[x, y] == 1:
                pygame.draw.rect(screen, GRAY, rect)  # Walls
            elif maze[x, y] == 2:
                pygame.draw.rect(screen, GREEN, rect)  # Goal
            pygame.draw.rect(screen, BLACK, rect, 2)  # Grid outline

# Draw agent
def draw_agent(agent_pos):
    pygame.draw.circle(screen, BLUE, (agent_pos[0] * CELL_SIZE + CELL_SIZE // 2, agent_pos[1] * CELL_SIZE + CELL_SIZE // 2), CELL_SIZE // 3)

# Run the game with AI
def run_game():
    agent_pos = (0, 0)
    running = True

    while running:
        screen.fill(WHITE)
        draw_maze()
        draw_agent(agent_pos)
        pygame.display.flip()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Use Q-table to decide movement
        action = np.argmax(Q_table[agent_pos[0], agent_pos[1]])
        agent_pos, _ = take_step(agent_pos, action)

        pygame.time.delay(300)  # Slow down movement

    pygame.quit()

run_game()
