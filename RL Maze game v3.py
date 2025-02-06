import numpy as np
import random
import pygame

# Maze setup
MAZE_SIZE = 5  # 5x5 grid
REWARD_GOAL = 10
REWARD_STEP = -1
REWARD_WALL = -5
REWARD_ENEMY = -10

# Define maze (0 = empty, 1 = wall, 2 = goal)
maze = np.zeros((MAZE_SIZE, MAZE_SIZE))
maze[4, 4] = 2  # Goal position

# Add obstacles (walls)
wall_positions = [(1, 1), (1, 2), (2, 3), (3, 1)]
for pos in wall_positions:
    maze[pos] = 1  # Mark as a wall

# Q-learning setup
alpha = 0.1  # Learning rate
gamma = 0.9  # Discount factor
epsilon = 1.0  # Exploration-exploitation tradeoff (set high for full exploration)
epsilon_decay = 0.995  # Decay rate (slow decay)
min_epsilon = 0.1  # Minimum epsilon (do not let it go too low)
ACTIONS = [(0, -1), (0, 1), (-1, 0), (1, 0)]  # Up, Down, Left, Right

# Initialize Q-table
Q_table = np.zeros((MAZE_SIZE, MAZE_SIZE, 4))

# Enemy setup (starts at (2,2))
enemy_pos = (2, 2)
enemy_moves = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # Moves in a square
enemy_step = 0  # Track current movement step

# Function to take a step in the environment
def take_step(state, action):
    global enemy_pos, enemy_step

    # Move enemy in a fixed pattern
    enemy_step = (enemy_step + 1) % len(enemy_moves)
    enemy_x = max(0, min(enemy_pos[0] + enemy_moves[enemy_step][0], MAZE_SIZE - 1))
    enemy_y = max(0, min(enemy_pos[1] + enemy_moves[enemy_step][1], MAZE_SIZE - 1))
    enemy_pos = (enemy_x, enemy_y)

    # Move agent
    new_x = max(0, min(state[0] + ACTIONS[action][0], MAZE_SIZE - 1))
    new_y = max(0, min(state[1] + ACTIONS[action][1], MAZE_SIZE - 1))

    # If new position is a wall, stay in the same place
    if maze[new_x, new_y] == 1:
        return state, REWARD_WALL

    new_state = (new_x, new_y)

    # If the agent collides with the enemy, give a penalty
    if new_state == enemy_pos:
        return new_state, REWARD_ENEMY

    reward = REWARD_GOAL if maze[new_x, new_y] == 2 else REWARD_STEP
    return new_state, reward

# Train the agent using Q-learning
num_episodes = 2000  # More episodes for more exploration
for episode in range(num_episodes):
    state = (0, 0)  # Start position
    done = False

    while not done:
        # Exploration with epsilon-greedy policy
        action = np.random.randint(4) if np.random.rand() < epsilon else np.argmax(Q_table[state[0], state[1]])
        new_state, reward = take_step(state, action)

        # Q-learning update rule
        best_future_q = np.max(Q_table[new_state[0], new_state[1]])
        Q_table[state[0], state[1], action] += alpha * (reward + gamma * best_future_q - Q_table[state[0], state[1], action])

        state = new_state
        if maze[state[0], state[1]] == 2:  # Goal reached
            done = True

    # Decay epsilon slowly to allow more exploration in the beginning
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
RED = (255, 0, 0)  # Enemy
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

# Draw enemy
def draw_enemy():
    pygame.draw.circle(screen, RED, (enemy_pos[0] * CELL_SIZE + CELL_SIZE // 2, enemy_pos[1] * CELL_SIZE + CELL_SIZE // 2), CELL_SIZE // 3)

# Run the game with AI
def run_game():
    agent_pos = (0, 0)
    running = True

    while running:
        screen.fill(WHITE)
        draw_maze()
        draw_enemy()
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
