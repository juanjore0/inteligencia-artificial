import pygame
import numpy as np
import sys
import random
from collections import defaultdict

pygame.init()

SCREEN_WIDTH = 600
SCREEN_HEIGHT = 460
CELL_SIZE = 40
FPS = 60

MAZE_OFFSET_X = 60
MAZE_OFFSET_Y = 50

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GRAY = (100, 100, 100)
GREEN = (0, 200, 0)
YELLOW = (255, 215, 0)
BROWN = (139, 69, 19)
BLUE = (100, 100, 255)
RED = (255, 100, 100)

screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Ventana")
clock = pygame.time.Clock()

maze = np.array([
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
    [1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 0, 1],
    [1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1],
    [1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
])

MAZE_ROWS = maze.shape[0]
MAZE_COLS = maze.shape[1]

class Mouse:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.speed = CELL_SIZE
        
    def move(self, dx, dy):
        new_x = self.x + dx * self.speed
        new_y = self.y + dy * self.speed
        
        maze_col = new_x // CELL_SIZE
        maze_row = new_y // CELL_SIZE
        
        if 0 <= maze_row < MAZE_ROWS and 0 <= maze_col < MAZE_COLS:
            if maze[maze_row][maze_col] == 0:
                self.x = new_x
                self.y = new_y
    
    def draw(self):
        x = self.x + MAZE_OFFSET_X
        y = self.y + MAZE_OFFSET_Y
        pygame.draw.circle(screen, GRAY, (x + CELL_SIZE // 2, y + CELL_SIZE // 2), CELL_SIZE // 3)
        
        ear_offset = CELL_SIZE // 4
        pygame.draw.circle(screen, GRAY, (x + CELL_SIZE // 4, y + CELL_SIZE // 4), CELL_SIZE // 6)
        pygame.draw.circle(screen, GRAY, (x + 3 * CELL_SIZE // 4, y + CELL_SIZE // 4), CELL_SIZE // 6)
        
        pygame.draw.circle(screen, BLACK, (x + CELL_SIZE // 3, y + CELL_SIZE // 2), CELL_SIZE // 10)
        pygame.draw.circle(screen, BLACK, (x + 2 * CELL_SIZE // 3, y + CELL_SIZE // 2), CELL_SIZE // 10)
        
        pygame.draw.line(screen, BLACK, 
                        (x + CELL_SIZE // 2, y + 2 * CELL_SIZE // 3),
                        (x + CELL_SIZE // 2 + CELL_SIZE // 3, y + 3 * CELL_SIZE // 4), 2)

class Cheese:
    def __init__(self, x, y):
        self.x = x
        self.y = y
    
    def draw(self):
        x = self.x + MAZE_OFFSET_X
        y = self.y + MAZE_OFFSET_Y
        pygame.draw.rect(screen, YELLOW, (x + 5, y + 5, CELL_SIZE - 10, CELL_SIZE - 10))
        pygame.draw.circle(screen, BROWN, (x + CELL_SIZE // 3, y + CELL_SIZE // 3), 3)
        pygame.draw.circle(screen, BROWN, (x + 2 * CELL_SIZE // 3, y + CELL_SIZE // 3), 3)
        pygame.draw.circle(screen, BROWN, (x + CELL_SIZE // 2, y + 2 * CELL_SIZE // 3), 3)

class QLearning:
    def __init__(self, learning_rate=0.1, discount_factor=0.95, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01):
        self.q_table = defaultdict(lambda: defaultdict(float))
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.actions = [(0, -1), (0, 1), (-1, 0), (1, 0)]
    
    def get_state(self, mouse):
        return (mouse.x // CELL_SIZE, mouse.y // CELL_SIZE)
    
    def choose_action(self, state, training=True):
        if training and random.random() < self.epsilon:
            return random.choice(self.actions)
        else:
            q_values = [self.q_table[state][action] for action in self.actions]
            max_q = max(q_values)
            best_actions = [self.actions[i] for i, q in enumerate(q_values) if q == max_q]
            return random.choice(best_actions)
    
    def update_q_value(self, state, action, reward, next_state):
        current_q = self.q_table[state][action]
        max_next_q = max([self.q_table[next_state][a] for a in self.actions])
        new_q = current_q + self.learning_rate * (reward + self.discount_factor * max_next_q - current_q)
        self.q_table[state][action] = new_q
    
    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

def draw_maze():
    for row in range(MAZE_ROWS):
        for col in range(MAZE_COLS):
            x = col * CELL_SIZE + MAZE_OFFSET_X
            y = row * CELL_SIZE + MAZE_OFFSET_Y
            if maze[row][col] == 1:
                pygame.draw.rect(screen, BLACK, (x, y, CELL_SIZE, CELL_SIZE))
            else:
                pygame.draw.rect(screen, WHITE, (x, y, CELL_SIZE, CELL_SIZE))
                pygame.draw.rect(screen, GRAY, (x, y, CELL_SIZE, CELL_SIZE), 1)

def check_collision(mouse, cheese):
    mouse_col = mouse.x // CELL_SIZE
    mouse_row = mouse.y // CELL_SIZE
    cheese_col = cheese.x // CELL_SIZE
    cheese_row = cheese.y // CELL_SIZE
    
    return mouse_col == cheese_col and mouse_row == cheese_row

def show_victory_message():
    font = pygame.font.Font(None, 74)
    text = font.render("¡Victoria!", True, GREEN)
    text_rect = text.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2))
    screen.blit(text, text_rect)
    
    font_small = pygame.font.Font(None, 36)
    text_small = font_small.render("Presiona ESC para salir", True, BLACK)
    text_rect_small = text_small.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 + 60))
    screen.blit(text_small, text_rect_small)

def show_training_info(episode, total_episodes, epsilon, reward):
    font = pygame.font.Font(None, 24)
    info_text = f"Entrenamiento: Episodio {episode}/{total_episodes} | Epsilon: {epsilon:.3f} | Recompensa: {reward:.1f}"
    text_surface = font.render(info_text, True, BLACK)
    pygame.draw.rect(screen, WHITE, (0, 0, SCREEN_WIDTH, 30))
    screen.blit(text_surface, (10, 5))

def train_qlearning(episodes=200, max_steps=50, visualize=True):
    q_agent = QLearning()
    cheese_pos = (10, 7)
    
    for episode in range(episodes):
        mouse = Mouse(CELL_SIZE, CELL_SIZE)
        cheese = Cheese(cheese_pos[0] * CELL_SIZE, cheese_pos[1] * CELL_SIZE)
        
        total_reward = 0
        steps = 0
        done = False
        
        while not done and steps < max_steps:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
            
            state = q_agent.get_state(mouse)
            action = q_agent.choose_action(state, training=True)
            
            old_x, old_y = mouse.x, mouse.y
            mouse.move(action[0], action[1])
            next_state = q_agent.get_state(mouse)
            
            reward = -1
            if check_collision(mouse, cheese):
                reward = 100
                done = True
            elif (old_x, old_y) == (mouse.x, mouse.y):
                reward = -10
            
            q_agent.update_q_value(state, action, reward, next_state)
            total_reward += reward
            steps += 1
            
            if visualize and episode % 10 == 0:
                screen.fill(WHITE)
                draw_maze()
                cheese.draw()
                mouse.draw()
                show_training_info(episode + 1, episodes, q_agent.epsilon, total_reward)
                pygame.display.flip()
                clock.tick(30)
        
        q_agent.decay_epsilon()
        
        if episode % 50 == 0:
            print(f"Episodio {episode}/{episodes} - Recompensa: {total_reward:.1f} - Epsilon: {q_agent.epsilon:.3f}")
    
    return q_agent

def run_trained_mouse(q_agent):
    mouse = Mouse(CELL_SIZE, CELL_SIZE)
    cheese = Cheese(10 * CELL_SIZE, 7 * CELL_SIZE)
    
    victory = False
    running = True
    steps = 0
    max_steps = 100
    
    while running and not victory and steps < max_steps:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
        
        state = q_agent.get_state(mouse)
        action = q_agent.choose_action(state, training=False)
        mouse.move(action[0], action[1])
        
        if check_collision(mouse, cheese):
            victory = True
        
        screen.fill(WHITE)
        draw_maze()
        cheese.draw()
        mouse.draw()
        
        font = pygame.font.Font(None, 24)
        text = font.render(f"Ratón Entrenado - Paso {steps}", True, BLACK)
        pygame.draw.rect(screen, WHITE, (0, 0, SCREEN_WIDTH, 30))
        screen.blit(text, (10, 5))
        
        if victory:
            show_victory_message()
        
        pygame.display.flip()
        clock.tick(10)
        steps += 1
    
    if victory:
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
            
            screen.fill(WHITE)
            draw_maze()
            cheese.draw()
            mouse.draw()
            show_victory_message()
            pygame.display.flip()
            clock.tick(FPS)

def main():
    print("Iniciando entrenamiento del ratón con Q-Learning...")
    q_agent = train_qlearning(episodes=200, max_steps=50, visualize=True)
    print("Entrenamiento completado!")
    print("\nAhora el ratón entrenado buscará el queso...")
    
    run_trained_mouse(q_agent)
    
    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()
