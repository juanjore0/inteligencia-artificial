import pygame
import numpy as np
import random
import time

# Inicializar pygame
pygame.init()

# Configuración de pantalla
CELL_SIZE = 80
ROWS, COLS = 5, 5
WIDTH, HEIGHT = COLS * CELL_SIZE, ROWS * CELL_SIZE
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Laberinto con Q-Learning")

# Colores
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)

# Laberinto (SI tiene solución)
maze = np.array([
    [0, 0, 0, 1, 2],
    [1, 1, 0, 1, 0],
    [0, 1, 0, 0, 0],
    [0, 1, 1, 1, 0],
    [0, 0, 0, 0, 0]
])

# Acciones: arriba, abajo, izquierda, derecha
actions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

# Tabla Q
Q = np.zeros((ROWS, COLS, len(actions)))

# Parámetros
alpha = 0.1
gamma = 0.9
epsilon = 0.8
episodes = 500
max_steps = 10

# Control
attempts = 0
goal_reached = False

# Función para dibujar
def draw(agent_pos):
    screen.fill(WHITE)

    for r in range(ROWS):
        for c in range(COLS):
            rect = pygame.Rect(c * CELL_SIZE, r * CELL_SIZE, CELL_SIZE, CELL_SIZE)

            if maze[r][c] == 1:
                pygame.draw.rect(screen, BLACK, rect)
            elif maze[r][c] == 2:
                pygame.draw.rect(screen, GREEN, rect)
            else:
                pygame.draw.rect(screen, WHITE, rect)

            pygame.draw.rect(screen, BLACK, rect, 1)

    # Dibujar agente
    r, c = agent_pos
    pygame.draw.circle(screen, RED,
                       (c * CELL_SIZE + CELL_SIZE // 2,
                        r * CELL_SIZE + CELL_SIZE // 2),
                       CELL_SIZE // 4)

    pygame.display.flip()

# Validar movimiento
def is_valid(state):
    r, c = state
    return 0 <= r < ROWS and 0 <= c < COLS and maze[r][c] != 1

# Recompensa
def get_reward(state):
    if maze[state] == 2:
        return 100
    return -1

# Mostrar texto final
def show_text(text):
    font = pygame.font.SysFont(None, 50)
    img = font.render(text, True, BLUE)
    rect = img.get_rect(center=(WIDTH // 2, HEIGHT // 2))
    screen.blit(img, rect)
    pygame.display.flip()

# 🔁 ENTRENAMIENTO
for episode in range(episodes):
    state = (0, 0)
    attempts += 1

    for step in range(max_steps):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()

        r, c = state

        # Exploración vs explotación
        if random.uniform(0, 1) < epsilon:
            action_idx = random.randint(0, 3)
        else:
            action_idx = np.argmax(Q[r, c])

        dr, dc = actions[action_idx]
        next_state = (r + dr, c + dc)

        if not is_valid(next_state):
            next_state = state

        reward = get_reward(next_state)

        nr, nc = next_state

        # Actualización Q-learning
        Q[r, c, action_idx] += alpha * (
            reward + gamma * np.max(Q[nr, nc]) - Q[r, c, action_idx]
        )

        state = next_state

        draw(state)
        time.sleep(0.05)

        # Si llega a la meta
        if maze[state] == 2:
            goal_reached = True
            break

    if goal_reached:
        break

# 🎯 RESULTADO FINAL
if goal_reached:
    print(f"Meta alcanzada en {attempts} intentos")

    screen.fill(WHITE)
    show_text(f"Intentos: {attempts}")

# Mantener ventana abierta
while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            exit()