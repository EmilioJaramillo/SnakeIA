import torch
import random
import numpy as np
import os
from collections import deque
from juego import SnakeGameAI, Direction, Point
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib
import matplotlib.pyplot as plt

# Evitar crasheos de gráficos y usar backend sin ventana para plots
matplotlib.use('Agg') 

# Usar GPU si existe, sino CPU
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001

class Linear_QNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x

    def save(self, file_name='model.pth'):
        torch.save(self.state_dict(), file_name)

class QTrainer:
    def __init__(self, model, lr, gamma):
        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()

    def train_step(self, state, action, reward, next_state, done):
        state = torch.tensor(np.array(state), dtype=torch.float, device=DEVICE)
        next_state = torch.tensor(np.array(next_state), dtype=torch.float, device=DEVICE)
        action = torch.tensor(np.array(action), dtype=torch.long, device=DEVICE)
        reward = torch.tensor(np.array(reward), dtype=torch.float, device=DEVICE)

        if len(state.shape) == 1:
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done, )

        pred = self.model(state)
        target = pred.clone()
        for idx in range(len(done)):
            Q_new = reward[idx]
            if not done[idx]:
                Q_new = reward[idx] + self.gamma * torch.max(self.model(next_state[idx]))
            target[idx][torch.argmax(action[idx]).item()] = Q_new
    
        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()
        self.optimizer.step()

class Agent:
    def __init__(self):
        self.n_games = 0
        self.epsilon = 0 
        self.gamma = 0.9 
        self.memory = deque(maxlen=MAX_MEMORY) 
        
        self.model = Linear_QNet(11, 256, 3).to(DEVICE)
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)

        if os.path.exists('model.pth'):
            try:
                self.model.load_state_dict(torch.load('model.pth', map_location=DEVICE))
                self.model.eval()
                self.n_games = 150 # Empezar desde juego 150 si ya hay modelo 
                print(">>> ¡Cerebro encontrado! Modo Experto ACTIVADO.")
            except Exception as e:
                print(f">>> Error cargando: {e}. Reiniciando.")
        else:
            print(">>> Modo Entrenamiento activado.")

    def get_state(self, game):
        head = game.snake[0]
        point_l = Point(head.x - 20, head.y)
        point_r = Point(head.x + 20, head.y)
        point_u = Point(head.x, head.y - 20)
        point_d = Point(head.x, head.y + 20)
        
        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        state = [
            # Peligro
            (dir_r and game.is_collision(point_r)) or 
            (dir_l and game.is_collision(point_l)) or 
            (dir_u and game.is_collision(point_u)) or 
            (dir_d and game.is_collision(point_d)),

            (dir_u and game.is_collision(point_r)) or 
            (dir_d and game.is_collision(point_l)) or 
            (dir_l and game.is_collision(point_u)) or 
            (dir_r and game.is_collision(point_d)),

            (dir_d and game.is_collision(point_r)) or 
            (dir_u and game.is_collision(point_l)) or 
            (dir_r and game.is_collision(point_u)) or 
            (dir_l and game.is_collision(point_d)),
            
            # Dirección
            dir_l, dir_r, dir_u, dir_d,
            
            # Comida
            game.food.x < game.head.x, 
            game.food.x > game.head.x, 
            game.food.y < game.head.y, 
            game.food.y > game.head.y 
        ]
        return np.array(state, dtype=int)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)
        else:
            mini_sample = self.memory
        
        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        # --- ESTRATEGIA PARA 300 JUEGOS ---
        # Epsilon empieza alto y baja hasta 0 en el juego 150.
        self.epsilon = 150 - self.n_games
        if self.epsilon < 0: self.epsilon = 0

        final_move = [0,0,0]
        
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float, device=DEVICE)
            with torch.no_grad():
                prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1
            
        return final_move

def save_plot(scores, mean_scores):
    try:
        plt.figure(1)
        plt.clf()
        plt.title('Entrenando IA - Progreso')
        plt.xlabel('Juegos')
        plt.ylabel('Puntaje')
        plt.plot(scores, label='Score')
        plt.plot(mean_scores, label='Promedio')
        plt.legend()
        plt.savefig('grafico_progreso.png') 
        # No usamos plt.close() agresivamente para evitar errores de hilo
    except:
        pass

def train():
    print(">>> INICIANDO")
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    agent = Agent()
    game = SnakeGameAI(control_mode='ai')
    
    while True:
        state_old = agent.get_state(game)
        final_move = agent.get_action(state_old)
        
        # --- AQUÍ PASAMOS EL NÚMERO DE JUEGO PARA EL DASHBOARD ---
        reward, done, score = game.play_step(final_move, agent.n_games)
        
        state_new = agent.get_state(game)
        
        agent.train_short_memory(state_old, final_move, reward, state_new, done)
        agent.remember(state_old, final_move, reward, state_new, done)

        if done:
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()
            
            if score > record:
                record = score
                agent.model.save()
                print(f"! NUEVO RECORD: {record} !!!")

            print(f'Juego: {agent.n_games} | Puntuación: {score} | Record: {record}')

            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)
            save_plot(plot_scores, plot_mean_scores)

if __name__ == '__main__':
    train()