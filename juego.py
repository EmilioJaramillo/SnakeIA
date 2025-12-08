import pygame
import random
import os
import time
from enum import Enum
from collections import namedtuple
import numpy as np

pygame.init()
pygame.mixer.init()

# --- FUENTES ---
font_main = pygame.font.SysFont('arial', 20)
font_small = pygame.font.SysFont('arial', 16)
font_title = pygame.font.SysFont('arial', 50, bold=True)
font_subtitle = pygame.font.SysFont('arial', 25)
font_icon = pygame.font.SysFont('arial', 15, bold=True)
font_crash = pygame.font.SysFont('arial', 40, bold=True)

# --- COLORES ---
COLOR_BG = (15, 15, 25)      
COLOR_GRID = (40, 40, 60)    
COLOR_SNAKE_BODY = (0, 200, 100)  
COLOR_SNAKE_HEAD = (200, 255, 200)
COLOR_FOOD = (255, 50, 80)        
COLOR_TEXT = (255, 255, 255)
COLOR_ACCENT = (0, 255, 255)      
COLOR_BTN_ON = (0, 255, 100) 
COLOR_BTN_OFF = (200, 50, 50) 
COLOR_LIMIT = (255, 0, 50)
COLOR_HINT = (150, 150, 150)

class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4

Point = namedtuple('Point', 'x, y')
BLOCK_SIZE = 20

# --- CLASE PARA EFECTOS VISUALES ---
class Particle:
    def __init__(self, x, y, color):
        self.x = x
        self.y = y
        self.color = color
        self.vx = random.uniform(-3, 3)
        self.vy = random.uniform(-3, 3)
        self.life = random.randint(15, 30) 
        self.size = random.randint(2, 6)

    def update(self):
        self.x += self.vx
        self.y += self.vy
        self.life -= 1
        self.size -= 0.1

    def draw(self, surface):
        if self.life > 0 and self.size > 0:
            pygame.draw.circle(surface, self.color, (int(self.x), int(self.y)), int(self.size))

class SnakeGameAI:

    def __init__(self, w=640, h=480, control_mode='ai'):
        # Configuración de pantalla
        if w is None or h is None:
            info = pygame.display.Info()
            screen_w, screen_h = info.current_w, info.current_h
            self.w = int(screen_w * 0.9)
            self.h = int(screen_h * 0.9)
        else:
            self.w = w
            self.h = h

        self.w = (self.w // BLOCK_SIZE) * BLOCK_SIZE
        self.h = (self.h // BLOCK_SIZE) * BLOCK_SIZE
        
        self.display = pygame.display.set_mode((self.w, self.h)) 
        pygame.display.set_caption('Proyecto Snake Machine Learning')
        self.clock = pygame.time.Clock()
        
        # --- IMAGEN DE FONDO ---
        self.bg_image = None
        self._load_background()

        # Modo de control: 'ai' o 'human'
        self.control_mode = control_mode

        # Música
        self.music_on = True
        self.music_rect = pygame.Rect(self.w - 120, 10, 110, 30)
        self._load_and_play_music()

        # Variables de estado
        self.speed = 15       # VELOCIDAD NORMAL POR DEFECTO
        self.speed_text = "NORMAL"
        self.start_time = 0
        self.game_number = 0  

        # Partículas
        self.particles = []

        self.show_start_screen()
        self.reset()

    def _load_background(self):
        image_path = None
        if os.path.exists('fondo.png'): image_path = 'fondo.png'
        elif os.path.exists('fondo.jpg'): image_path = 'fondo.jpg'
        
        if image_path:
            try:
                img = pygame.image.load(image_path)
                self.bg_image = pygame.transform.scale(img, (self.w, self.h))
                self.dark_overlay = pygame.Surface((self.w, self.h))
                self.dark_overlay.set_alpha(180)
                self.dark_overlay.fill((0,0,0))
            except:
                self.bg_image = None

    def _draw_background(self):
        if self.bg_image:
            self.display.blit(self.bg_image, (0,0))
            self.display.blit(self.dark_overlay, (0,0)) 
        else:
            self.display.fill(COLOR_BG)

    def _load_and_play_music(self):
        if os.path.exists('musica.mp3'):
            try:
                pygame.mixer.music.load('musica.mp3')
                pygame.mixer.music.set_volume(0.3)
                pygame.mixer.music.play(-1)
            except:
                pass

    def toggle_music(self):
        self.music_on = not self.music_on
        if self.music_on:
            pygame.mixer.music.unpause()
        else:
            pygame.mixer.music.pause()

    def show_start_screen(self):
        waiting = True
        selected_mode = self.control_mode if hasattr(self, 'control_mode') else 'ai'
        while waiting:
            self._draw_background()
            center_x = self.w // 2
            center_y = self.h // 2

            title_text = font_title.render("PROYECTO SNAKE IA", True, COLOR_SNAKE_BODY)
            title_shadow = font_title.render("PROYECTO SNAKE IA", True, (0,0,0))
            
            title_rect = title_text.get_rect(center=(center_x, center_y - 80))
            self.display.blit(title_shadow, (title_rect.x + 3, title_rect.y + 3))
            self.display.blit(title_text, title_rect)
            
            name_text = font_subtitle.render("Optimizado con Red Neuronal - Emilio Jaramillo", True, COLOR_ACCENT)
            name_rect = name_text.get_rect(center=(center_x, center_y))
            self.display.blit(name_text, name_rect)
            
            # Opciones
            mode_ai_color = COLOR_ACCENT if selected_mode == 'ai' else COLOR_HINT
            mode_human_color = COLOR_ACCENT if selected_mode == 'human' else COLOR_HINT
            mode_ai_text = font_main.render("Modo IA (A)", True, mode_ai_color)
            mode_human_text = font_main.render("Modo Humano (H)", True, mode_human_color)
            ai_rect = mode_ai_text.get_rect(center=(center_x - 140, self.h - 140))
            human_rect = mode_human_text.get_rect(center=(center_x + 140, self.h - 140))
            self.display.blit(mode_ai_text, ai_rect)
            self.display.blit(mode_human_text, human_rect)

            start_text = font_main.render("[ ESPACIO: Iniciar ]", True, COLOR_FOOD)
            start_rect = start_text.get_rect(center=(center_x, self.h - 90))
            self.display.blit(start_text, start_rect)

            pygame.display.flip()
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    quit()
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        self.control_mode = selected_mode
                        waiting = False
                    if event.key == pygame.K_ESCAPE:
                        pygame.quit()
                        quit()
                    if event.key == pygame.K_a: selected_mode = 'ai'
                    if event.key == pygame.K_h: selected_mode = 'human'
                    if event.key in (pygame.K_LEFT, pygame.K_UP): selected_mode = 'ai'
                    if event.key in (pygame.K_RIGHT, pygame.K_DOWN): selected_mode = 'human'

    def reset(self):
        self.direction = Direction.RIGHT
        self.head = Point(self.w/2, self.h/2)
        self.snake = [self.head,
                      Point(self.head.x-BLOCK_SIZE, self.head.y),
                      Point(self.head.x-(2*BLOCK_SIZE), self.head.y)]
        self.score = 0
        self.food = None
        self._place_food()
        self.frame_iteration = 0
        self.start_time = time.time()
        

    def _place_food(self):
        cols = (self.w - BLOCK_SIZE) // BLOCK_SIZE
        rows = (self.h - BLOCK_SIZE) // BLOCK_SIZE
        if cols <= 0 or rows <= 0: return 
        x = random.randint(0, cols - 1) * BLOCK_SIZE
        y = random.randint(0, rows - 1) * BLOCK_SIZE
        self.food = Point(x, y)
        if self.food in self.snake:
            self._place_food()

    def show_game_over(self):
       
        crash_text = font_crash.render("GAME OVER", True, COLOR_BTN_OFF)
        crash_rect = crash_text.get_rect(center=(self.w/2, self.h/2))
        bg_rect = pygame.Rect(self.w/2 - 200, self.h/2 - 40, 400, 80)
        
        pygame.draw.rect(self.display, (0,0,0), bg_rect)
        pygame.draw.rect(self.display, COLOR_BTN_OFF, bg_rect, 2)
        self.display.blit(crash_text, crash_rect)
        pygame.display.flip()
        
        wait_time = 1000 if self.control_mode == 'human' else 100
        pygame.time.wait(wait_time)

  
    def play_step(self, action, game_gen=0):
        self.frame_iteration += 1
        self.game_number = game_gen 

        # Variables para resize seguro en este frame
        pending_resize = None
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
            # Ignorar eventos de redimensionado para mantener tamaño fijo
            if event.type == pygame.VIDEORESIZE:
                continue
            
            if event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    if self.music_rect.collidepoint(event.pos):
                        self.toggle_music()
            
            if event.type == pygame.KEYDOWN:
              
                if event.key == pygame.K_1:
                    self.speed = 10
                    self.speed_text = "LENTO"
                elif event.key == pygame.K_2:
                    self.speed = 15
                    self.speed_text = "NORMAL"
                elif event.key == pygame.K_3:
                    self.speed = 400
                    self.speed_text = "RAPIDO"
                elif event.key == pygame.K_F11:
                    pass  

            
                
                if self.control_mode == 'human':
                    if event.key == pygame.K_RIGHT and self.direction != Direction.LEFT:
                        self.direction = Direction.RIGHT
                    elif event.key == pygame.K_LEFT and self.direction != Direction.RIGHT:
                        self.direction = Direction.LEFT
                    elif event.key == pygame.K_UP and self.direction != Direction.DOWN:
                        self.direction = Direction.UP
                    elif event.key == pygame.K_DOWN and self.direction != Direction.UP:
                        self.direction = Direction.DOWN

                if event.key == pygame.K_ESCAPE:
                    self.show_start_screen() 
                    self.reset() 
                    return 0, False, 0 

       

        if self.control_mode == 'ai':
            self._move(action)
        else:
            x = self.head.x
            y = self.head.y
            if self.direction == Direction.RIGHT: x += BLOCK_SIZE
            elif self.direction == Direction.LEFT: x -= BLOCK_SIZE
            elif self.direction == Direction.DOWN: y += BLOCK_SIZE
            elif self.direction == Direction.UP: y -= BLOCK_SIZE
            self.head = Point(x, y)

        self.snake.insert(0, self.head)
        
        reward = 0
        game_over = False
        
        if self.is_collision() or self.frame_iteration > 100*len(self.snake):
            game_over = True
            reward = -10
            self._update_ui()
            self.show_game_over()
            return reward, game_over, self.score

        if self.head == self.food:
            self.score += 1
            reward = 10
            # --- EFECTO DE PARTÍCULAS ---
            for _ in range(12):
                self.particles.append(Particle(self.food.x + 10, self.food.y + 10, COLOR_FOOD))
            self._place_food()
        else:
            self.snake.pop()
        
        self._update_ui()
        self.clock.tick(self.speed)
        return reward, game_over, self.score

    def is_collision(self, pt=None):
        if pt is None:
            pt = self.head
        if pt.x > self.w - BLOCK_SIZE or pt.x < 0 or pt.y > self.h - BLOCK_SIZE or pt.y < 0:
            return True
        if pt in self.snake[1:]:
            return True
        return False

    def _update_ui(self):
        self._draw_background()
        
        # Grid
        for x in range(0, self.w, BLOCK_SIZE):
            pygame.draw.line(self.display, COLOR_GRID, (x, 0), (x, self.h))
        for y in range(0, self.h, BLOCK_SIZE):
            pygame.draw.line(self.display, COLOR_GRID, (0, y), (self.w, y))
        
        pygame.draw.rect(self.display, COLOR_LIMIT, pygame.Rect(0, 0, self.w, self.h), 4)

        # --- VISUALIZACIÓN IA ---
        if self.control_mode == 'ai':
            head_center = (self.head.x + BLOCK_SIZE//2, self.head.y + BLOCK_SIZE//2)
            if self.food:
                food_center = (self.food.x + BLOCK_SIZE//2, self.food.y + BLOCK_SIZE//2)
                # Línea a la comida 
                pygame.draw.line(self.display, (0, 100, 100), head_center, food_center, 1)

            # Sensores de peligro 
            dirs = [(0, BLOCK_SIZE), (0, -BLOCK_SIZE), (BLOCK_SIZE, 0), (-BLOCK_SIZE, 0)]
            for dx, dy in dirs:
                cx, cy = self.head.x + dx, self.head.y + dy
                is_danger = self.is_collision(Point(cx, cy))
                end_pos = (head_center[0] + dx, head_center[1] + dy)
                
                col = (255, 0, 0) if is_danger else (30, 30, 50)
                width = 3 if is_danger else 1
                pygame.draw.line(self.display, col, head_center, end_pos, width)

        # Partículas
        for p in self.particles[:]:
            p.update()
            p.draw(self.display)
            if p.life <= 0:
                self.particles.remove(p)

        # Serpiente
        for i, pt in enumerate(self.snake):
            rect = pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE)
            inner = pygame.Rect(pt.x+3, pt.y+3, BLOCK_SIZE-6, BLOCK_SIZE-6)
            pygame.draw.rect(self.display, (0, 140, 80), rect, border_radius=6)
            pygame.draw.rect(self.display, (0, 200, 100), inner, border_radius=6)
            if i == 0:
                cx, cy = pt.x + BLOCK_SIZE//2, pt.y + BLOCK_SIZE//2
                pygame.draw.circle(self.display, (255,255,255), (cx-4, cy-3), 3)
                pygame.draw.circle(self.display, (255,255,255), (cx+4, cy-3), 3)
                pygame.draw.circle(self.display, (0,0,0), (cx-4, cy-3), 1)
                pygame.draw.circle(self.display, (0,0,0), (cx+4, cy-3), 1)

        # Comida
        if self.food:
            apple = pygame.Surface((BLOCK_SIZE, BLOCK_SIZE), pygame.SRCALPHA)
            pygame.draw.circle(apple, COLOR_FOOD, (BLOCK_SIZE//2, BLOCK_SIZE//2), BLOCK_SIZE//2 - 3)
            pygame.draw.circle(apple, (160, 20, 40), (BLOCK_SIZE//2, BLOCK_SIZE//2), BLOCK_SIZE//2 - 3, 2)
            pygame.draw.polygon(apple, (0, 180, 80), [(BLOCK_SIZE//2 + 2, 4), (BLOCK_SIZE//2 + 10, 10), (BLOCK_SIZE//2 + 2, 12)])
            self.display.blit(apple, (self.food.x, self.food.y))

        # HUD
        text = font_main.render(f"Puntuación: {self.score}", True, COLOR_TEXT)
        self.display.blit(text, [10, 10])

        speed_surface = font_small.render(f"Vel: {self.speed_text} [1-2-3]", True, (150, 150, 200))
        self.display.blit(speed_surface, [10, 40])
        
        # Mostrar Generación
        gen_text = font_icon.render(f"Intento: {self.game_number}", True, COLOR_ACCENT)
        self.display.blit(gen_text, [10, 70])

        elapsed_time = int(time.time() - self.start_time)
        time_str = f"T: {elapsed_time // 60:02}:{elapsed_time % 60:02}"
        time_surface = font_small.render(time_str, True, (255, 255, 0))
        self.display.blit(time_surface, [10, 100])

        # Boton musica 
        btn_color = COLOR_BTN_ON if self.music_on else COLOR_BTN_OFF
        pygame.draw.rect(self.display, btn_color, self.music_rect, border_radius=8)
        pygame.draw.rect(self.display, (255,255,255), self.music_rect, 2, border_radius=8)
        btn_text = font_icon.render("MUSICA", True, (0,0,0))
        text_rect = btn_text.get_rect(center=self.music_rect.center)
        self.display.blit(btn_text, text_rect)

        pygame.display.flip()

    def _move(self, action):
        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = clock_wise.index(self.direction)

        if np.array_equal(action, [1, 0, 0]):
            new_dir = clock_wise[idx]
        elif np.array_equal(action, [0, 1, 0]):
            next_idx = (idx + 1) % 4
            new_dir = clock_wise[next_idx]
        else:
            next_idx = (idx - 1) % 4
            new_dir = clock_wise[next_idx]
        self.direction = new_dir
        
        x = self.head.x
        y = self.head.y
        if self.direction == Direction.RIGHT: x += BLOCK_SIZE
        elif self.direction == Direction.LEFT: x -= BLOCK_SIZE
        elif self.direction == Direction.DOWN: y += BLOCK_SIZE
        elif self.direction == Direction.UP: y -= BLOCK_SIZE
        self.head = Point(x, y)