"""
Python 3.9 программа битвы человек-машина
программа на Python по изучению обучения с подкреплением - Reinforcement Learning
Название файла connect4_aiplayer.py

Version: 0.1
Author: Andrej Marinchenko
Date: 2021-12-22
"""
import math  # подключаем библиотеку работы с математическими функциями
import random  # библиотека случайных значений
import sys  # работа с системными файлами

import pygame  # библиотека для разработки мультимедийных приложений, таких как видеоигры, с использованием Python

# подключаем собственные файлы
from utils import *  # программа утилит (обработки начального тестового датасета из 1000 игр, оценки результата игры)
from MCTS import MCTS  # программа дерева Монте-Карло
from Coach import Coach  # модель для сохранения, обучения и оценки
from parl.utils import logger  # логирование программы
from connect4_game import Connect4Game  # класс, реализующий общий интерфейс игры alpha-zero

# определяем цвета игры
BLUE = (0, 0, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
YELLOW = (255, 255, 0)

ROW_COUNT = 6  # количество строк
COLUMN_COUNT = 7  # количество колонок

PLAYER = 1  # обозначение игрока
AI = -1  # обозначение модели
EMPTY = 0  #

WINDOW_LENGTH = 4
SQUARESIZE = 100
RADIUS = int(SQUARESIZE / 2 - 5)

# функция нарисовать игровую доску
def draw_board(board, screen, height):
    for c in range(COLUMN_COUNT):
        for r in range(ROW_COUNT):
            pygame.draw.rect(screen, BLUE,
                             (c * SQUARESIZE, r * SQUARESIZE + SQUARESIZE,
                              SQUARESIZE, SQUARESIZE))
            pygame.draw.circle(screen, BLACK,
                               (int(c * SQUARESIZE + SQUARESIZE / 2),
                                int(r * SQUARESIZE + SQUARESIZE * 3 / 2)),
                               RADIUS)

    for c in range(COLUMN_COUNT):
        for r in range(ROW_COUNT):
            if board[r][c] == PLAYER:
                pygame.draw.circle(screen, RED,
                                   (int(c * SQUARESIZE + SQUARESIZE / 2),
                                    int(r * SQUARESIZE + SQUARESIZE * 3 / 2)),
                                   RADIUS)
            elif board[r][c] == AI:
                pygame.draw.circle(screen, YELLOW,
                                   (int(c * SQUARESIZE + SQUARESIZE / 2),
                                    int(r * SQUARESIZE + SQUARESIZE * 3 / 2)),
                                   RADIUS)
    pygame.display.update()


game = Connect4Game()  # класс, реализующий общий интерфейс игры alpha-zero из файла connect4_game.py
board = game._base_board  # задаем доску
game_over = False  # переменная - игра не окончена
current_board = board.np_pieces  # текущая доска - доска с комнями
game.display(current_board)  # отобразить текущую доску

pygame.init()  # инициализация доски

width = COLUMN_COUNT * SQUARESIZE
height = (ROW_COUNT + 1) * SQUARESIZE
size = (width, height)
screen = pygame.display.set_mode(size)
draw_board(current_board, screen, height)  # функция нарисовать игровую доску
my_font = pygame.font.SysFont('monospace', 75)
turn = random.choice([PLAYER, AI])

# получаем стартовые аргументы из лучшей сохраненной модели
args = dotdict({
    'load_folder_file': ('/home/user/PycharmProjects/Kaggle_ConnectX/saved_model', 'best.pth.tar'),
    # 'load_folder_file': ('./saved_model', 'best_model'),
    'numMCTSSims': 800,  # Количество игровых ходов для моделирования MCTS
    'cpuct': 4,
})
logger.info('Загрузка контрольной точки {}...'.format(args.load_folder_file))
c = Coach(game, args)
c.loadModel()  # считываем модель
agent = c.current_agent  # подключаем агента обученной модели
mcts = MCTS(game, agent, args)  # получаем дерево монте-карло

while not game_over:  # пока игра не закончится
    for event in pygame.event.get():
        if event.type == pygame.QUIT:   # если выбрано - выход
            sys.exit()  # завершить игру

        if event.type == pygame.MOUSEMOTION:  # если совершено движение мышкой
            pygame.draw.rect(screen, BLACK, (0, 0, width, SQUARESIZE))
            posx = event.pos[0]
            if turn == PLAYER:
                pygame.draw.circle(screen, RED, (posx, int(SQUARESIZE / 2)),
                                   RADIUS)

        pygame.display.update()  # обновить экран

        if event.type == pygame.MOUSEBUTTONDOWN:  # нажатие мышкой
            pygame.draw.rect(screen, BLACK, (0, 0, width, SQUARESIZE))

            if turn == PLAYER:
                posx = event.pos[0]
                col = int(math.floor(posx / SQUARESIZE))

                if board.is_valid_move(col):
                    current_board, _ = game.getNextState(
                        current_board, turn, col)

                    if game.getGameEnded(current_board, PLAYER) == 1:
                        # label = my_font.render('You win !!!', 1, RED)
                        label = my_font.render('Ты выйграл !!!', 1, RED)
                        screen.blit(label, (40, 10))
                        game_over = True
                    elif game.getGameEnded(current_board, PLAYER) == 1e-4:
                        label = my_font.render('Ничья !!!', 1, RED)
                        screen.blit(label, (40, 10))
                        game_over = True
                    else:
                        turn = -turn

                    game.display(current_board)
                    draw_board(current_board, screen, height)

        if turn == AI and not game_over:
            x = game.getCanonicalForm(current_board, turn)
            col = int(np.argmax(mcts.getActionProb(x, temp=0)))
            col = np.argmax(pi)  # pi: вектор политики размера self.getActionSize()
            if board.is_valid_move(col):
                current_board, _ = game.getNextState(current_board, turn, col)
                if game.getGameEnded(current_board, PLAYER) == -1:
                    # label = my_font.render('You lose !!!', 1, RED)
                    label = my_font.render('Ты проиграл !!!', 1, RED)
                    screen.blit(label, (40, 10))
                    game_over = True
                elif game.getGameEnded(current_board, PLAYER) == 1e-4:
                    label = my_font.render('Ничья !!!', 1, RED)
                    screen.blit(label, (40, 10))
                    game_over = True
                else:
                    turn = -turn

                game.display(current_board)
                draw_board(current_board, screen, height)

    if game_over:
        pygame.time.wait(5000)

