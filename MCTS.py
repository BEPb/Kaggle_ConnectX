"""
Python 3.9 программа дерева Монте-Карло на основе нейронной сети
программа на Python по изучению обучения с подкреплением - Reinforcement Learning
Название файла actor.py

Version: 0.1
Author: Andrej Marinchenko
Date: 2021-12-23
"""

import math  # библиотека математических функций
import time  # библиотека временных функций

import numpy as np  # обработка массивов данных

EPS = 1e-8


class MCTS():
    """
    Этот класс обрабатывает дерево MCTS.
    Принимает экземпляр Board и ключевые аргументы. Инициализирует список состояний игры и таблицы статистики.
    """

    def __init__(self, game, nn_agent, args, dirichlet_noise=False):  # инициализация класса
        self.game = game  # состояние игры
        self.nn_agent = nn_agent  # состояние нейронной сети
        self.args = args  # аргументы заданные в main.py
        self.dirichlet_noise = dirichlet_noise  # шум Дирихле
        self.Qsa = {}  # Ожидаемое вознаграждение за выполнение действия 'a' в состоянии 's'. Начальное значение Q(s,а) = 0
        self.Nsa = {}  # Количество раз, когда действие 'a' было выполнено в состоянии 's'
        self.Ns = {}  # Количество раз в состоянии 's'
        self.Ps = {}  # Вероятность действия в состоянии 's', значение стратегии, предсказанное нейронной сетью

        self.Es = {}  # game.getGameEnded получить завершение игры для доски s
        self.Vs = {}  # game.getValidMoves получить действительные ходы для доски s

    def getActionProb(self, canonicalBoard, temp=1):
        """
        Эта функция выполняет numMCTSSims-симуляции MCTS, начиная с
         canonicalBoard.
         Возврат: probs: вектор политики, в котором вероятность i-го действия равна пропорционально Nsa [(s, a)] ** (1./temp)
        """
        for i in range(self.args.numMCTSSims):
            dir_noise = (i == 0 and self.dirichlet_noise)
            self.search(canonicalBoard, dirichlet_noise=dir_noise)

        s = self.game.stringRepresentation(canonicalBoard)
        counts = [
            self.Nsa[(s, a)] if (s, a) in self.Nsa else 0
            for a in range(self.game.getActionSize())
        ]

        if temp == 0:
            bestAs = np.array(np.argwhere(counts == np.max(counts))).flatten()
            bestA = np.random.choice(bestAs)
            probs = [0] * len(counts)
            probs[bestA] = 1
            return probs

        counts = [x**(1. / temp) for x in counts]
        counts_sum = float(sum(counts))
        probs = [x / counts_sum for x in counts]
        return probs

    def search(self, canonicalBoard, dirichlet_noise=False):
        """
        Эта функция выполняет одну итерацию MCTS. Это рекурсивно называется
         пока не будет найден листовой узел. На каждом узле выбирается действие, которое
         имеет максимальную верхнюю доверительную границу.

         Как только листовой узел найден, вызывается нейронная сеть, чтобы вернуть
         начальная политика P и значение v для состояния. Это значение распространяется
         вверх по пути поиска. В случае, если листовой узел является конечным состоянием,
         результат распространяется вверх по пути поиска. Значения Ns, Nsa, Qsa равны
         обновлено.

         ПРИМЕЧАНИЕ: возвращаемые значения являются отрицательными по отношению к текущему значению.
         государство. Это сделано, поскольку v находится в [-1,1] и если v - значение a
         state для текущего игрока, тогда его значение равно -v для другого игрока.

         Возврат: v: отрицательное значение текущего canonicalBoard
        """

        s = self.game.stringRepresentation(canonicalBoard)

        if s not in self.Es:
            self.Es[s] = self.game.getGameEnded(canonicalBoard, 1)
        if self.Es[s] != 0:
            # terminal node
            return -self.Es[s]

        if s not in self.Ps:
            # leaf node
            self.Ps[s], v = self.nn_agent.predict(canonicalBoard)

            valids = self.game.getValidMoves(canonicalBoard, 1)
            self.Ps[s] = self.Ps[s] * valids  # masking invalid moves
            if dirichlet_noise:
                self.applyDirNoise(s, valids)
            sum_Ps_s = np.sum(self.Ps[s])
            if sum_Ps_s > 0:
                self.Ps[s] /= sum_Ps_s  # renormalize
            else:
                '''
                если все допустимые ходы были замаскированы, сделать все допустимые ходы равновероятными
                NB! Все допустимые ходы могут быть замаскированы, если либо ваша архитектура NNet недостаточна, 
                либо у вас есть переоснащение или что-то еще. 
                Если у вас есть десятки или сотни таких сообщений, вам следует обратить внимание на вашу NNet и / 
                или процесс обучения.
                '''

                print("Все допустимые ходы были замаскированы, что позволяло обходное решение.")
                self.Ps[s] = self.Ps[s] + valids
                self.Ps[s] /= np.sum(self.Ps[s])

            self.Vs[s] = valids
            self.Ns[s] = 0
            return -v

        valids = self.Vs[s]
        if dirichlet_noise:
            self.applyDirNoise(s, valids)
            sum_Ps_s = np.sum(self.Ps[s])
            self.Ps[s] /= sum_Ps_s  # renormalize
        cur_best = -float('inf')
        best_act = -1

        # выбрать действие с наивысшей верхней границей уверенности
        for a in range(self.game.getActionSize()):
            if valids[a]:
                if (s, a) in self.Qsa:
                    u = self.Qsa[
                        (s, a)] + self.args.cpuct * self.Ps[s][a] * math.sqrt(
                            self.Ns[s]) / (1 + self.Nsa[(s, a)])
                else:
                    u = self.args.cpuct * self.Ps[s][a] * math.sqrt(
                        self.Ns[s] + EPS)  # Q = 0 ?

                if u > cur_best:
                    cur_best = u
                    best_act = a

        a = best_act
        next_s, next_player = self.game.getNextState(canonicalBoard, 1, a)
        next_s = self.game.getCanonicalForm(next_s, next_player)

        v = self.search(next_s)

        if (s, a) in self.Qsa:
            self.Qsa[(s, a)] = (self.Nsa[(s, a)] * self.Qsa[
                (s, a)] + v) / (self.Nsa[(s, a)] + 1)
            self.Nsa[(s, a)] += 1

        else:
            self.Qsa[(s, a)] = v
            self.Nsa[(s, a)] = 1

        self.Ns[s] += 1
        return -v

    def applyDirNoise(self, s, valids):
        dir_values = np.random.dirichlet(
            [self.args.dirichletAlpha] * np.count_nonzero(valids))
        dir_idx = 0
        for idx in range(len(self.Ps[s])):
            if self.Ps[s][idx]:
                self.Ps[s][idx] = (0.75 * self.Ps[s][idx]) + (
                    0.25 * dir_values[dir_idx])
                dir_idx += 1
